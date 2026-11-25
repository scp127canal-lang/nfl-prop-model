"""Season-by-season walk-forward backtest -> DuckDB ``backtest_results`` / ``backtest_bets``.

For every evaluation season Y: train on seasons ``train_start .. Y-1`` only, then predict and
settle every season-Y game. Features are already point-in-time, so in-season information is
limited to what was known before each game.

Price sources
-------------
* **schedules closing lines** (moneyline, spread, total with odds; nflverse, one book). Bets are
  placed at the close, so CLV is undefined (NULL) for these rows.
* **raw_odds snapshots** (Odds API), when present: any game market or player prop. The bet price
  is the first snapshot (``bet_snapshot="first"``) or the last one; the closing price is the last
  snapshot before kickoff. CLV = EV of the bet price at the devigged closing probability. Player
  names go through ``entity_resolution`` to ``gsis_id`` (rule 4); unmatched names are skipped and
  counted.

Model probabilities: game markets come from the score simulation of the season's game model;
player yardage/count props from ``simulate_game`` (game model trained < Y); anytime/first TD
from the calibrated TD models trained < Y. Every comparison uses devigged fair probabilities
(rule 5, sharp books first).

Metrics per (season, market_category) and overall: Brier & log-loss (model vs devigged market),
#bets, staked, profit, ROI (fractional Kelly and flat 1u), hit rate, mean edge/EV and CLV.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import duckdb
import numpy as np
import pandas as pd

from src.evaluation import devig as dv
from src.models import game_sim as gs
from src.models import td_model as tdm

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "nfl_model.duckdb"
EPS = 1e-6

# player_weekly columns used to settle props
SETTLE_COLUMNS = {"pass_yds": "passing_yards", "pass_tds": "passing_tds", "pass_completions": "completions",
                  "pass_attempts": "attempts", "pass_interceptions": "passing_interceptions",
                  "rush_yds": "rushing_yards", "rush_attempts": "carries", "reception_yds": "receiving_yards",
                  "receptions": "receptions"}


@dataclass
class BacktestConfig:
    eval_seasons: Sequence[int] | None = None   # default: every completed season after train_start
    train_start: int = 2018
    n_trials: int = 20_000
    prop_trials: int = 10_000
    devig_method: str = "shin"
    sharp_books: Sequence[str] = dv.SHARP_BOOKS
    min_edge_pct: float = 2.0                   # model - fair, percentage points
    min_ev_pct: float = 0.0
    kelly: float = dv.DEFAULT_KELLY_FRACTION
    bankroll: float = dv.DEFAULT_BANKROLL
    max_stake_pct: float = dv.DEFAULT_MAX_STAKE_PCT
    game_model_kind: str = "ridge"
    include_td: bool = True
    include_props: bool = True
    td_params: dict[str, Any] | None = None
    td_rounds: int = 400
    bet_snapshot: str = "first"
    seed: int = 7
    notes: str = ""


def market_category(market_key: str) -> str:
    if market_key == "h2h":
        return "moneyline"
    if market_key in ("spreads", "alternate_spreads"):
        return "spread"
    if market_key in ("totals", "alternate_totals"):
        return "total"
    if market_key in ("team_totals", "alternate_team_totals"):
        return "team_total"
    if market_key == "player_anytime_td":
        return "player_anytime_td"
    if market_key in ("player_1st_td", "player_last_td"):
        return market_key.replace("1st", "first")
    if market_key.startswith("player_pass"):
        return "player_passing"
    if market_key.startswith("player_rush"):
        return "player_rushing"
    if market_key.startswith("player_reception") or market_key == "player_receptions":
        return "player_receiving"
    return "player_other"


# ======================================================================================
# Probability helpers
# ======================================================================================
def p_over(x: np.ndarray, line: float) -> tuple[float, float]:
    """(P(X > line), P(X == line)) with mild smoothing so 0 < p < 1."""
    n = len(x)
    return (float((x > line).sum()) + 0.5) / (n + 1), float((x == line).sum()) / n


def team_params_from_model(model: gs.GameModel, rows: pd.DataFrame) -> dict[str, gs.TeamParams]:
    pred = model.predict(rows)
    out = {}
    for _, r in pred.iterrows():
        out[r["team"]] = gs.TeamParams(
            team=r["team"], pred_points=float(r["pred_points"]), pred_plays=float(r["pred_plays"]),
            pred_pass_rate=float(r["pred_pass_rate"]), sd_plays=model.resid["sd_plays"],
            sd_logit_pass_rate=model.resid["sd_logit_pass_rate"], beta_plays=model.resid["beta_plays"],
            beta_pass_rate=model.resid["beta_pass_rate"], sack_rate=model.league["sack_rate"],
            scramble_rate=model.league["scramble_rate"], target_rate=model.league["target_rate"],
            int_rate=model.league["int_rate"], pass_td_share=model.league["pass_td_share"],
            sd_points=model.resid.get("sd_points"))
    return out


def game_market_prob(home: np.ndarray, away: np.ndarray, market_key: str, outcome: str, point: float | None,
                     home_team: str, away_team: str, description: str | None = None) -> tuple[float, float]:
    """(P(win), P(push)) for a game-level selection from simulated scores."""
    if market_key == "h2h":
        mine, theirs = (home, away) if outcome == home_team else (away, home)
        win = (mine > theirs).mean() + 0.5 * (mine == theirs).mean()  # sim ties ~ OT coin flip
        return float(np.clip(win, EPS, 1 - EPS)), 0.0
    if market_key in ("spreads", "alternate_spreads"):
        margin = (home - away) if outcome == home_team else (away - home)
        return p_over(margin + point, 0.0)
    if market_key in ("totals", "alternate_totals", "team_totals", "alternate_team_totals"):
        x = home + away
        if market_key in ("team_totals", "alternate_team_totals"):
            x = home if description == home_team else away
        over_p, push = p_over(x, point)
        return (over_p, push) if outcome == "Over" else (max(1 - over_p - push, EPS), push)
    raise ValueError(f"not a game market: {market_key}")


def _settle(actual: float, line: float, side: str) -> str:
    if actual == line:
        return "push"
    won = actual > line if side == "Over" else actual < line
    return "win" if won else "loss"


# ======================================================================================
# Candidate builders
# ======================================================================================
def schedule_candidates(game: pd.Series, home: np.ndarray, away: np.ndarray, method: str) -> list[dict]:
    """Closing-line candidates from nflverse schedules for one game (both sides of each market)."""
    out = []
    h, a = game["home_team"], game["away_team"]
    margin = game["home_score"] - game["away_score"]
    total = game["home_score"] + game["away_score"]
    markets = [
        ("h2h", [(h, None, game.get("home_moneyline")), (a, None, game.get("away_moneyline"))]),
        ("spreads", [(h, -game["spread_line"], game.get("home_spread_odds")),
                     (a, game["spread_line"], game.get("away_spread_odds"))]),
        ("totals", [("Over", game["total_line"], game.get("over_odds")),
                    ("Under", game["total_line"], game.get("under_odds"))]),
    ]
    for mkey, sides in markets:
        prices = [s[2] for s in sides]
        if any(p is None or pd.isna(p) or p == 0 for p in prices) or any(
                s[1] is not None and pd.isna(s[1]) for s in sides):
            continue
        dec = dv.american_to_decimal(prices)
        fair = dv.devig(dec, method)
        for (name, point, _), d, f in zip(sides, dec, fair):
            p_win, p_push = game_market_prob(home, away, mkey, name, point, h, a)
            if mkey == "h2h":
                outcome = "push" if margin == 0 else ("win" if (margin > 0) == (name == h) else "loss")
            elif mkey == "spreads":
                m = margin if name == h else -margin
                outcome = _settle(m + point, 0.0, "Over")
            else:
                outcome = _settle(total, point, name)
            out.append({"game_id": game["game_id"], "market_key": mkey, "selection": name, "line": point,
                        "player_id": None, "bet_book": "nflverse_close", "bet_decimal": float(d),
                        "fair_prob": float(f), "close_fair_prob": np.nan, "model_p_win": p_win,
                        "model_p_push": p_push, "outcome": outcome, "price_source": "schedules_close"})
    return out


def map_events_to_games(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    from src.utils.entity_resolution import team_to_abbr

    ev = con.execute("""SELECT DISTINCT event_id, home_team, away_team, CAST(commence_time_utc AS DATE) AS d
                        FROM raw_odds""").df()
    if ev.empty:
        return ev.assign(game_id=pd.Series(dtype=str))
    sched = con.execute("SELECT game_id, home_team, away_team, game_date FROM schedules").df()
    ev["h"], ev["a"] = ev["home_team"].map(team_to_abbr), ev["away_team"].map(team_to_abbr)
    m = ev.merge(sched, left_on=["h", "a"], right_on=["home_team", "away_team"], suffixes=("", "_s"))
    m = m[(pd.to_datetime(m["game_date"]) - pd.to_datetime(m["d"])).abs() <= pd.Timedelta(days=1)]
    return m[["event_id", "game_id"]].drop_duplicates("event_id")


def raw_odds_snapshots(con: duckdb.DuckDBPyConnection, game_ids: Sequence[str], bet_snapshot: str
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(bet-time rows, closing rows) from raw_odds for the given games (pre-kickoff snapshots only)."""
    if not game_ids or not gs_table_exists(con, "raw_odds"):
        return pd.DataFrame(), pd.DataFrame()
    emap = map_events_to_games(con)
    emap = emap[emap["game_id"].isin(game_ids)]
    if emap.empty:
        return pd.DataFrame(), pd.DataFrame()
    con.register("_emap", emap)
    df = con.execute("""SELECT r.*, e.game_id FROM raw_odds r JOIN _emap e USING (event_id)
                        WHERE r.snapshot_ts_utc < r.commence_time_utc""").df()
    con.unregister("_emap")
    if df.empty:
        return df, df
    snaps = df.groupby("event_id")["snapshot_ts_utc"].agg(["min", "max"])
    first = df["snapshot_ts_utc"] == df["event_id"].map(snaps["min" if bet_snapshot == "first" else "max"])
    last = df["snapshot_ts_utc"] == df["event_id"].map(snaps["max"])
    return df[first].copy(), df[last].copy()


def gs_table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                            [table]).fetchall()[0][0])


def raw_odds_candidates(con: duckdb.DuckDBPyConnection, season_games: pd.DataFrame, score_sims: dict,
                        prop_sims: dict, td_probs: pd.DataFrame | None, actuals: pd.DataFrame,
                        cfg: BacktestConfig, stats: dict[str, int]) -> list[dict]:
    from src.utils.entity_resolution import PlayerResolver

    bet_rows, close_rows = raw_odds_snapshots(con, season_games["game_id"].tolist(), cfg.bet_snapshot)
    if bet_rows.empty:
        return []
    fair_bet = dv.fair_probabilities(bet_rows, cfg.devig_method, cfg.sharp_books)
    fair_close = dv.fair_probabilities(close_rows, cfg.devig_method, cfg.sharp_books)
    close_map = fair_close.set_index("selection_key")["fair_prob"] if not fair_close.empty else pd.Series(dtype=float)
    gmap = bet_rows.drop_duplicates("event_id").set_index("event_id")["game_id"]
    games = season_games.set_index("game_id")
    resolver = PlayerResolver(con) if fair_bet["market_key"].str.startswith("player_").any() else None
    out = []
    for _, r in fair_bet.iterrows():
        gid = gmap[r["event_id"]]
        g = games.loc[gid]
        mk, name, point = r["market_key"], r["outcome_name"], r["outcome_point"]
        point = None if pd.isna(point) else float(point)
        base = {"game_id": gid, "market_key": mk, "line": point, "bet_book": r["best_book"],
                "bet_decimal": r["best_price_decimal"], "fair_prob": r["fair_prob"],
                "close_fair_prob": float(close_map.get(r["selection_key"], np.nan)), "price_source": "raw_odds"}
        if not mk.startswith("player_"):
            if gid not in score_sims:
                continue
            home, away = score_sims[gid]
            p_win, p_push = game_market_prob(home, away, mk, name, point, g["home_team_full"], g["away_team_full"],
                                             r.get("outcome_description"))
            hs, as_ = g["home_score"], g["away_score"]
            if mk == "h2h":
                mine = hs if name == g["home_team_full"] else as_
                theirs = as_ if name == g["home_team_full"] else hs
                outcome = "push" if mine == theirs else ("win" if mine > theirs else "loss")
            elif mk in ("spreads", "alternate_spreads"):
                m = (hs - as_) if name == g["home_team_full"] else (as_ - hs)
                outcome = _settle(m + point, 0.0, "Over")
            else:
                x = hs + as_ if mk in ("totals", "alternate_totals") else (
                    hs if r.get("outcome_description") == g["home_team_full"] else as_)
                outcome = _settle(x, point, name)
            out.append({**base, "selection": name if mk != "team_totals" else f"{r['outcome_description']} {name}",
                        "player_id": None, "model_p_win": p_win, "model_p_push": p_push, "outcome": outcome})
            continue
        # ---- player props: entity resolution first (rule 4) ----
        if not cfg.include_props or actuals.empty:
            stats["unsupported_prop"] += 1
            continue
        res = resolver.resolve(r["outcome_description"], [g["home_team_full"], g["away_team_full"]], mk)
        if not res.matched:
            stats["unmatched_players"] += 1
            continue
        pid = res.gsis_id
        act = actuals[(actuals["game_id"] == gid) & (actuals["gsis_id"] == pid)]
        if act.empty:
            stats["void_not_played"] += 1
            continue
        act = act.iloc[0]
        if mk in ("player_anytime_td", "player_1st_td"):
            target = "anytime_td" if mk == "player_anytime_td" else "first_td"
            tp = None if td_probs is None else td_probs[(td_probs["game_id"] == gid) & (td_probs["gsis_id"] == pid)]
            if tp is not None and len(tp):
                p_yes = float(tp[f"p_{target}"].iloc[0])
            elif gid in prop_sims and pid in prop_sims[gid].stats:
                p_yes = prop_sims[gid].prob_at_least(pid, target, 1)
            else:
                continue
            yes = bool(act[target] > 0)
            is_yes = name == "Yes"
            out.append({**base, "selection": f"{r['outcome_description']} {name}", "player_id": pid,
                        "model_p_win": p_yes if is_yes else 1 - p_yes, "model_p_push": 0.0,
                        "outcome": "win" if yes == is_yes else "loss"})
            continue
        stat = gs.MARKET_STATS.get(mk)
        col = SETTLE_COLUMNS.get(stat or "")
        if col is None or gid not in prop_sims or pid not in prop_sims[gid].stats or point is None:
            stats["unsupported_prop"] += 1
            continue
        x = prop_sims[gid].get(pid, stat)
        over_p, push = p_over(x, point)
        p_win = over_p if name == "Over" else max(1 - over_p - push, EPS)
        out.append({**base, "selection": f"{r['outcome_description']} {name}", "player_id": pid,
                    "model_p_win": p_win, "model_p_push": push, "outcome": _settle(float(act[col]), point, name)})
    return out


# ======================================================================================
# Scoring & summary
# ======================================================================================
def score_candidates(c: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    c = c.copy()
    c["market_category"] = c["market_key"].map(market_category)
    p_push = c["model_p_push"].clip(0, 0.999)
    c["model_prob"] = (c["model_p_win"] / (1 - p_push)).clip(EPS, 1 - EPS)  # conditional on no push
    c["ev_pct"] = dv.ev_pct(c["model_p_win"], c["bet_decimal"], p_push)
    c["edge_pct"] = dv.edge_pct(c["model_prob"], c["fair_prob"])
    c["clv_pct"] = np.where(c["close_fair_prob"].notna(), dv.clv_pct(c["bet_decimal"], c["close_fair_prob"]), np.nan)
    c["is_bet"] = (c["edge_pct"] >= cfg.min_edge_pct) & (c["ev_pct"] > cfg.min_ev_pct)
    c["stake"] = np.where(c["is_bet"], dv.kelly_stake(c["model_p_win"], c["bet_decimal"], cfg.bankroll, cfg.kelly,
                                                      p_push, cfg.max_stake_pct), 0.0)
    c.loc[c["stake"] <= 0, "is_bet"] = False
    payout = np.select([c["outcome"] == "win", c["outcome"] == "loss"], [c["bet_decimal"] - 1.0, -1.0], 0.0)
    c["profit"] = c["stake"] * payout
    c["flat_stake"] = c["is_bet"].astype(float)
    c["flat_profit"] = c["flat_stake"] * payout
    c["y"] = np.where(c["outcome"] == "win", 1.0, np.where(c["outcome"] == "loss", 0.0, np.nan))
    return c


def _brier(p, y):
    return float(np.mean((p - y) ** 2)) if len(y) else np.nan


def _logloss(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(y) else np.nan


def summarize(c: pd.DataFrame, extra_calibration: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    groups = [((s, m), g) for (s, m), g in c.groupby(["season", "market_category"])] + \
             [(("ALL", m), g) for m, g in c.groupby("market_category")]
    for (season, cat), g in groups:
        s = g[g["y"].notna()]
        bets = g[g["is_bet"]]
        settled = bets[bets["outcome"] != "push"]
        clv = bets["clv_pct"].dropna()
        staked = bets["stake"].sum()
        rows.append({
            "season": None if season == "ALL" else int(season), "scope": "all" if season == "ALL" else "season",
            "market_category": cat, "n_predictions": len(s),
            "brier_model": _brier(s["model_prob"], s["y"]), "brier_market": _brier(s["fair_prob"], s["y"]),
            "logloss_model": _logloss(s["model_prob"], s["y"]), "logloss_market": _logloss(s["fair_prob"], s["y"]),
            "n_bets": len(bets), "staked": float(staked), "profit": float(bets["profit"].sum()),
            "roi_pct": float(100 * bets["profit"].sum() / staked) if staked > 0 else np.nan,
            "flat_roi_pct": float(100 * bets["flat_profit"].sum() / len(bets)) if len(bets) else np.nan,
            "hit_rate": float((settled["outcome"] == "win").mean()) if len(settled) else np.nan,
            "avg_edge_pct": float(bets["edge_pct"].mean()) if len(bets) else np.nan,
            "avg_ev_pct": float(bets["ev_pct"].mean()) if len(bets) else np.nan,
            "n_clv": len(clv), "avg_clv_pct": float(clv.mean()) if len(clv) else np.nan,
            "clv_beat_rate": float((clv > 0).mean()) if len(clv) else np.nan,
            "trained_through": g["trained_through"].max() if "trained_through" in g else None,
        })
    out = pd.DataFrame(rows)
    if extra_calibration is not None and len(extra_calibration):
        out = pd.concat([out, extra_calibration], ignore_index=True)
    return out


def td_calibration_rows(td_df: pd.DataFrame, season: int, models: dict[str, tdm.CalibratedTDModel]) -> list[dict]:
    rows = []
    d = td_df[(td_df["season"] == season) & (td_df["played"] == 1)]
    for target, m in models.items():
        p = m.predict_proba(d)
        y = d[target].to_numpy(float)
        base = np.full_like(y, m.meta["base_rate"])
        rows.append({"season": season, "scope": "season", "market_category": f"model_{target}",
                     "n_predictions": len(d), "brier_model": _brier(p, y), "brier_market": _brier(base, y),
                     "logloss_model": _logloss(p, y), "logloss_market": _logloss(base, y), "n_bets": 0,
                     "trained_through": m.meta["trained_through"]})
    return rows


# ======================================================================================
# Runner
# ======================================================================================
def run_backtest(con: duckdb.DuckDBPyConnection, cfg: BacktestConfig | None = None,
                 write: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or BacktestConfig()
    run_id = uuid.uuid4().hex[:12]
    gd = gs.build_game_dataset(con)
    sched = con.execute("SELECT * FROM schedules").df()
    from src.utils.entity_resolution import TEAM_NAME_TO_ABBR

    abbr_to_name = {v: k for k, v in TEAM_NAME_TO_ABBR.items()}
    sched["home_team_full"] = sched["home_team"].map(abbr_to_name).fillna(sched["home_team"])
    sched["away_team_full"] = sched["away_team"].map(abbr_to_name).fillna(sched["away_team"])
    played = sched.dropna(subset=["home_score", "away_score"])
    seasons = cfg.eval_seasons or sorted(s for s in played["season"].unique() if s > cfg.train_start
                                         and played[played["season"] == s]["week"].max() >= 17)
    has_raw = gs_table_exists(con, "raw_odds") and con.execute("SELECT COUNT(*) FROM raw_odds").fetchall()[0][0] > 0
    td_df = tdm.build_td_dataset(con) if cfg.include_td else None
    actuals = pd.DataFrame()
    if has_raw and cfg.include_props:
        actuals = con.execute("SELECT * FROM player_weekly").df()
        if td_df is not None:
            actuals = actuals.merge(td_df[td_df["played"] == 1][["game_id", "gsis_id", "anytime_td", "first_td"]],
                                    on=["game_id", "gsis_id"], how="left").fillna({"anytime_td": 0, "first_td": 0})

    rng = np.random.default_rng(cfg.seed)
    all_cands, calib_rows = [], []
    counters = {"unmatched_players": 0, "void_not_played": 0, "unsupported_prop": 0}
    for season in seasons:
        train = gd[(gd["season"] < season) & (gd["season"] >= cfg.train_start)]
        model = gs.GameModel(kind=cfg.game_model_kind).fit(train)
        trained_through = str(model.trained_through.date())
        season_games = played[played["season"] == season]
        score_sims, cands = {}, []
        for _, g in season_games.iterrows():
            rows = gd[gd["game_id"] == g["game_id"]]
            if len(rows) != 2 or rows["implied_points"].isna().any():
                continue
            params = team_params_from_model(model, rows)
            _, _, pts = gs.simulate_points(params, cfg.n_trials, rng)
            home, away = pts[g["home_team"]], pts[g["away_team"]]
            score_sims[g["game_id"]] = (home, away)
            cands += schedule_candidates(g, home, away, cfg.devig_method)

        td_models, td_probs = {}, None
        if cfg.include_td and td_df is not None:
            start = pd.to_datetime(td_df.loc[td_df["season"] == season, "game_date"]).min()
            try:
                td_models = {t: tdm.train_td_model(td_df[td_df["season"] >= cfg.train_start], t, before=start,
                                                   params=cfg.td_params, num_boost_round=cfg.td_rounds)
                             for t in tdm.TARGETS}
                calib_rows += td_calibration_rows(td_df, season, td_models)
                sd = td_df[(td_df["season"] == season) & td_df["eligible"]].copy()
                for t, m in td_models.items():
                    sd[f"p_{t}"] = m.predict_proba(sd)
                td_probs = sd[["game_id", "gsis_id", "p_anytime_td", "p_first_td"]]
            except ValueError as exc:
                logger.warning("TD models skipped for %s: %s", season, exc)

        if has_raw:
            prop_sims = {}
            if cfg.include_props:
                emap = map_events_to_games(con)
                prop_events = con.execute("SELECT DISTINCT event_id FROM raw_odds WHERE is_player_prop").df()
                for gid in emap[emap["event_id"].isin(prop_events["event_id"])]["game_id"]:
                    if gid in set(season_games["game_id"]):
                        try:
                            prop_sims[gid] = gs.simulate_game(gid, con=con, model=model,
                                                              n_trials=cfg.prop_trials, seed=cfg.seed)
                        except ValueError as exc:
                            logger.warning("prop sim skipped for %s: %s", gid, exc)
            cands += raw_odds_candidates(con, season_games, score_sims, prop_sims, td_probs, actuals, cfg, counters)

        c = pd.DataFrame(cands)
        if c.empty:
            continue
        c["season"], c["trained_through"] = season, trained_through
        all_cands.append(c)
        logger.info("season %s: %d candidates (trained through %s)", season, len(c), trained_through)

    if not all_cands:
        raise ValueError("No backtest candidates: need completed seasons with market lines")
    bets = score_candidates(pd.concat(all_cands, ignore_index=True), cfg)
    calib = pd.DataFrame(calib_rows)
    if len(calib):
        overall = calib.groupby("market_category").apply(
            lambda g: pd.Series({"n_predictions": g["n_predictions"].sum(),
                                 **{k: np.average(g[k], weights=g["n_predictions"]) for k in
                                    ("brier_model", "brier_market", "logloss_model", "logloss_market")}}),
            include_groups=False).reset_index().assign(season=None, scope="all", n_bets=0)
        calib = pd.concat([calib, overall], ignore_index=True)
    summary = summarize(bets, calib)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    meta = {"run_id": run_id, "created_at_utc": now, "config": json.dumps(asdict(cfg), default=list),
            **{f"n_{k}": v for k, v in counters.items()}}
    summary = summary.assign(**meta)
    bets = bets.assign(run_id=run_id, created_at_utc=now)
    if write:
        _append(con, "backtest_results", summary)
        _append(con, "backtest_bets", bets.drop(columns=["y"]))
        logger.info("Wrote run %s: %d summary rows, %d candidate rows", run_id, len(summary), len(bets))
    return summary, bets


def _append(con: duckdb.DuckDBPyConnection, table: str, df: pd.DataFrame) -> None:
    con.register("_bt", df)
    if gs_table_exists(con, table):
        con.execute(f"INSERT INTO {table} BY NAME SELECT * FROM _bt")
    else:
        con.execute(f"CREATE TABLE {table} AS SELECT * FROM _bt")
    con.unregister("_bt")


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Walk-forward backtest (train 2018..Y-1, evaluate Y).")
    p.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    p.add_argument("--seasons", default=None, help="Comma list of evaluation seasons.")
    p.add_argument("--min-edge", type=float, default=2.0)
    p.add_argument("--kelly", type=float, default=dv.DEFAULT_KELLY_FRACTION)
    p.add_argument("--bankroll", type=float, default=dv.DEFAULT_BANKROLL)
    p.add_argument("--devig", choices=["shin", "multiplicative"], default="shin")
    p.add_argument("--no-td", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = BacktestConfig(eval_seasons=[int(s) for s in args.seasons.split(",")] if args.seasons else None,
                         min_edge_pct=args.min_edge, kelly=args.kelly, bankroll=args.bankroll,
                         devig_method=args.devig, include_td=not args.no_td)
    with duckdb.connect(str(args.db_path)) as con:
        summary, _ = run_backtest(con, cfg)
    cols = ["season", "market_category", "n_predictions", "brier_model", "brier_market", "logloss_model",
            "logloss_market", "n_bets", "roi_pct", "flat_roi_pct", "hit_rate", "avg_edge_pct", "avg_clv_pct"]
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(summary[cols].sort_values(["market_category", "season"], na_position="last")
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
