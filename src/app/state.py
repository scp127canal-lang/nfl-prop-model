"""Session state, DuckDB access and cached computations for the Streamlit app.

Rerun safety (CLAUDE.md rule 2)
-------------------------------
* Nothing in this module makes an HTTP call at import or on a normal rerun. The only live path is
  :func:`pull_live_odds_callback`, wired exclusively to the sidebar ``"Pull Live Odds"`` button's
  ``on_click``. Under pytest the underlying ``OddsClient`` refuses live calls anyway (rule 1).
* DuckDB is opened read-only and only for the duration of a query, so CLI jobs can still write.
  Every read and every simulation is memoised with ``st.cache_data`` / ``st.cache_resource``,
  keyed on the database file's version (mtime + size) so a write invalidates stale results.

The heavy lifting lives in plain functions taking a connection (``price_markets``,
``line_shop_matrix``, ``equity_curve`` ...) so it can be unit-tested without Streamlit.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

from src.evaluation import backtest as bt
from src.evaluation import devig as dv
from src.ingestion import odds_client as oc
from src.models import game_sim as gs
from src.utils import entity_resolution as er
from src.utils.secrets import get_secret

logger = logging.getLogger(__name__)

GAME_MARKETS = {"h2h", "spreads", "totals", "alternate_spreads", "alternate_totals", "team_totals",
                "alternate_team_totals"}
TD_STATS = {"anytime_td", "first_td"}
STAT_LABELS = {
    "pass_yds": "Passing yards", "pass_attempts": "Pass attempts", "pass_completions": "Completions",
    "pass_tds": "Passing TDs", "pass_interceptions": "Interceptions", "rush_yds": "Rushing yards",
    "rush_attempts": "Rush attempts", "rush_longest": "Longest rush", "receptions": "Receptions",
    "reception_yds": "Receiving yards", "reception_longest": "Longest reception", "targets": "Targets",
    "anytime_td": "Touchdowns", "first_td": "First TD", "rec_tds": "Receiving TDs", "rush_tds": "Rushing TDs",
}
STAT_ORDER = ["pass_yds", "pass_completions", "pass_attempts", "pass_tds", "pass_interceptions", "reception_yds",
              "receptions", "targets", "rush_yds", "rush_attempts", "anytime_td", "reception_longest",
              "rush_longest", "first_td"]
MARKET_GROUP_LABELS = {"main": "Main (ML / spread / total)", "team_totals": "Team totals",
                       "alternates": "Alternate lines", "player_props": "Player props"}
SHARP_AUTO = "auto"


# ======================================================================================
# Paths, connections, cache keys
# ======================================================================================
DEPLOY_DB_PATH = oc.PROJECT_ROOT / "data" / "deploy_model.duckdb"


def db_path() -> Path:
    """``NFL_DB_PATH`` override -> full local ``nfl_model.duckdb`` -> compact ``deploy_model.duckdb``.

    The full DB is gitignored, so on Streamlit Community Cloud only the deploy DB exists."""
    if os.getenv("NFL_DB_PATH"):
        return Path(os.environ["NFL_DB_PATH"])
    full = Path(oc.DEFAULT_DB_PATH)
    return full if full.exists() or not DEPLOY_DB_PATH.exists() else DEPLOY_DB_PATH


def raw_dir() -> Path:
    return Path(os.getenv("NFL_RAW_ODDS_DIR") or oc.DEFAULT_RAW_DIR)


def db_version(path: Path | None = None) -> str:
    """Cache key that changes whenever the database file (or its WAL) is written."""
    path = path or db_path()
    parts = []
    for p in (path, path.with_name(path.name + ".wal")):
        if p.exists():
            s = p.stat()
            parts.append(f"{s.st_mtime_ns}:{s.st_size}")
    return "|".join(parts) or "missing"


@contextmanager
def connect(path: Path | None = None) -> Iterator[duckdb.DuckDBPyConnection]:
    """Short-lived read-only connection (never held across reruns)."""
    path = path or db_path()
    if not path.exists():
        raise FileNotFoundError(f"DuckDB not found at {path}; run src.ingestion.nfl_data first")
    con = duckdb.connect(str(path), read_only=True)
    try:
        yield con
    finally:
        con.close()


def table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bt.gs_table_exists(con, table)


# ======================================================================================
# Settings / session state
# ======================================================================================
@dataclass(frozen=True)
class Settings:
    bankroll: float = dv.DEFAULT_BANKROLL
    kelly: float = dv.DEFAULT_KELLY_FRACTION
    min_edge_pct: float = 2.0
    sharp_book: str = SHARP_AUTO
    devig_method: str = "shin"
    max_stake_pct: float = dv.DEFAULT_MAX_STAKE_PCT
    n_trials: int = gs.N_TRIALS
    seed: int = 11

    def sharp_priority(self) -> tuple[str, ...]:
        if self.sharp_book == SHARP_AUTO:
            return tuple(dv.SHARP_BOOKS)
        return (self.sharp_book,) + tuple(b for b in dv.SHARP_BOOKS if b != self.sharp_book)


SESSION_DEFAULTS: dict[str, Any] = {
    "bankroll": dv.DEFAULT_BANKROLL, "kelly": dv.DEFAULT_KELLY_FRACTION, "min_edge_pct": 2.0,
    "sharp_book": SHARP_AUTO, "devig_method": "shin", "last_pull": None, "pull_error": None,
    "live_calls": 0,
}


def init_session_state() -> None:
    for k, v in SESSION_DEFAULTS.items():
        st.session_state.setdefault(k, v)


def settings_from_session() -> Settings:
    s = st.session_state
    return Settings(bankroll=float(s["bankroll"]), kelly=float(s["kelly"]), min_edge_pct=float(s["min_edge_pct"]),
                    sharp_book=s["sharp_book"], devig_method=s["devig_method"])


# ======================================================================================
# Odds snapshots
# ======================================================================================
def latest_odds(con: duckdb.DuckDBPyConnection, include_started: bool = False,
                now: datetime | None = None) -> pd.DataFrame:
    """Latest pre-kickoff quote per (event, book, market) from ``raw_odds`` plus ``game_id``."""
    if not table_exists(con, "raw_odds"):
        return pd.DataFrame()
    now = (now or datetime.now(timezone.utc)).replace(tzinfo=None)
    df = con.execute("""
        SELECT * FROM raw_odds r
        WHERE snapshot_ts_utc < commence_time_utc
        QUALIFY snapshot_ts_utc = MAX(snapshot_ts_utc) OVER (PARTITION BY event_id, bookmaker_key, market_key)""").df()
    if df.empty:
        return df
    if not include_started:
        df = df[pd.to_datetime(df["commence_time_utc"]) > pd.Timestamp(now)]
    emap = bt.map_events_to_games(con)
    df = df.merge(emap, on="event_id", how="left")
    df["home_abbr"] = df["home_team"].map(er.team_to_abbr)
    df["away_abbr"] = df["away_team"].map(er.team_to_abbr)
    df["game"] = df["away_abbr"].fillna(df["away_team"]) + " @ " + df["home_abbr"].fillna(df["home_team"])
    df["base_market"] = df["market_key"].str.removesuffix("_alternate")
    return df.reset_index(drop=True)


def make_resolver(con: duckdb.DuckDBPyConnection) -> er.PlayerResolver:
    """Entity resolver (rule 4) that works on a read-only DB: candidates and reviewed mappings
    are copied into an in-memory cache table, so nothing is written to the main database."""
    mem = duckdb.connect(":memory:")
    resolver = er.PlayerResolver(mem, candidates=er.load_candidates(con))
    if table_exists(con, "player_id_map"):
        mem.register("_pim", con.execute("SELECT * FROM player_id_map").df())
        mem.execute("INSERT INTO player_id_map SELECT * FROM _pim")
        mem.unregister("_pim")
    return resolver


def resolve_player_rows(odds: pd.DataFrame, resolver: er.PlayerResolver) -> tuple[pd.Series, list[str]]:
    """gsis_id for every player-prop row (NaN when unmatched) and the list of unmatched names."""
    ids = pd.Series(np.nan, index=odds.index, dtype=object)
    unmatched: set[str] = set()
    props = odds[odds["is_player_prop"].astype(bool) & odds["outcome_description"].notna()]
    for (name, home, away, mk), idx in props.groupby(
            ["outcome_description", "home_team", "away_team", "base_market"]).groups.items():
        res = resolver.resolve(name, [home, away], mk)
        if res.matched:
            ids.loc[idx] = res.gsis_id
        else:
            unmatched.add(name)
    return ids, sorted(unmatched)


# ======================================================================================
# Model pricing
# ======================================================================================
def td_predictions(con: duckdb.DuckDBPyConnection, game_ids: Sequence[str]) -> pd.DataFrame:
    if not game_ids or not table_exists(con, "td_predictions"):
        return pd.DataFrame(columns=["game_id", "gsis_id", "p_anytime_td", "p_first_td"])
    q = f"SELECT game_id, gsis_id, p_anytime_td, p_first_td FROM td_predictions WHERE game_id IN ({','.join('?' * len(game_ids))})"
    return con.execute(q, list(game_ids)).df()


def _p_over(x: np.ndarray, line: float) -> tuple[float, float]:
    return bt.p_over(x, line)


def simulate_for_game(con: duckdb.DuckDBPyConnection, game_id: str, n_trials: int, seed: int,
                      overrides: dict | None = None) -> tuple[gs.SimResult | None, tuple[np.ndarray, np.ndarray] | None]:
    """(player SimResult or None, (home_pts, away_pts) or None). Game model trained < game date."""
    home, away = con.execute("SELECT home_team, away_team FROM schedules WHERE game_id = ?", [game_id]).fetchall()[0]
    try:
        sim = gs.simulate_game(game_id, overrides, con=con, n_trials=n_trials, seed=seed)
        return sim, (sim.teams[home]["points"], sim.teams[away]["points"])
    except ValueError as exc:
        logger.info("player sim unavailable for %s (%s); scores only", game_id, exc)
    try:
        params, _ = gs.game_inputs(con, game_id)
    except ValueError as exc:
        logger.warning("no game model inputs for %s: %s", game_id, exc)
        return None, None
    _, _, pts = gs.simulate_points(params, n_trials, np.random.default_rng(seed))
    return None, (pts[home], pts[away])


def player_history(sim: gs.SimResult | None) -> dict[str, tuple[float, bool]]:
    """gsis_id -> (games with >20% snaps before this game, cold-start flag) from the sim inputs."""
    if sim is None or "qual_games" not in sim.players:
        return {}
    cold = sim.players["cold_start"] if "cold_start" in sim.players else sim.players["qual_games"] < gs.COLD_START_GAMES
    return {pid: (float(q), bool(c)) for pid, q, c in zip(sim.players["gsis_id"], sim.players["qual_games"], cold)}


def price_markets(con: duckdb.DuckDBPyConnection, sharp_priority: Sequence[str] = dv.SHARP_BOOKS,
                  devig_method: str = "shin", n_trials: int = gs.N_TRIALS, seed: int = 11,
                  include_started: bool = False, now: datetime | None = None) -> tuple[pd.DataFrame, dict]:
    """Every quoted selection with devigged fair probability, best price and model probability.

    Fair probabilities always come from :func:`devig.fair_probabilities` (sharp book first,
    consensus fallback, rule 5); player names go through the entity resolver (rule 4).
    """
    diag: dict[str, Any] = {"unmatched_players": [], "unsupported_markets": {}, "games_without_model": []}
    odds = latest_odds(con, include_started, now)
    if odds.empty:
        return pd.DataFrame(), diag
    odds = odds[odds["game_id"].notna()].copy()
    diag["unmapped_events"] = int(odds["game_id"].isna().sum())
    if odds["is_player_prop"].astype(bool).any():
        odds["gsis_id"], diag["unmatched_players"] = resolve_player_rows(odds, make_resolver(con))
    else:
        odds["gsis_id"] = np.nan
    fair = dv.fair_probabilities(odds, devig_method, sharp_priority)
    if fair.empty:
        return pd.DataFrame(), diag
    meta = odds.drop_duplicates("event_id").set_index("event_id")[
        ["game_id", "game", "home_team", "away_team", "home_abbr", "away_abbr", "commence_time_utc"]]
    fair = fair.join(meta, on="event_id")
    keys = dv.add_group_keys(odds)[["selection_key", "gsis_id"]].drop_duplicates("selection_key")
    fair = fair.merge(keys, on="selection_key", how="left")
    tdp = td_predictions(con, fair["game_id"].dropna().unique().tolist())

    rows = []
    for gid, g in fair.groupby("game_id", sort=False):
        sim, scores = simulate_for_game(con, gid, n_trials, seed)
        if scores is None:
            diag["games_without_model"].append(gid)
            continue
        depth = player_history(sim)
        for r in g.itertuples(index=False):
            mk, name = r.market_key, r.outcome_name
            point = None if pd.isna(r.outcome_point) else float(r.outcome_point)
            p_win = p_push = None
            label, side = name, name
            if mk in GAME_MARKETS:
                if mk in ("spreads", "alternate_spreads", "totals", "alternate_totals", "team_totals",
                          "alternate_team_totals") and point is None:
                    continue
                p_win, p_push = bt.game_market_prob(scores[0], scores[1], mk, name, point, r.home_team,
                                                    r.away_team, r.outcome_description)
                if mk in ("team_totals", "alternate_team_totals"):
                    label = er.team_to_abbr(r.outcome_description) or r.outcome_description
                elif mk in ("h2h", "spreads", "alternate_spreads"):
                    label = er.team_to_abbr(name) or name
                    side = "ML" if mk == "h2h" else ""
                else:
                    label = r.game
            else:
                base = mk.removesuffix("_alternate")
                stat = gs.MARKET_STATS.get(base)
                label = r.outcome_description
                pid = r.gsis_id
                if stat is None or not isinstance(pid, str):
                    key = "unmatched_player" if stat is not None else base
                    diag["unsupported_markets"][key] = diag["unsupported_markets"].get(key, 0) + 1
                    continue
                if stat in TD_STATS:
                    tp = tdp[(tdp["game_id"] == gid) & (tdp["gsis_id"] == pid)]
                    if len(tp):
                        p_yes = float(tp[f"p_{stat}"].iloc[0])
                    elif sim is not None and pid in sim.stats:
                        p_yes = sim.prob_at_least(pid, stat, 1)
                    else:
                        continue
                    if point is not None and point > 0.5 and sim is not None and pid in sim.stats:
                        p_yes = sim.prob_over(pid, stat, point)    # "2+ TDs" style lines
                    p_win, p_push = (p_yes if name in ("Yes", "Over") else 1 - p_yes), 0.0
                else:
                    if sim is None or pid not in sim.stats or point is None or stat not in sim.stats[pid]:
                        diag["unsupported_markets"][base] = diag["unsupported_markets"].get(base, 0) + 1
                        continue
                    over, push = _p_over(sim.get(pid, stat), point)
                    p_win, p_push = (over if name == "Over" else max(1 - over - push, bt.EPS)), push
            rows.append({
                "game_id": gid, "game": r.game, "commence_time_utc": r.commence_time_utc, "event_id": r.event_id,
                "market_key": mk, "market_category": bt.market_category(mk.removesuffix("_alternate")),
                "is_alternate": "alternate" in mk, "selection": label, "side": side, "line": point,
                "gsis_id": r.gsis_id if isinstance(r.gsis_id, str) else None,
                "best_book": r.best_book, "best_american": r.best_price_american,
                "best_decimal": r.best_price_decimal, "fair_prob": r.fair_prob, "fair_source": r.fair_source,
                "devig_method": r.devig_method, "n_books": r.n_books, "model_p_win": float(p_win),
                "model_p_push": float(p_push), "selection_key": r.selection_key,
                "hist_games": depth.get(r.gsis_id, (np.nan, False))[0] if isinstance(r.gsis_id, str) else np.nan,
                "cold_start": bool(depth.get(r.gsis_id, (np.nan, False))[1]) if isinstance(r.gsis_id, str) else False})
    out = pd.DataFrame(rows)
    if out.empty:
        return out, diag
    out["model_prob"] = (out["model_p_win"] / (1 - out["model_p_push"].clip(0, 0.999))).clip(bt.EPS, 1 - bt.EPS)
    out["fair_american"] = dv.decimal_to_american(1 / out["fair_prob"].clip(bt.EPS, 1 - bt.EPS))
    out["model_american"] = dv.decimal_to_american(1 / out["model_prob"])
    return out, diag


def apply_bankroll(priced: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """EV %, edge %, fractional-Kelly stake and the +EV flag (cheap; reruns on every slider move)."""
    if priced.empty:
        return priced.assign(ev_pct=[], edge_pct=[], kelly_stake=[], is_bet=[])
    c = priced.copy()
    p, push, d = c["model_p_win"].to_numpy(float), c["model_p_push"].to_numpy(float), c["best_decimal"].to_numpy(float)
    c["ev_pct"] = dv.ev_pct(p, d, push)
    c["edge_pct"] = dv.edge_pct(c["model_prob"].to_numpy(float), c["fair_prob"].to_numpy(float))
    c["kelly_stake"] = dv.kelly_stake(p, d, settings.bankroll, settings.kelly, push, settings.max_stake_pct)
    c["is_bet"] = (c["ev_pct"] > 0) & (c["edge_pct"] >= settings.min_edge_pct) & (c["kelly_stake"] > 0)
    return c


SCANNER_COLUMNS = {"game": "Game", "selection": "Player/Team", "market_key": "Market", "side": "Side",
                   "line": "Line", "best_book": "Best Book", "best_american": "Best Odds",
                   "fair_american": "Sharp No-Vig Fair", "model_american": "Model Fair", "edge_pct": "Edge %",
                   "ev_pct": "EV %", "kelly_stake": "Kelly Stake $", "hist_games": "Hist. Games",
                   "fair_source": "Fair Source", "n_books": "Books"}
DEFAULT_MAX_EDGE_PCT = 15.0
DEFAULT_MIN_HIST_GAMES = 3


def filter_evaluated(evaluated: pd.DataFrame, positive_only: bool = True, categories: Sequence[str] | None = None,
                     games: Sequence[str] | None = None, books: Sequence[str] | None = None,
                     include_alternates: bool = True, max_edge_pct: float | None = None,
                     min_hist_games: int | None = None) -> pd.DataFrame:
    """Row filter behind the scanner table and its KPIs.

    ``max_edge_pct`` drops selections whose edge exceeds the cap (edges that large are almost
    always model error, e.g. thin-history players, not market error). ``min_hist_games`` drops
    player props for players with fewer games above 20% snap share; game markets are unaffected.
    """
    d = evaluated
    if d.empty:
        return d
    if positive_only:
        d = d[d["is_bet"]]
    if categories:
        d = d[d["market_category"].isin(categories)]
    if games:
        d = d[d["game"].isin(games)]
    if books:
        d = d[d["best_book"].isin(books)]
    if not include_alternates:
        d = d[~d["is_alternate"]]
    if max_edge_pct is not None:
        d = d[d["edge_pct"] <= max_edge_pct]
    if min_hist_games:
        hist = d["hist_games"] if "hist_games" in d else pd.Series(np.nan, index=d.index)
        is_prop = d["market_key"].str.startswith("player_")
        d = d[~is_prop | (hist.fillna(0) >= min_hist_games)]
    return d


def format_scanner(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame(columns=list(SCANNER_COLUMNS.values()))
    d = d.copy()
    if "hist_games" not in d:
        d["hist_games"] = np.nan
    return d.sort_values("edge_pct", ascending=False)[list(SCANNER_COLUMNS)].rename(columns=SCANNER_COLUMNS)


def scanner_table(evaluated: pd.DataFrame, positive_only: bool = True, categories: Sequence[str] | None = None,
                  games: Sequence[str] | None = None, books: Sequence[str] | None = None,
                  include_alternates: bool = True, max_edge_pct: float | None = None,
                  min_hist_games: int | None = None) -> pd.DataFrame:
    return format_scanner(filter_evaluated(evaluated, positive_only, categories, games, books, include_alternates,
                                           max_edge_pct, min_hist_games))


# ======================================================================================
# Line shopping
# ======================================================================================
def selection_label(df: pd.DataFrame) -> pd.Series:
    desc = df["outcome_description"].fillna("").astype(str)
    total_like = (df["market_key"].str.contains("total") | df["market_key"].str.startswith("player_")).to_numpy()
    pts = pd.to_numeric(df["outcome_point"], errors="coerce").to_numpy(float)
    pt = pd.Series(["" if np.isnan(v) else (f" {v:g}" if tl else f" {v:+g}") for v, tl in zip(pts, total_like)],
                   index=df.index)
    return (desc.where(desc == "", desc + " ") + df["outcome_name"].astype(str) + pt).str.strip()


def line_shop_matrix(odds: pd.DataFrame, event_id: str, market_key: str, method: str = "shin",
                     sharp_priority: Sequence[str] = dv.SHARP_BOOKS, stale_minutes: float = 30.0,
                     off_market_pp: float = 3.0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(American price matrix [selection x book], flag matrix, per-book detail).

    Flags: ``best`` (best price for the selection), ``off`` (that book's own no-vig probability is
    >= ``off_market_pp`` points from the fair line), ``stale`` (book's market updated
    >= ``stale_minutes`` before the freshest book). A cell can be e.g. ``"best off"``.
    """
    d = odds[(odds["event_id"] == event_id) & (odds["market_key"] == market_key)].copy()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    per_book = dv.devig_book_groups(d, method)
    fair = dv.fair_probabilities(d, method, sharp_priority).set_index("selection_key")
    per_book["fair_prob"] = per_book["selection_key"].map(fair["fair_prob"])
    per_book["label"] = selection_label(per_book)
    upd = pd.to_datetime(per_book["market_last_update_utc"].fillna(per_book["bookmaker_last_update_utc"]))
    per_book["lag_min"] = (upd.max() - upd).dt.total_seconds() / 60
    per_book["off_pp"] = 100 * (per_book["fair_prob_book"] - per_book["fair_prob"])
    order = per_book.drop_duplicates("label").sort_values(["outcome_description", "outcome_point", "outcome_name"],
                                                          na_position="first")["label"].tolist()
    books = sorted(per_book["bookmaker_key"].unique(), key=lambda b: (b not in sharp_priority, b))
    price = per_book.pivot_table(index="label", columns="bookmaker_key", values="price_american", aggfunc="last")
    price = price.reindex(index=order, columns=books)
    best = per_book.groupby("label")["price_decimal"].transform("max") == per_book["price_decimal"]
    per_book["flag"] = (np.where(best, "best ", "") + np.where(per_book["off_pp"].abs() >= off_market_pp, "off ", "")
                        + np.where(per_book["lag_min"] >= stale_minutes, "stale", "")).astype(str)
    per_book["flag"] = per_book["flag"].str.strip()
    flags = per_book.pivot_table(index="label", columns="bookmaker_key", values="flag", aggfunc="last")
    flags = flags.reindex(index=order, columns=books).fillna("")
    fair_by_label = per_book.dropna(subset=["fair_prob"]).drop_duplicates("label").set_index("label")["fair_prob"]
    price.insert(0, "Fair (no-vig)", fair_by_label.reindex(order).map(
        lambda p: float(dv.decimal_to_american(1 / p)) if pd.notna(p) else np.nan).to_numpy())
    flags.insert(0, "Fair (no-vig)", "fair")
    detail = per_book[["bookmaker_key", "label", "price_american", "fair_prob_book", "fair_prob", "off_pp",
                       "lag_min", "flag"]].sort_values(["label", "bookmaker_key"])
    return price, flags, detail


# ======================================================================================
# Player explorer
# ======================================================================================
def explorer_games(con: duckdb.DuckDBPyConnection, include_past: bool = False,
                   today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Games that have player features (simulatable), upcoming first."""
    today = (today or pd.Timestamp.now()).normalize()
    df = con.execute("""
        SELECT s.game_id, s.season, s.week, s.game_date, s.home_team, s.away_team, s.home_score
        FROM schedules s WHERE s.game_id IN (SELECT DISTINCT game_id FROM player_features)
        ORDER BY s.game_date, s.game_id""").df()
    df["game_date"] = pd.to_datetime(df["game_date"])
    if not include_past:
        df = df[df["game_date"] >= today - pd.Timedelta(days=1)]
    else:
        df = df.sort_values("game_date", ascending=False)
    df["label"] = (df["away_team"] + " @ " + df["home_team"] + "  ·  " + df["game_date"].dt.strftime("%a %b %d")
                   + "  (wk " + df["week"].astype(str) + ")")
    return df.reset_index(drop=True)


def usage_overrides(player: pd.Series, target_share: float, carry_share: float, snap_share: float,
                    tol: float = 1e-4) -> dict[str, dict[str, float]]:
    """Override dict for ``simulate_game``. Snap share scales opportunity: shares are multiplied
    by ``snap_share / baseline snap share`` (teammates are rescaled by the simulator)."""
    base_snap = float(player.get("snap_share_ewm5") or 0.0)
    ratio = snap_share / base_snap if base_snap > tol else 1.0
    t, c = target_share * ratio, carry_share * ratio
    ov: dict[str, float] = {}
    if abs(t - float(player["target_share"])) > tol:
        ov["target_share"] = float(np.clip(t, 0.0, 0.95))
    if abs(c - float(player["carry_share"])) > tol:
        ov["carry_share"] = float(np.clip(c, 0.0, 0.95))
    return {player["gsis_id"]: ov} if ov else {}


def player_book_lines(odds: pd.DataFrame, game_id: str, gsis_id: str, stat: str,
                      resolver: er.PlayerResolver | None = None) -> pd.DataFrame:
    """Every book's main and alternate line for ``gsis_id``'s ``stat`` in this game."""
    cols = ["bookmaker_key", "market_key", "line", "Over", "Under", "is_alternate"]
    if odds.empty or "game_id" not in odds:
        return pd.DataFrame(columns=cols)
    d = odds[(odds["game_id"] == game_id) & odds["is_player_prop"].astype(bool)].copy()
    d = d[d["base_market"].map(gs.MARKET_STATS.get) == stat]
    if d.empty:
        return pd.DataFrame(columns=cols)
    if "gsis_id" not in d:
        if resolver is None:
            raise ValueError("resolver required to map player names to gsis_id")
        d["gsis_id"], _ = resolve_player_rows(d, resolver)
    d = d[d["gsis_id"] == gsis_id]
    if d.empty:
        return pd.DataFrame(columns=cols)
    d["line"] = pd.to_numeric(d["outcome_point"], errors="coerce").fillna(0.5)
    d["side"] = d["outcome_name"].replace({"Yes": "Over", "No": "Under"})
    out = d.pivot_table(index=["bookmaker_key", "market_key", "line"], columns="side", values="price_american",
                        aggfunc="last").reset_index()
    for side in ("Over", "Under"):
        if side not in out:
            out[side] = np.nan
    out["is_alternate"] = out["market_key"].str.endswith("_alternate")
    return out[cols].sort_values(["line", "bookmaker_key"]).reset_index(drop=True)


def line_probabilities(x: np.ndarray, lines: Sequence[float]) -> pd.DataFrame:
    rows = []
    for ln in sorted(set(float(v) for v in lines)):
        over, push = bt.p_over(x, ln)
        under = max(1 - over - push, bt.EPS)
        rows.append({"line": ln, "p_over": over, "p_push": push, "p_under": under,
                     "fair_over": float(dv.decimal_to_american(1 / (over / (1 - push)))) if push < 1 else np.nan,
                     "fair_under": float(dv.decimal_to_american(1 / (under / (1 - push)))) if push < 1 else np.nan})
    return pd.DataFrame(rows)


# ======================================================================================
# Backtest
# ======================================================================================
def backtest_runs(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    if not table_exists(con, "backtest_results"):
        return pd.DataFrame(columns=["run_id", "created_at_utc", "config"])
    return con.execute("""SELECT run_id, MAX(created_at_utc) AS created_at_utc, ANY_VALUE(config) AS config
                          FROM backtest_results GROUP BY run_id ORDER BY 2 DESC""").df()


def load_backtest(con: duckdb.DuckDBPyConnection, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    res = con.execute("SELECT * FROM backtest_results WHERE run_id = ?", [run_id]).df()
    if not table_exists(con, "backtest_bets"):
        return res, pd.DataFrame()
    bets = con.execute("""SELECT b.*, s.game_date FROM backtest_bets b LEFT JOIN schedules s USING (game_id)
                          WHERE b.run_id = ?""", [run_id]).df()
    bets["game_date"] = pd.to_datetime(bets["game_date"])
    return res, bets


def run_bankroll(results: pd.DataFrame, default: float = dv.DEFAULT_BANKROLL) -> float:
    try:
        return float(json.loads(results["config"].dropna().iloc[0]).get("bankroll", default))
    except (IndexError, ValueError, TypeError, AttributeError):
        return default


def equity_curve(bets: pd.DataFrame, bankroll: float, by_category: bool = True) -> pd.DataFrame:
    """Bankroll after each betting day (stakes sized on the fixed starting bankroll, as in the backtest)."""
    b = bets[bets["is_bet"].astype(bool)].dropna(subset=["game_date"])
    if b.empty:
        return pd.DataFrame(columns=["game_date", "series", "bankroll", "n_bets"])
    parts = [b.assign(series="All markets")]
    if by_category:
        parts.append(b.assign(series=b["market_category"]))
    d = pd.concat(parts)
    daily = d.groupby(["series", "game_date"]).agg(profit=("profit", "sum"), n=("profit", "size")).reset_index()
    daily = daily.sort_values(["series", "game_date"])
    daily["bankroll"] = bankroll + daily.groupby("series")["profit"].cumsum()
    daily["n_bets"] = daily.groupby("series")["n"].cumsum()
    return daily[["game_date", "series", "bankroll", "n_bets"]]


def calibration_table(bets: pd.DataFrame, categories: Sequence[str] | None = None, n_bins: int = 10) -> pd.DataFrame:
    """Reliability curve (model and market) on settled, non-push predictions."""
    d = bets[bets["outcome"].isin(["win", "loss"])]
    if categories:
        d = d[d["market_category"].isin(categories)]
    if d.empty:
        return pd.DataFrame(columns=["source", "bin", "mean_pred", "observed", "n"])
    y = (d["outcome"] == "win").astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for source, col in (("Model", "model_prob"), ("Market (devigged)", "fair_prob")):
        p = d[col].astype(float)
        b = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
        g = pd.DataFrame({"p": p.to_numpy(), "y": y.to_numpy(), "bin": b}).groupby("bin")
        t = g.agg(mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index()
        out.append(t.assign(source=source))
    return pd.concat(out, ignore_index=True)[["source", "bin", "mean_pred", "observed", "n"]]


def clv_summary(bets: pd.DataFrame) -> pd.DataFrame:
    b = bets[bets["is_bet"].astype(bool) & bets["clv_pct"].notna()]
    if b.empty:
        return pd.DataFrame(columns=["market_category", "n_bets", "avg_clv_pct", "median_clv_pct", "beat_close_pct"])
    return (b.groupby("market_category")["clv_pct"]
            .agg(n_bets="size", avg_clv_pct="mean", median_clv_pct="median",
                 beat_close_pct=lambda s: 100 * (s > 0).mean()).reset_index())


def roi_by_market(results: pd.DataFrame) -> pd.DataFrame:
    r = results[(results["scope"] == "all") & (results["n_bets"] > 0)]
    return r[["market_category", "n_bets", "staked", "profit", "roi_pct", "flat_roi_pct", "hit_rate",
              "avg_edge_pct", "avg_clv_pct"]].sort_values("roi_pct", ascending=False)


# ======================================================================================
# Live odds pull (button callback only)
# ======================================================================================
def app_password() -> str:
    """Optional dashboard password (``APP_PASSWORD`` in env/.env or ``st.secrets``); "" = no gate."""
    oc.load_dotenv(oc.PROJECT_ROOT / ".env")
    return get_secret("APP_PASSWORD")


def check_password(candidate: str) -> bool:
    expected = app_password()
    return not expected or hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


def is_authenticated() -> bool:
    return not app_password() or bool(st.session_state.get("authenticated"))


def api_key_configured() -> bool:
    try:
        oc.load_api_key()
        return True
    except RuntimeError:
        return False


def last_known_quota(directory: Path | None = None) -> dict[str, Any]:
    """Remaining credits as recorded in the newest cached payload (zero-credit: reads disk only)."""
    directory = directory or raw_dir()
    files = sorted(Path(directory).glob("*.json"), reverse=True) if Path(directory).exists() else []
    for p in files:
        try:
            w = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        q = w.get("quota") or {}
        if q.get("remaining") is not None:
            return {**q, "as_of": w.get("fetched_at_utc"), "n_files": len(files)}
    return {"remaining": None, "used": None, "last": None, "as_of": None, "n_files": len(files)}


def selected_markets(groups: Sequence[str]) -> list[str]:
    return oc.resolve_markets(list(groups)) if groups else []


def estimate_pull_cost(groups: Sequence[str], regions: Sequence[str], max_events: int) -> int:
    """Upper-bound credits: events x markets x regions (events listing itself costs 0)."""
    markets = selected_markets(groups)
    return max_events * oc.estimate_event_cost(markets, regions) if markets else 0


def _clear_caches() -> None:
    st.cache_data.clear()


def pull_live_odds_callback(groups: Sequence[str], regions: Sequence[str], max_events: int,
                            cache_ttl_minutes: int = 15) -> None:
    """``on_click`` of the "Pull Live Odds" button: the app's ONLY live HTTP path (rules 1-2).

    Re-uses fresh cached payloads (``cache_ttl``), caps spend at the displayed estimate and logs
    ``x-requests-remaining`` / ``x-requests-last`` (done inside ``odds_client``)."""
    from datetime import timedelta

    if not is_authenticated():          # defence in depth: the button is only rendered after login
        st.session_state["pull_error"] = "not authenticated"
        return

    st.session_state["pull_error"] = None
    st.session_state["live_calls"] = st.session_state.get("live_calls", 0) + 1
    markets = selected_markets(groups)
    budget = estimate_pull_cost(groups, regions, max_events)
    try:
        summary = asyncio.run(oc.run_live(oc.load_api_key(), markets=markets, regions=list(regions),
                                          max_events=max_events, max_credits=max(budget, 1),
                                          cache_ttl=timedelta(minutes=cache_ttl_minutes),
                                          raw_dir=raw_dir(), db_path=db_path()))
        st.session_state["last_pull"] = {
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "events": summary.events_seen,
            "fetched": len(summary.fetched), "from_cache": len(summary.reused_from_cache),
            "skipped": len(summary.skipped), "rows": summary.rows_ingested, "credits_spent": summary.credits_spent,
            "remaining": summary.quota.get("remaining"), "last": summary.quota.get("last")}
    except Exception as exc:  # noqa: BLE001 - surface any failure in the UI, never crash the rerun
        logger.exception("live odds pull failed")
        st.session_state["pull_error"] = oc.redact(str(exc))
    _clear_caches()


def load_cached_odds_callback() -> None:
    """Zero-credit: ingest the latest cached payload per event from data/raw_odds/."""
    st.session_state["pull_error"] = None
    try:
        s = oc.run_dry(raw_dir=raw_dir(), db_path=db_path())
        st.session_state["last_pull"] = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                         "events": s.events_seen, "fetched": 0, "from_cache": s.events_seen,
                                         "skipped": 0, "rows": s.rows_ingested, "credits_spent": 0,
                                         "remaining": last_known_quota().get("remaining"), "last": 0}
    except Exception as exc:  # noqa: BLE001
        logger.exception("cached ingest failed")
        st.session_state["pull_error"] = str(exc)
    _clear_caches()


# ======================================================================================
# Streamlit-cached wrappers (keyed on db version)
# ======================================================================================
@st.cache_data(show_spinner=False, ttl=600)
def cached_latest_odds(version: str, include_started: bool) -> pd.DataFrame:
    with connect() as con:
        return latest_odds(con, include_started)


@st.cache_data(show_spinner="Simulating games & pricing every market…", ttl=600)
def cached_price_markets(version: str, sharp_priority: tuple[str, ...], devig_method: str, n_trials: int,
                         seed: int, include_started: bool) -> tuple[pd.DataFrame, dict]:
    with connect() as con:
        return price_markets(con, sharp_priority, devig_method, n_trials, seed, include_started)


@st.cache_data(show_spinner=False)
def cached_explorer_games(version: str, include_past: bool) -> pd.DataFrame:
    with connect() as con:
        return explorer_games(con, include_past)


@st.cache_data(show_spinner=False)
def cached_players(version: str, game_id: str) -> pd.DataFrame:
    with connect() as con:
        return gs.load_players(con, game_id)


@st.cache_resource(show_spinner=False, max_entries=16)
def cached_game_model(version: str, game_date: str, kind: str = "ridge") -> gs.GameModel:
    with connect() as con:
        gd = gs.build_game_dataset(con)
    return gs.GameModel(kind=kind).fit(gd[pd.to_datetime(gd["game_date"]) < pd.Timestamp(game_date)])


@st.cache_data(show_spinner="Running Monte Carlo…", max_entries=48)
def cached_simulation(version: str, game_id: str, overrides_json: str, n_trials: int, seed: int) -> gs.SimResult:
    with connect() as con:
        gdate = con.execute("SELECT game_date FROM schedules WHERE game_id = ?", [game_id]).fetchall()[0][0]
        model = cached_game_model(version, str(gdate))
        return gs.simulate_game(game_id, json.loads(overrides_json) or None, con=con, model=model,
                                n_trials=n_trials, seed=seed)


@st.cache_data(show_spinner=False)
def cached_player_lines(version: str, game_id: str, gsis_id: str, stat: str) -> pd.DataFrame:
    odds = cached_latest_odds(version, True)
    if odds.empty:
        return player_book_lines(odds, game_id, gsis_id, stat)
    with connect() as con:
        return player_book_lines(odds, game_id, gsis_id, stat, make_resolver(con))


@st.cache_data(show_spinner=False)
def cached_backtest_runs(version: str) -> pd.DataFrame:
    with connect() as con:
        return backtest_runs(con)


@st.cache_data(show_spinner="Loading backtest…")
def cached_backtest(version: str, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    with connect() as con:
        return load_backtest(con, run_id)


@st.cache_data(show_spinner=False)
def cached_bookmakers(version: str) -> list[str]:
    odds = cached_latest_odds(version, True)
    return sorted(odds["bookmaker_key"].unique()) if not odds.empty else []
