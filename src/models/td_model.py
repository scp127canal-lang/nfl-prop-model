"""Calibrated LightGBM classifiers for ``anytime_td`` and ``first_td``.

Features (all point-in-time: EWMA over the player's games strictly before the target game, or
pre-game market info):

* red-zone snap share: on field (``participation``) for team plays inside the 20. Where
  participation is missing (e.g. the current season) that game's offensive snap share stands in,
  and ``rz_snap_proxy_ewm5`` records how much of the EWMA came from the stand-in
* inside-the-10 carries & targets, red-zone targets/carries/touches
* usage & efficiency from ``player_features`` (spillover-adjusted shares, snap share, route
  participation, YPRR, RYOE/carry, historical TDs/game)
* market: implied team total and spread from the schedule lines, plus home flag

Labels: ``anytime_td`` = the player is ``td_player_id`` on any TD play in the game; ``first_td`` = he
scored the game's first TD (a defensive/special-teams first TD makes every offensive player 0).
Training uses only games the player actually appeared in (books void props for inactive
players).

Calibration: rows are split by date. LightGBM (monotone constraints on the usage and market
features) is fit on the earlier part, then a Platt (logistic) map on the raw logit is fit on the
later held-out part. Outputs are clipped to [1e-4, 1 - 1e-4], so 0 < p < 1 always holds. For
``first_td`` each game's probabilities are capped so they sum to at most
1 - P(first TD is non-offensive).
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "nfl_model.duckdb"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models"

TARGETS = ("anytime_td", "first_td")
SPANS = (3, 5)
P_MIN, P_MAX = 1e-4, 1 - 1e-4
PARTICIPATION_MIN_COVERAGE = 0.8
RZ_STATE_METRICS = ("rz_snap_share", "rz_snap_proxy", "i10_carries", "i10_targets", "tds")

MONOTONE_UP = {"implied_team_total", "rz_snap_share_ewm3", "rz_snap_share_ewm5", "i10_carries_ewm3",
               "i10_carries_ewm5", "i10_targets_ewm3", "i10_targets_ewm5", "rz_targets_ewm5", "rz_carries_ewm5",
               "rz_touches_ewm5", "target_share_ewm5_adj", "carry_share_ewm5_adj", "tds_ewm5"}
FEATURES = [
    "rz_snap_share_ewm3", "rz_snap_share_ewm5", "rz_snap_proxy_ewm5",
    "i10_carries_ewm3", "i10_carries_ewm5", "i10_targets_ewm3", "i10_targets_ewm5",
    "rz_targets_ewm5", "rz_carries_ewm5", "rz_touches_ewm5",
    "target_share_ewm5_adj", "carry_share_ewm5_adj", "snap_share_ewm5", "route_participation_ewm5",
    "yprr_ewm5", "ryoe_per_carry_ewm5", "tds_ewm3", "tds_ewm5", "prior_games",
    "implied_team_total", "team_spread", "is_home", "is_qb", "is_rb", "is_wr", "is_te",
]
DEFAULT_PARAMS = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 100,
                  "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5.0,
                  "verbose": -1, "seed": 11}


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                            [table]).fetchall()[0][0])


# ======================================================================================
# Dataset
# ======================================================================================
def red_zone_game_stats_sql(has_participation: bool) -> str:
    part = """
    part AS (SELECT game_id, CAST(play_id AS DOUBLE) AS play_id, offense_players FROM participation
             WHERE offense_players IS NOT NULL AND offense_players <> ''),
    part_cov AS (SELECT b.game_id, b.posteam AS team, COUNT(*) AS part_rz_plays
                 FROM base b JOIN part p USING (game_id, play_id) WHERE b.yardline_100 <= 20 GROUP BY ALL),
    rz_snaps AS (
        SELECT game_id, team, gsis_id, COUNT(*) AS rz_snaps FROM (
            SELECT b.game_id, b.posteam AS team, UNNEST(string_split(p.offense_players, ';')) AS gsis_id
            FROM base b JOIN part p USING (game_id, play_id) WHERE b.yardline_100 <= 20)
        GROUP BY ALL),""" if has_participation else """
    part_cov AS (SELECT NULL::VARCHAR AS game_id, NULL::VARCHAR AS team, NULL::BIGINT AS part_rz_plays WHERE FALSE),
    rz_snaps AS (SELECT NULL::VARCHAR AS game_id, NULL::VARCHAR AS team, NULL::VARCHAR AS gsis_id,
                        NULL::BIGINT AS rz_snaps WHERE FALSE),"""
    return f"""
    WITH base AS (
        SELECT * FROM pbp WHERE season_type IN ('REG', 'POST') AND posteam IS NOT NULL
          AND play_type IN ('pass', 'run') AND COALESCE(two_point_attempt, 0) = 0
    ),
    {part}
    team_rz AS (SELECT game_id, posteam AS team, COUNT(*) FILTER (WHERE yardline_100 <= 20) AS team_rz_plays
                FROM base GROUP BY ALL),
    i10 AS (
        SELECT game_id, team, gsis_id, SUM(car) AS i10_carries, SUM(tgt) AS i10_targets FROM (
            SELECT game_id, posteam AS team, rusher_player_id AS gsis_id, 1 AS car, 0 AS tgt FROM base
            WHERE rush_attempt = 1 AND yardline_100 <= 10 AND rusher_player_id IS NOT NULL
            UNION ALL
            SELECT game_id, posteam, receiver_player_id, 0, 1 FROM base
            WHERE pass_attempt = 1 AND COALESCE(sack, 0) = 0 AND yardline_100 <= 10 AND receiver_player_id IS NOT NULL)
        GROUP BY ALL),
    tds AS (SELECT game_id, td_player_id AS gsis_id, COUNT(*) AS tds FROM pbp
            WHERE touchdown = 1 AND td_player_id IS NOT NULL GROUP BY ALL),
    first_td AS (SELECT game_id, arg_min(td_player_id, play_id) AS first_td_player FROM pbp
                 WHERE touchdown = 1 AND td_player_id IS NOT NULL GROUP BY 1),
    snaps AS (SELECT game_id, team, gsis_id, MAX(offense_pct) AS offense_pct, MAX(offense_snaps) AS offense_snaps
              FROM snap_counts WHERE gsis_id IS NOT NULL GROUP BY ALL),
    apps AS (SELECT game_id, team, gsis_id FROM player_weekly WHERE gsis_id IS NOT NULL
             UNION SELECT game_id, team, gsis_id FROM snaps WHERE offense_snaps > 0)
    SELECT a.game_id, a.team, a.gsis_id, s.game_date,
           tr.team_rz_plays, pc.part_rz_plays, COALESCE(rs.rz_snaps, 0) AS rz_snaps, sn.offense_pct,
           COALESCE(i.i10_carries, 0) AS i10_carries, COALESCE(i.i10_targets, 0) AS i10_targets,
           COALESCE(t.tds, 0) AS tds, (COALESCE(t.tds, 0) > 0)::INT AS anytime_td,
           (f.first_td_player IS NOT NULL AND f.first_td_player = a.gsis_id)::INT AS first_td
    FROM apps a
    JOIN schedules s ON s.game_id = a.game_id
    LEFT JOIN team_rz tr ON tr.game_id = a.game_id AND tr.team = a.team
    LEFT JOIN part_cov pc ON pc.game_id = a.game_id AND pc.team = a.team
    LEFT JOIN rz_snaps rs ON rs.game_id = a.game_id AND rs.team = a.team AND rs.gsis_id = a.gsis_id
    LEFT JOIN snaps sn ON sn.game_id = a.game_id AND sn.team = a.team AND sn.gsis_id = a.gsis_id
    LEFT JOIN i10 i ON i.game_id = a.game_id AND i.team = a.team AND i.gsis_id = a.gsis_id
    LEFT JOIN tds t ON t.game_id = a.game_id AND t.gsis_id = a.gsis_id
    LEFT JOIN first_td f ON f.game_id = a.game_id
    """


def red_zone_game_stats(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(red_zone_game_stats_sql(_table_exists(con, "participation"))).df()
    df["game_date"] = pd.to_datetime(df["game_date"]).astype("datetime64[ns]")
    num = ["team_rz_plays", "part_rz_plays", "rz_snaps", "offense_pct", "i10_carries", "i10_targets", "tds",
           "anytime_td", "first_td"]
    df[num] = df[num].astype("float64")
    covered = (df["part_rz_plays"].fillna(0) >= PARTICIPATION_MIN_COVERAGE * df["team_rz_plays"]) & (df["team_rz_plays"] > 0)
    real = df["rz_snaps"] / df["part_rz_plays"].where(df["part_rz_plays"] > 0)
    df["rz_snap_share"] = np.where(covered, real.clip(0, 1), df["offense_pct"]).astype(float)
    df.loc[df["team_rz_plays"].fillna(0) == 0, "rz_snap_share"] = np.nan  # no red-zone plays that game
    df["rz_snap_proxy"] = (~covered).astype(float)
    return df


def _asof_ewma(stats: pd.DataFrame, panel: pd.DataFrame, cols: Sequence[str], spans: Sequence[int]) -> pd.DataFrame:
    """EWMA after each player-game, attached to panel rows from games strictly before them."""
    stats = stats.sort_values(["gsis_id", "game_date", "game_id"]).reset_index(drop=True)
    state = stats[["gsis_id", "game_date"]].rename(columns={"game_date": "state_date"}).copy()
    vals = stats[list(cols)].astype(float)
    vals["gsis_id"] = stats["gsis_id"]
    for span in spans:
        ew = (vals.groupby("gsis_id", sort=False)[list(cols)].ewm(span=span, min_periods=1).mean()
              .reset_index(level=0, drop=True).sort_index())
        for c in cols:
            state[f"{c}_ewm{span}"] = ew[c].to_numpy()
    panel = panel.assign(game_date=pd.to_datetime(panel["game_date"]).astype("datetime64[ns]")).sort_values("game_date")
    state = state.sort_values("state_date")
    out = pd.merge_asof(panel, state, left_on="game_date", right_on="state_date", by="gsis_id",
                        allow_exact_matches=False, direction="backward")
    return out.drop(columns=["state_date"])


def build_td_dataset(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """One row per player_features row (eligible or appeared) with TD features and, for games the
    player appeared in, labels (``played`` = 1)."""
    rz = red_zone_game_stats(con)
    pf = con.execute("""
        SELECT f.*, s.home_team, s.spread_line, s.total_line
        FROM player_features f JOIN schedules s USING (game_id)""").df()
    pf = pf[pf["position"].isin(["QB", "RB", "FB", "WR", "TE"])].copy()
    pf["team_spread"] = np.where(pf["team"] == pf["home_team"], pf["spread_line"], -pf["spread_line"])
    pf["implied_team_total"] = (pf["total_line"] + pf["team_spread"]) / 2
    pf["is_home"] = (pf["team"] == pf["home_team"]).astype(int)
    for pos, col in (("QB", "is_qb"), ("RB", "is_rb"), ("WR", "is_wr"), ("TE", "is_te")):
        pf[col] = (pf["position"] == pos).astype(int)
    pf.loc[pf["position"] == "FB", "is_rb"] = 1
    df = _asof_ewma(rz, pf, RZ_STATE_METRICS, SPANS)
    labels = rz[["game_id", "gsis_id", "anytime_td", "first_td"]].drop_duplicates(["game_id", "gsis_id"])
    df = df.merge(labels.assign(played=1), on=["game_id", "gsis_id"], how="left")
    df["played"] = df["played"].fillna(0).astype(int)
    return df.sort_values(["game_date", "game_id", "gsis_id"]).reset_index(drop=True)


# ======================================================================================
# Model
# ======================================================================================
def _fit_platt(raw: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    z = logit(np.clip(raw, 1e-6, 1 - 1e-6))

    def nll(ab):
        p = np.clip(expit(ab[0] * z + ab[1]), 1e-12, 1 - 1e-12)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = minimize(nll, x0=np.array([1.0, 0.0]), method="BFGS")
    return float(res.x[0]), float(res.x[1])


@dataclass
class CalibratedTDModel:
    target: str
    features: list[str]
    booster_str: str = ""
    platt_a: float = 1.0
    platt_b: float = 0.0
    first_td_cap: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)
    _booster: Any = field(default=None, repr=False, compare=False)

    @property
    def booster(self):
        if self._booster is None:
            import lightgbm as lgb

            self._booster = lgb.Booster(model_str=self.booster_str)
        return self._booster

    def raw_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.booster.predict(df.reindex(columns=self.features).astype(float).to_numpy())

    def predict_proba(self, df: pd.DataFrame, normalize_games: bool = True) -> np.ndarray:
        z = logit(np.clip(self.raw_proba(df), 1e-6, 1 - 1e-6))
        p = np.clip(expit(self.platt_a * z + self.platt_b), P_MIN, P_MAX)
        if self.target == "first_td" and normalize_games and "game_id" in df:
            s = pd.Series(p, index=df.index).groupby(df["game_id"]).transform("sum").to_numpy()
            p = np.clip(np.where(s > self.first_td_cap, p * self.first_td_cap / s, p), P_MIN, P_MAX)
        return p

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {k: v for k, v in self.__dict__.items() if k != "_booster"}
        Path(path).write_bytes(pickle.dumps(state))

    @classmethod
    def load(cls, path: Path) -> "CalibratedTDModel":
        return cls(**pickle.loads(Path(path).read_bytes()))


def train_td_model(df: pd.DataFrame, target: str, before: Any = None, cal_frac: float = 0.25,
                   params: dict[str, Any] | None = None, num_boost_round: int = 400,
                   features: Sequence[str] = FEATURES) -> CalibratedTDModel:
    """Fit on played rows with ``game_date < before``; Platt-calibrate on the latest ``cal_frac`` of dates."""
    import lightgbm as lgb

    if target not in TARGETS:
        raise ValueError(f"target must be one of {TARGETS}")
    d = df[(df["played"] == 1) & df[target].notna()]
    if before is not None:
        d = d[pd.to_datetime(d["game_date"]) < pd.Timestamp(before)]
    dates = np.sort(d["game_date"].unique())
    if len(dates) < 4 or d[target].nunique() < 2:
        raise ValueError(f"Not enough data to train {target}: {len(d)} rows, {len(dates)} dates")
    cal_start = dates[int(len(dates) * (1 - cal_frac))]
    fit, cal = d[d["game_date"] < cal_start], d[d["game_date"] >= cal_start]
    if fit[target].nunique() < 2 or cal[target].nunique() < 2:
        raise ValueError(f"{target}: both classes needed in fit and calibration windows")
    feats = [f for f in features if f in d.columns]
    p = {**DEFAULT_PARAMS, **(params or {}),
         "monotone_constraints": [1 if f in MONOTONE_UP else 0 for f in feats]}
    booster = lgb.train(p, lgb.Dataset(fit[feats].astype(float).to_numpy(), fit[target].to_numpy()),
                        num_boost_round=num_boost_round)
    m = CalibratedTDModel(target=target, features=feats, booster_str=booster.model_to_string())
    m._booster = booster
    m.platt_a, m.platt_b = _fit_platt(m.raw_proba(cal), cal[target].to_numpy(float))
    if target == "first_td":
        per_game = d.groupby("game_id")[target].sum()
        m.first_td_cap = float(np.clip(per_game.mean(), 0.5, 1.0))  # share of games with an offensive first TD
    m.meta = {"n_fit": len(fit), "n_cal": len(cal), "base_rate": float(d[target].mean()),
              "trained_through": str(pd.to_datetime(d["game_date"]).max().date()),
              "cal_start": str(pd.Timestamp(cal_start).date())}
    return m


def evaluate(model: CalibratedTDModel, df: pd.DataFrame) -> dict[str, float]:
    d = df[(df["played"] == 1) & df[model.target].notna()]
    y = d[model.target].to_numpy(float)
    p = model.predict_proba(d)
    out = {"n": len(d), "base_rate": float(y.mean()), "mean_pred": float(p.mean()),
           "brier": float(np.mean((p - y) ** 2)), "brier_baseline": float(np.mean((y.mean() - y) ** 2)),
           "logloss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))}
    bins = pd.qcut(p, 10, labels=False, duplicates="drop")
    out["calibration"] = pd.DataFrame({"p": p, "y": y, "bin": bins}).groupby("bin").agg(
        mean_pred=("p", "mean"), observed=("y", "mean"), n=("y", "size")).reset_index(drop=True)
    return out


def predict_game(df: pd.DataFrame, game_id: str, models: dict[str, CalibratedTDModel]) -> pd.DataFrame:
    rows = df[(df["game_id"] == game_id) & df["eligible"] & ~df["is_out"].fillna(False).astype(bool)].copy()
    for target, m in models.items():
        rows[f"p_{target}"] = m.predict_proba(rows)
    cols = ["game_id", "team", "gsis_id", "player_name", "position"] + [f"p_{t}" for t in models]
    return rows[cols].sort_values(f"p_{next(iter(models))}", ascending=False).reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train calibrated TD models and score upcoming games.")
    p.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    p.add_argument("--holdout-season", type=int, default=None, help="Train before this season, report metrics on it.")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with duckdb.connect(str(args.db_path)) as con:
        df = build_td_dataset(con)
        if args.holdout_season:
            start = pd.to_datetime(df.loc[df["season"] == args.holdout_season, "game_date"]).min()
            for t in TARGETS:
                m = train_td_model(df, t, before=start)
                ev = evaluate(m, df[df["season"] == args.holdout_season])
                cal = ev.pop("calibration")
                print(t, {k: round(v, 4) if isinstance(v, float) else v for k, v in ev.items()})
                print(cal.round(4).to_string())
        models = {t: train_td_model(df, t) for t in TARGETS}
        for t, m in models.items():
            m.save(args.model_dir / f"td_{t}.pkl")
        upcoming = df[(df["played"] == 0) & df["eligible"] & (pd.to_datetime(df["game_date"]) >=
                      pd.to_datetime(df.loc[df["played"] == 1, "game_date"]).max())]
        upcoming = upcoming[~upcoming["is_out"].fillna(False).astype(bool)].copy()
        for t, m in models.items():
            upcoming[f"p_{t}"] = m.predict_proba(upcoming)
        out = upcoming[["season", "week", "game_id", "game_date", "team", "gsis_id", "player_name", "position",
                        "p_anytime_td", "p_first_td"]]
        con.register("_td", out)
        con.execute("CREATE OR REPLACE TABLE td_predictions AS SELECT * FROM _td")
        logger.info("Wrote %d upcoming td_predictions", len(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
