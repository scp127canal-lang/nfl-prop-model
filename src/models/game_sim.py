"""Market-anchored game model + vectorized Monte Carlo player simulator.

1. **Game model** (``GameModel``): per team-game predicts
   * team points, as the market-implied total ``(total + team_spread) / 2`` plus a Ridge/LightGBM
     residual capped at ``MAX_POINTS_DEVIATION``, so the model can lean on the market but never
     override it;
   * offensive plays (dropbacks + designed runs) and pass rate (logit) from team_features plus market
     spread/total.
   Residual spreads and game-script slopes (how plays / pass rate react to the realized margin)
   are estimated from the training residuals.

2. **Simulator** (``simulate``): 10,000 trials, fully vectorized over trials.
   * Team TDs / FGs ~ Poisson around predicted points -> realized margin -> game script.
   * Plays ~ Normal(+script), dropbacks ~ Binomial(plays, pass rate(script)), sacks / scrambles
     ~ Binomial, pass attempts = dropbacks - sacks - scrambles.
   * Player targets & carries are Negative-Binomial-type (Gamma-mixed) counts. By default
     (``allocation="constrained"``) the team's simulated targets (Binomial(attempts, target rate))
     and carries are split by Multinomial(share_i x Gamma(k_i)), so players are overdispersed
     like an NB but always sum to the team total. ``allocation="independent_nb"`` draws each
     player from an NB(mean = share x team volume) on its own; that overstated QB completion /
     yardage spread by ~30% against 2021-25 data, so it isn't the default.
     Receptions ~ Binomial(targets, catch rate), i.e. a thinned NB.
   * Yards per reception ~ Log-Normal, yards per carry ~ shifted Gamma, drawn per event so totals
     **and** longest-play markets come from the same draws.
   * Pass TDs go to receivers in proportion to each trial's receptions x red-zone factor, and rush
     TDs to rushers by carries x red-zone factor. QB lines are the sums over his receivers
     (completions = sum of receptions, pass yards = sum of receiving yards).

Depth-chart mode (``simulate_game(..., depth={...})``, used by the dashboard): the user's
inactive list, explicit re-activations and starting-QB picks are turned into overrides
(``depth_overrides``) and :func:`apply_usage_overrides` then

* zeroes every inactive player and hands his target / carry share to active teammates in
  proportion to their share (same position group weighted 2x; carries only to RB/FB);
* promotes a replacement QB when the starter is out (depth chart, then snap share) or the user
  picked one. A QB who is not the team's established QB1 throws with ``qb_efficiency`` (default
  0.85x yards per attempt, split as ``catch_rate x f^0.4`` and ``ypr x f^0.6``);
* lowers the team's passing efficiency by ``MISSING_STARTER_PENALTY`` (13.5%) per 0.25 of target
  share of missing WR1 / WR2 / TE1, the same way.

Weather (``simulate_game(..., weather={...})``, from ``src.ingestion.weather_client``), both teams:

* wind above ``WIND_THRESHOLD_MPH`` (15): passing efficiency -1.2% and pass rate -0.8% (relative)
  per mph over the threshold; the lost dropbacks become designed runs (carries = plays - dropbacks);
* precipitation probability >= 50% or temperature < 25°F: a further -3% passing efficiency each
  (-5% when both), weighted toward yards per reception (deep balls) rather than catch rate.
Domes / unavailable forecasts change nothing.

``simulate_game(game_id, usage_overrides=None)`` returns a ``SimResult`` holding the full trial
arrays, so any line (Over X.5, alt X+) is priced with ``prob_over`` / ``prob_at_least``.

Point-in-time: ``simulate_game`` trains the game model on games strictly before the game's date,
uses team/player features (already as-of) and player efficiency from ``player_weekly`` with
``game_date < target``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd
from scipy.special import expit, logit

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "nfl_model.duckdb"

N_TRIALS = 10_000
MAX_POINTS_DEVIATION = 3.0      # model may move a team total at most this far from the market
TD_POINT_SHARE = 0.72           # share of offensive points from TDs (incl. PAT); rest from FGs
POINTS_PER_TD = 6.95
# Gamma shape k (Var = mu + mu^2/k) of player volume given team volume. Method-of-moments on
# 2021-2025 actual targets/carries vs pre-game share x realized team volume.
NB_K_TARGETS = {"WR": 30.0, "TE": 40.0, "RB": 15.0, "FB": 15.0, "QB": 15.0, "OTHER": 15.0}
NB_K_CARRIES = {"RB": 16.0, "FB": 10.0, "QB": 8.0, "WR": 5.0, "TE": 5.0, "OTHER": 8.0}
MAX_TEAM_TARGET_SHARE = 0.97   # eligible players' shares are capped (EWMAs from different weeks
MAX_TEAM_CARRY_SHARE = 0.98    # can sum to > 1 after injuries / role changes)
LOGNORMAL_SIGMA_REC = {"WR": 0.80, "TE": 0.72, "RB": 0.85, "FB": 0.85, "OTHER": 0.80}
RUSH_SD = {"RB": 6.0, "FB": 4.0, "QB": 6.5, "WR": 7.5, "TE": 4.5}
RUSH_SHIFT = 4.0                # shifted Gamma: yards = Gamma(a, scale) - shift (allows losses)
EFFICIENCY_LOOKBACK_GAMES = 32
COLD_START_GAMES = 3            # < this many games with >20% snaps -> blend usage with the role prior
PRIOR = {  # position priors for shrinkage: (value, pseudo-count)
    "catch_rate": {"WR": (0.63, 30), "TE": (0.70, 30), "RB": (0.78, 30), "FB": (0.78, 30), "QB": (0.5, 30)},
    "ypr": {"WR": (12.5, 20), "TE": (10.5, 20), "RB": (7.5, 20), "FB": (7.0, 20), "QB": (8.0, 20)},
    "ypc": {"RB": (4.3, 40), "FB": (3.5, 40), "QB": (5.0, 40), "WR": (6.5, 40), "TE": (4.0, 40)},
    "int_rate": (0.0225, 200),
}
OTHER_PARAMS = {"catch_rate": 0.65, "ypr": 10.0}
ROSTER_ONLY_RESIDUAL_FRACTION = 0.8   # roster-only backups share at most 80% of the unassigned volume
ROSTER_ONLY_MIN_ROOM = 0.04           # ... but always at least this much per team (never a hard zero)
BACKUP_QB_EFFICIENCY = 0.85           # yards-per-attempt multiplier for a non-QB1 starter
MISSING_STARTER_PENALTY = 0.135       # passing efficiency loss per 0.25 target share of missing WR1/WR2/TE1
MISSING_STARTER_UNIT = 0.25
MIN_PASS_EFFICIENCY = 0.6
YPA_CATCH_EXPONENT = 0.4              # f = catch-rate part x yards-per-reception part
REDISTRIBUTE_FLOOR = 0.01             # every active teammate gets a little of a vacated share
SAME_GROUP_WEIGHT = 2.0
QB_STARTER_CARRY_PRIOR = 0.08
QB1_MIN_QUAL_GAMES = 3
WIND_THRESHOLD_MPH = 15.0
WIND_PASS_EFF_PER_MPH = 0.012
WIND_PASS_RATE_PER_MPH = 0.008
WIND_MAX_EXCESS_MPH = 25.0            # cap: 40 mph winds don't extrapolate linearly forever
PRECIP_PROB_THRESHOLD = 50
COLD_TEMP_F = 25.0
WET_OR_COLD_PENALTY = 0.03            # one condition
WET_AND_COLD_PENALTY = 0.05           # both
WEATHER_CATCH_EXPONENT = 0.25         # precip/cold hit yards per catch (deep passing) harder

# Odds API market key -> SimResult stat name
MARKET_STATS = {
    "player_pass_yds": "pass_yds", "player_pass_tds": "pass_tds", "player_pass_completions": "pass_completions",
    "player_pass_attempts": "pass_attempts", "player_pass_interceptions": "pass_interceptions",
    "player_rush_yds": "rush_yds", "player_rush_attempts": "rush_attempts", "player_rush_longest": "rush_longest",
    "player_reception_yds": "reception_yds", "player_receptions": "receptions",
    "player_reception_longest": "reception_longest", "player_anytime_td": "anytime_td",
    "player_1st_td": "first_td",
}


# ======================================================================================
# Regressors (numpy Ridge / LightGBM)
# ======================================================================================
class Ridge:
    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Ridge":
        self.mu_, self.sd_ = X.mean(0), X.std(0)
        self.sd_[self.sd_ == 0] = 1.0
        Z = (X - self.mu_) / self.sd_
        self.b0_ = y.mean()
        self.w_ = np.linalg.solve(Z.T @ Z + self.alpha * np.eye(Z.shape[1]), Z.T @ (y - self.b0_))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.b0_ + ((X - self.mu_) / self.sd_) @ self.w_


class LGBMRegressor:
    def __init__(self, num_boost_round: int = 300, **params: Any):
        self.num_boost_round = num_boost_round
        self.params = {"objective": "regression", "learning_rate": 0.03, "num_leaves": 15,
                       "min_data_in_leaf": 40, "lambda_l2": 5.0, "feature_fraction": 0.8, "verbose": -1,
                       "seed": 7, **params}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LGBMRegressor":
        import lightgbm as lgb

        self.booster_ = lgb.train(self.params, lgb.Dataset(X, y), num_boost_round=self.num_boost_round)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.booster_.predict(X)


def _make_regressor(kind: str, **kw: Any):
    if kind == "ridge":
        return Ridge(**kw)
    if kind == "lgbm":
        return LGBMRegressor(**kw)
    raise ValueError(f"unknown model kind {kind!r}")


# ======================================================================================
# Game dataset & model
# ======================================================================================
MARKET_FEATURES = ["implied_points", "team_spread", "total_line"]
CONTEXT_FEATURES = ["is_home", "rest_days", "opp_rest_days"]
_TEAM_METRICS = ["off_epa_per_dropback", "off_epa_per_rush", "off_success_rate", "def_epa_per_dropback",
                 "def_epa_per_rush", "def_success_rate", "pace_sec_per_play", "off_plays_per_game"]
TEAM_FEATURES = [f"{p}{m}_{w}" for p in ("", "opp_") for m in _TEAM_METRICS for w in (5, 10)]


TEAM_GAME_ACTUALS_TABLE = "team_game_actuals"
# Per (game, offense) realized volume/outcomes aggregated from pbp. The deploy DB ships this
# pre-aggregated table instead of the multi-GB ``pbp`` (scripts/export_deploy_db.py).
TEAM_GAME_ACTUALS_SQL = """
            SELECT game_id, posteam AS team,
                   COUNT(*) AS plays, SUM(qb_dropback) AS dropbacks, SUM(COALESCE(sack, 0)) AS sacks,
                   SUM(COALESCE(qb_scramble, 0)) AS scrambles,
                   SUM(CASE WHEN pass_attempt = 1 AND COALESCE(sack, 0) = 0 THEN 1 ELSE 0 END) AS pass_attempts,
                   SUM(CASE WHEN pass_attempt = 1 AND COALESCE(sack, 0) = 0 AND receiver_player_id IS NOT NULL
                            THEN 1 ELSE 0 END) AS targets,
                   SUM(COALESCE(interception, 0)) AS interceptions,
                   SUM(COALESCE(pass_touchdown, 0)) AS pass_tds, SUM(COALESCE(rush_touchdown, 0)) AS rush_tds
            FROM pbp
            WHERE season_type IN ('REG', 'POST') AND posteam IS NOT NULL AND play_type IN ('pass', 'run')
              AND COALESCE(two_point_attempt, 0) = 0
            GROUP BY ALL
"""


def build_game_dataset(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """team_features + market lines + realized outcomes (NULL for unplayed games)."""
    has_pbp = bool(con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'pbp'").fetchone()[0])
    actual = TEAM_GAME_ACTUALS_SQL if has_pbp else f"SELECT * FROM {TEAM_GAME_ACTUALS_TABLE}"
    return con.execute(f"""
        WITH actual AS (
{actual}
        )
        SELECT f.*,
               CASE WHEN f.team = s.home_team THEN s.spread_line ELSE -s.spread_line END AS team_spread,
               s.total_line,
               (s.total_line + CASE WHEN f.team = s.home_team THEN s.spread_line ELSE -s.spread_line END) / 2
                   AS implied_points,
               CASE WHEN f.team = s.home_team THEN s.home_score ELSE s.away_score END AS points,
               CASE WHEN f.team = s.home_team THEN s.away_score ELSE s.home_score END AS opp_points,
               a.* EXCLUDE (game_id, team)
        FROM team_features f
        JOIN schedules s USING (game_id)
        LEFT JOIN actual a ON a.game_id = f.game_id AND a.team = f.team
        ORDER BY f.game_date, f.game_id, f.team
    """).df()


@dataclass
class GameModel:
    kind: str = "ridge"
    alpha_points: float = 200.0     # heavy shrinkage -> stays close to the market
    alpha_volume: float = 10.0
    feature_cols: list[str] = field(default_factory=lambda: MARKET_FEATURES + CONTEXT_FEATURES + TEAM_FEATURES)
    fill_values: dict[str, float] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)
    resid: dict[str, float] = field(default_factory=dict)
    league: dict[str, float] = field(default_factory=dict)
    trained_through: Any = None
    n_train: int = 0

    def _X(self, df: pd.DataFrame) -> np.ndarray:
        X = df.reindex(columns=self.feature_cols).astype(float)
        if "rest_days" in X:
            X["rest_days"] = X["rest_days"].clip(3, 21)
            X["opp_rest_days"] = X["opp_rest_days"].clip(3, 21)
        return X.fillna(self.fill_values).fillna(0.0).to_numpy()

    def fit(self, df: pd.DataFrame) -> "GameModel":
        d = df.dropna(subset=["points", "plays", "dropbacks", "implied_points"])
        d = d[d["plays"] > 0]
        if len(d) < 10:
            raise ValueError(f"Need at least 10 played team-games with market lines to fit; got {len(d)}")
        self.feature_cols = [c for c in self.feature_cols if c in d.columns]
        self.fill_values = d[self.feature_cols].astype(float).mean().fillna(0.0).to_dict()
        self.fill_values.update({"rest_days": 7.0, "opp_rest_days": 7.0})
        X = self._X(d)
        pr = (d["dropbacks"] / d["plays"]).clip(0.2, 0.9).to_numpy()
        y = {"points_resid": (d["points"] - d["implied_points"]).to_numpy(float),
             "plays": d["plays"].to_numpy(float), "logit_pass_rate": logit(pr)}
        for target, values in y.items():
            kw = ({"alpha": self.alpha_points if target == "points_resid" else self.alpha_volume}
                  if self.kind == "ridge" else {})
            self.models[target] = _make_regressor(self.kind, **kw).fit(X, values)
        pred = self.predict(d)
        margin_resid = ((d["points"] - d["opp_points"]) - (pred["pred_points"] - pred["pred_opp_points"])).to_numpy(float)
        r_plays = y["plays"] - pred["pred_plays"].to_numpy()
        r_pr = y["logit_pass_rate"] - logit(pred["pred_pass_rate"].to_numpy())
        ok = np.isfinite(margin_resid)  # opponent row may be missing (e.g. dropped for NaNs)
        if ok.sum() < 10:
            raise ValueError("Too few complete games to estimate game-script effects")
        margin_resid, r_plays, r_pr = margin_resid[ok], r_plays[ok], r_pr[ok]
        var_m = max(np.var(margin_resid), 1e-6)
        # Observed pass rate = latent rate + binomial sampling noise. The simulator draws
        # dropbacks ~ Binomial(plays, rate) itself, so remove the sampling part (delta method on
        # the logit: Var ~= 1 / (n p (1-p))) to avoid counting it twice.
        n_pl = y["plays"][ok]
        binom_var = float(np.mean(1.0 / (n_pl * pr[ok] * (1 - pr[ok]))))
        beta_plays = float(np.cov(r_plays, margin_resid)[0, 1] / var_m)
        beta_pr = float(np.cov(r_pr, margin_resid)[0, 1] / var_m)
        self.resid = {
            "sd_points": float(np.std(y["points_resid"] - (pred["pred_points"] - d["implied_points"]).to_numpy())),
            "beta_plays": beta_plays, "beta_pass_rate": beta_pr,
            "sd_plays": float(np.sqrt(max(np.var(r_plays) - beta_plays ** 2 * var_m, 1.0))),
            "sd_logit_pass_rate": float(np.sqrt(max(np.var(r_pr) - beta_pr ** 2 * var_m - binom_var, 1e-4))),
        }
        db = d["dropbacks"].sum()
        att = d["pass_attempts"].sum()
        tds = (d["pass_tds"] + d["rush_tds"]).sum()
        self.league = {
            "sack_rate": float(d["sacks"].sum() / db), "scramble_rate": float(d["scrambles"].sum() / (db - d["sacks"].sum())),
            "target_rate": float(d["targets"].sum() / att), "int_rate": float(d["interceptions"].sum() / att),
            "pass_td_share": float(d["pass_tds"].sum() / tds) if tds else 0.6,
        }
        self.trained_through = pd.to_datetime(d["game_date"]).max()
        self.n_train = len(d)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._X(df)
        resid = np.clip(self.models["points_resid"].predict(X), -MAX_POINTS_DEVIATION, MAX_POINTS_DEVIATION)
        out = pd.DataFrame({"game_id": df["game_id"].to_numpy(), "team": df["team"].to_numpy()}, index=df.index)
        out["pred_points"] = df["implied_points"].to_numpy(float) + resid
        out["pred_plays"] = np.clip(self.models["plays"].predict(X), 45, 80)
        out["pred_pass_rate"] = np.clip(expit(self.models["logit_pass_rate"].predict(X)), 0.35, 0.80)
        opp = out.set_index(["game_id", "team"])["pred_points"]
        out["pred_opp_points"] = [opp.get((g, o), np.nan) for g, o in zip(df["game_id"], df["opponent"])]
        return out


def fit_game_model(con: duckdb.DuckDBPyConnection, before: date | pd.Timestamp | None = None,
                   kind: str = "ridge") -> GameModel:
    df = build_game_dataset(con)
    if before is not None:
        df = df[pd.to_datetime(df["game_date"]) < pd.Timestamp(before)]
    return GameModel(kind=kind).fit(df)


# ======================================================================================
# Inputs for one game
# ======================================================================================
@dataclass
class TeamParams:
    team: str
    pred_points: float
    pred_plays: float
    pred_pass_rate: float
    sd_plays: float
    sd_logit_pass_rate: float
    beta_plays: float
    beta_pass_rate: float
    sack_rate: float
    scramble_rate: float
    target_rate: float
    int_rate: float
    pass_td_share: float
    sd_points: float | None = None   # target outcome SD of team points; None -> Poisson scoring
    qb_attempt_share: float = 1.0    # share of the team's passing credited to the starting QB


def load_players(con: duckdb.DuckDBPyConnection, game_id: str) -> pd.DataFrame:
    """Active skill players for a game with shares and shrunk efficiency (all as-of game date)."""
    have = {r[0] for r in con.execute("DESCRIBE player_features").fetchall()}
    opt = {"qual_games": "NULL::INTEGER", "target_share_prior": "NULL::DOUBLE", "carry_share_prior": "NULL::DOUBLE",
           "snap_share_expected": "NULL::DOUBLE", "roster_only": "FALSE", "roster_status": "NULL::VARCHAR",
           "depth_rank": "NULL::DOUBLE", "target_share_ewm3": "NULL::DOUBLE", "target_share_ewm5": "NULL::DOUBLE",
           "carry_share_ewm3": "NULL::DOUBLE", "carry_share_ewm5": "NULL::DOUBLE"}
    extra = ", ".join(f"{c if c in have else d} AS {c}" for c, d in opt.items())
    pf = con.execute(f"""
        SELECT gsis_id, player_name, team, position, game_date, is_out, injury_status,
               target_share_ewm3_adj, target_share_ewm5_adj, carry_share_ewm3_adj, carry_share_ewm5_adj,
               snap_share_ewm5, rz_targets_ewm5, rz_carries_ewm5, prior_games, {extra}
        FROM player_features WHERE game_id = ? AND eligible""", [game_id]).df()
    if pf.empty:
        raise ValueError(f"No eligible players in player_features for {game_id}")
    gdate = pd.to_datetime(pf["game_date"].iloc[0]).date()
    ids = pf["gsis_id"].tolist()
    eff = con.execute(f"""
        SELECT gsis_id, SUM(targets) AS tgt, SUM(receptions) AS rec, SUM(receiving_yards) AS rec_yds,
               SUM(carries) AS car, SUM(rushing_yards) AS rush_yds, SUM(attempts) AS att,
               SUM(passing_interceptions) AS ints
        FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY gsis_id ORDER BY game_date DESC) AS rn
              FROM player_weekly WHERE gsis_id IN ({','.join('?' * len(ids))}) AND game_date < ?)
        WHERE rn <= {EFFICIENCY_LOOKBACK_GAMES} GROUP BY 1""", ids + [gdate]).df()
    df = pf.merge(eff, on="gsis_id", how="left").fillna({c: 0.0 for c in ["tgt", "rec", "rec_yds", "car", "rush_yds", "att", "ints"]})

    def shrink(num, den, key):
        out = []
        for n, d_, pos in zip(num, den, df["position"]):
            prior, k = PRIOR[key].get(pos, PRIOR[key]["WR"]) if isinstance(PRIOR[key], dict) else PRIOR[key]
            out.append((n + prior * k) / (d_ + k))
        return np.array(out)

    df["catch_rate"] = np.clip(shrink(df["rec"], df["tgt"], "catch_rate"), 0.3, 0.95)
    df["ypr"] = np.clip(shrink(df["rec_yds"], df["rec"], "ypr"), 3.0, 25.0)
    df["ypc"] = np.clip(shrink(df["rush_yds"], df["car"], "ypc"), 1.5, 9.0)
    prior_int, k_int = PRIOR["int_rate"]
    df["int_rate"] = np.clip((df["ints"] + prior_int * k_int) / (df["att"] + k_int), 0.005, 0.06)
    df["target_share_own"] = df[["target_share_ewm3_adj", "target_share_ewm5_adj"]].mean(axis=1).fillna(0.0)
    df["carry_share_own"] = df[["carry_share_ewm3_adj", "carry_share_ewm5_adj"]].mean(axis=1).fillna(0.0)
    blend_cold_start(df)
    # Usage if he plays, ignoring the injury spillover that zeroed an Out starter: lets the
    # dashboard re-activate a listed player or promote a backup QB with a sensible share.
    w_own = 1.0 - df["usage_prior_weight"]
    for c in ("target_share", "carry_share"):
        raw = df[[f"{c}_ewm3", f"{c}_ewm5"]].mean(axis=1)
        own = raw.where(raw.notna(), df[f"{c}_own"]).fillna(0.0)
        prior = pd.to_numeric(df[f"{c}_prior"], errors="coerce").fillna(0.0)
        df[f"{c}_if_active"] = (w_own * own + (1 - w_own) * prior).where(df["usage_prior_weight"] > 0, own)
    df["roster_only"] = df["roster_only"].fillna(False).astype(bool)
    df["active"] = ~df["is_out"].fillna(False).astype(bool)
    # One starting QB per team: the healthy QB highest on the depth chart, then by recent snaps.
    df["is_starting_qb"] = False
    for team, g in df[(df["position"] == "QB") & df["active"]].groupby("team"):
        df.loc[_qb_order(g).index[0], "is_starting_qb"] = True
    backup_qb = (df["position"] == "QB") & ~df["is_starting_qb"]
    df.loc[backup_qb, ["target_share", "carry_share"]] = 0.0
    df.loc[backup_qb, "active"] = False
    df.loc[~df["active"], ["target_share", "carry_share"]] = 0.0
    _cap_roster_only(df)
    for team in df["team"].unique():
        for col, cap in (("target_share", MAX_TEAM_TARGET_SHARE), ("carry_share", MAX_TEAM_CARRY_SHARE)):
            m = df["team"] == team
            tot = df.loc[m, col].sum()
            if tot > cap:
                df.loc[m, col] *= cap / tot
    return df.drop(columns=["tgt", "rec", "rec_yds", "car", "rush_yds", "att", "ints"]).reset_index(drop=True)


def _qb_order(qbs: pd.DataFrame) -> pd.DataFrame:
    """QBs ordered by depth-chart rank (unranked last), then recent snap share and experience."""
    return qbs.assign(_dr=pd.to_numeric(qbs["depth_rank"], errors="coerce").fillna(99.0),
                      _ss=qbs["snap_share_ewm5"].fillna(0.0)).sort_values(
        ["_dr", "_ss", "prior_games"], ascending=[True, False, False]).drop(columns=["_dr", "_ss"])


def _cap_roster_only(df: pd.DataFrame, frac: float = ROSTER_ONLY_RESIDUAL_FRACTION) -> None:
    """Roster-only backups (no recent snaps) may only use part of the volume not assigned to the
    established players (at least ``ROSTER_ONLY_MIN_ROOM``), so adding the full roster dilutes
    the starters by a few percent at most (in place)."""
    ro = df["roster_only"] & df["active"]
    if not ro.any():
        return
    for team in df.loc[ro, "team"].unique():
        m = df["team"] == team
        for col, cap in (("target_share", MAX_TEAM_TARGET_SHARE), ("carry_share", MAX_TEAM_CARRY_SHARE)):
            regular = df.loc[m & df["active"] & ~df["roster_only"], col].sum()
            room = max(max(cap - regular, 0.0) * frac, ROSTER_ONLY_MIN_ROOM)
            extra = df.loc[m & ro, col].sum()
            if extra > room:
                df.loc[m & ro, col] *= room / extra if extra > 0 else 0.0


def _renormalize(df: pd.DataFrame, col: str, team: str, fixed: set[str], total: float,
                 cap: float = MAX_TEAM_CARRY_SHARE) -> None:
    """Scale non-overridden active players so the team's share sums to ``total`` (<= cap)."""
    m = (df["team"] == team) & df["active"]
    free = m & ~df["gsis_id"].isin(fixed)
    target_total = min(total, cap)
    fixed_sum = df.loc[m & df["gsis_id"].isin(fixed), col].sum()
    free_sum = df.loc[free, col].sum()
    remaining = max(target_total - fixed_sum, 0.0)
    if free_sum > 0:
        df.loc[free, col] *= remaining / free_sum
    if df.loc[m, col].sum() > cap:  # overrides alone exceed the cap
        df.loc[m, col] *= cap / df.loc[m, col].sum()


def blend_cold_start(df: pd.DataFrame, min_games: int = COLD_START_GAMES) -> pd.DataFrame:
    """Shrink thin-history usage toward the role prior (in place).

    weight on own usage = qualifying games / ``min_games`` (capped at 1), so a player with 0, 1, 2
    games above 20% snaps gets 0, 1/3, 2/3 of his own EWMA and the rest from
    ``target/carry_share_prior`` (position x snap-tier league rate x expected snap share, built
    point-in-time in ``player_usage``). Established players are unchanged. Tables built before the
    prior existed (NULL priors) fall back to own usage.
    """
    q = pd.to_numeric(df.get("qual_games"), errors="coerce")
    w = np.clip(q / min_games, 0.0, 1.0)
    has_prior = q.notna() & df["target_share_prior"].notna()
    w = w.where(has_prior, 1.0)
    df["usage_prior_weight"] = (1.0 - w).astype(float)
    df["cold_start"] = has_prior & (q < min_games)
    for c in ("target_share", "carry_share"):
        prior = pd.to_numeric(df[f"{c}_prior"], errors="coerce").fillna(0.0)
        df[c] = (w * df[f"{c}_own"] + (1 - w) * prior).astype(float)
    df["qual_games"] = q.fillna(df["prior_games"]).astype(int)
    return df


def _redistribute(df: pd.DataFrame, team: str, vacated: Mapping[str, float], group: str | None) -> None:
    """Hand vacated target / carry share to the team's active players (in place): in proportion
    to their share (+ a small floor), same position group weighted ``SAME_GROUP_WEIGHT``x;
    targets skip QBs, carries go to RB/FB when the team has any."""
    m = (df["team"] == team) & df["active"]
    pos = df["position"]
    pgroup = pos.replace({"FB": "RB"})
    for col, amount in vacated.items():
        if amount <= 0:
            continue
        if col == "target_share":
            rec = m & (pos != "QB")
        else:
            rec = m & pos.isin(["RB", "FB"])
            rec = rec if rec.any() else m
        if not rec.any():
            continue
        w = (df.loc[rec, col] + REDISTRIBUTE_FLOOR) * np.where(pgroup[rec] == group, SAME_GROUP_WEIGHT, 1.0)
        df.loc[rec, col] += amount * w / w.sum()


def _pass_efficiency(df: pd.DataFrame, team: str, factor: float, skip: set[str]) -> None:
    """Scale the team's receiving efficiency so yards per attempt moves by ``factor`` (in place)."""
    if abs(factor - 1.0) < 1e-9:
        return
    m = (df["team"] == team) & ~df["gsis_id"].isin(skip)
    df.loc[m, "catch_rate"] = np.clip(df.loc[m, "catch_rate"] * factor ** YPA_CATCH_EXPONENT, 0.3, 0.95)
    df.loc[m, "ypr"] = np.clip(df.loc[m, "ypr"] * factor ** (1 - YPA_CATCH_EXPONENT), 3.0, 25.0)


def missing_starter_share(df: pd.DataFrame, team: str) -> tuple[float, list[str]]:
    """Target share (pre-injury EWMA) of the team's WR1 / WR2 / TE1 who are not active."""
    t = df[df["team"] == team]
    usage = t[["target_share_ewm3", "target_share_ewm5"]].mean(axis=1).fillna(t["target_share_if_active"]).fillna(0.0)
    t = t.assign(_u=usage)
    starters = pd.concat([t[t["position"] == "WR"].nlargest(2, "_u"), t[t["position"] == "TE"].nlargest(1, "_u")])
    starters = starters[starters["_u"] > 0]
    gone = starters[~starters["active"]]
    return float(gone["_u"].sum()), gone["player_name"].astype(str).tolist() if "player_name" in gone else []


def team_qb1(df: pd.DataFrame, team: str) -> str | None:
    """The team's established QB1: most recent snap share among QBs with enough qualifying games."""
    q = df[(df["team"] == team) & (df["position"] == "QB")]
    est = q[pd.to_numeric(q["qual_games"], errors="coerce").fillna(0) >= QB1_MIN_QUAL_GAMES]
    pool = est if len(est) else q
    if pool.empty:
        return None
    return str(pool.sort_values(["snap_share_ewm5", "prior_games"], ascending=False, na_position="last")
               ["gsis_id"].iloc[0])


def _set_starting_qb(df: pd.DataFrame, team: str, qb: str, fixed: dict[tuple[str, str], set[str]]) -> None:
    m = (df["team"] == team) & (df["position"] == "QB")
    old = df.index[m & df["is_starting_qb"]]
    df.loc[old, ["target_share", "carry_share"]] = 0.0
    df.loc[old, ["active", "is_starting_qb"]] = False
    i = df.index[df["gsis_id"] == qb][0]
    q = pd.to_numeric(pd.Series([df.at[i, "qual_games"]]), errors="coerce").fillna(0).iloc[0]
    w = min(float(q) / COLD_START_GAMES, 1.0)
    df.at[i, "carry_share"] = w * float(df.at[i, "carry_share_if_active"]) + (1 - w) * QB_STARTER_CARRY_PRIOR
    df.at[i, "target_share"] = float(df.at[i, "target_share_if_active"]) * w
    df.at[i, "active"] = True
    df.at[i, "is_starting_qb"] = True
    fixed[(team, "carry_share")].add(qb)
    fixed[(team, "target_share")].add(qb)


def apply_usage_overrides(players: pd.DataFrame, overrides: Mapping[str, Mapping[str, Any]] | None,
                          teams: Sequence[str], depth_adjust: bool = False
                          ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Apply per-player (keyed by gsis_id) and per-team (keyed by team abbr) overrides.

    Player keys: ``target_share``, ``carry_share``, ``catch_rate``, ``ypr``, ``ypc``, ``active``.
    ``active=False`` zeroes him and hands his share to active teammates (``_redistribute``) unless
    ``redistribute=False``; ``active=True`` brings back a listed-out player at his pre-injury usage.
    Team keys: ``points``, ``plays``, ``pass_rate``, ``starting_qb`` (gsis_id), ``qb_efficiency``,
    ``qb_attempt_share``.
    Explicit shares are fixed and the rest of the team is rescaled so team totals are unchanged.

    ``depth_adjust`` (dashboard depth mode) also auto-promotes a QB when the starter is out and
    applies the backup-QB and missing-WR1/WR2/TE1 efficiency penalties. The second return value
    holds team overrides plus, per team, ``_adjustments`` describing what was applied.
    """
    df = players.copy()
    team_over: dict[str, dict[str, Any]] = {}
    overrides = overrides or {}
    if not overrides and not depth_adjust:
        return df, team_over
    for c in ("target_share_if_active", "carry_share_if_active"):
        if c not in df:
            df[c] = df[c.removesuffix("_if_active")]
    for c in ("target_share_ewm3", "target_share_ewm5", "depth_rank"):
        if c not in df:
            df[c] = np.nan
    fixed: dict[tuple[str, str], set[str]] = {(t, c): set() for t in teams for c in ("target_share", "carry_share")}
    explicit_eff: set[str] = set()
    player_ov: dict[int, Mapping[str, Any]] = {}
    for key, ov in overrides.items():
        if key in teams:
            team_over[key] = dict(ov)
            continue
        idx = df.index[df["gsis_id"] == key]
        if len(idx) == 0:
            raise KeyError(f"usage_overrides: {key!r} is not an eligible player or team in this game")
        player_ov[idx[0]] = ov

    # 1) inactive players: zero, then hand their volume to active teammates
    vacated: dict[str, list[tuple[str, dict[str, float]]]] = {}
    changed: dict[str, list[str]] = {t: [] for t in teams}
    for i, ov in player_ov.items():
        if ov.get("active") is False:
            team = df.at[i, "team"]
            if df.at[i, "active"]:
                changed.setdefault(team, []).append(str(df.at[i, "gsis_id"]))
            shares = {c: float(df.at[i, c]) if df.at[i, "active"] else 0.0 for c in ("target_share", "carry_share")}
            df.loc[i, ["target_share", "carry_share"]] = 0.0
            df.at[i, "active"] = False
            if "snap_share_ewm5" in df:
                df.at[i, "snap_share_ewm5"] = 0.0
            if df.at[i, "position"] == "QB" and df.at[i, "is_starting_qb"]:
                df.at[i, "is_starting_qb"] = False
                shares["target_share"] = 0.0          # a QB's volume goes to his replacement
                shares["carry_share"] = 0.0
            if ov.get("redistribute", True):
                vacated.setdefault(team, []).append((df.at[i, "position"], shares))
    for team, items in vacated.items():
        for pos, shares in items:
            _redistribute(df, team, shares, "RB" if pos == "FB" else pos)
    totals = {(t, c): df.loc[(df["team"] == t) & df["active"], c].sum() for t in teams
              for c in ("target_share", "carry_share")}

    # 2) re-activated (listed-out) players come back at their pre-injury usage
    for i, ov in player_ov.items():
        if ov.get("active") is True and not df.at[i, "active"]:
            team = df.at[i, "team"]
            df.at[i, "active"] = True
            if df.at[i, "position"] != "QB":
                changed.setdefault(team, []).append(str(df.at[i, "gsis_id"]))
            if df.at[i, "position"] != "QB":
                for c in ("target_share", "carry_share"):
                    df.at[i, c] = float(df.at[i, f"{c}_if_active"])
                    fixed[(team, c)].add(df.at[i, "gsis_id"])
            else:
                df.at[i, "active"] = False      # a QB only plays as the starter (below)
                df.at[i, "is_out"] = False

    # 3) starting QB: explicit pick, else auto-promotion when the starter is out
    adjustments: dict[str, dict[str, Any]] = {t: {} for t in teams}
    for team in teams:
        tov = team_over.get(team, {})
        qb = tov.get("starting_qb")
        if qb is not None and not ((df["gsis_id"] == qb) & (df["team"] == team) & (df["position"] == "QB")).any():
            raise KeyError(f"starting_qb {qb!r} is not a QB of {team} in this game")
        has_starter = ((df["team"] == team) & df["is_starting_qb"] & df["active"]).any()
        back = [df.at[i, "gsis_id"] for i, o in player_ov.items() if o.get("active") is True
                and df.at[i, "team"] == team and df.at[i, "position"] == "QB"]
        if qb is None and back and team_qb1(df, team) in back:
            qb = team_qb1(df, team)         # the re-activated QB1 starts again
        if qb is None and depth_adjust and not has_starter:
            pool = df[(df["team"] == team) & (df["position"] == "QB") & ~df["is_out"].fillna(False).astype(bool)]
            pool = pool[~pool.index.isin([i for i, o in player_ov.items() if o.get("active") is False])]
            if len(pool):
                qb = str(_qb_order(pool)["gsis_id"].iloc[0])
        if qb is not None:
            current = df.loc[(df["team"] == team) & df["is_starting_qb"] & df["active"], "gsis_id"].tolist()
            if current != [qb]:
                _set_starting_qb(df, team, qb, fixed)
                changed.setdefault(team, []).extend(current + [qb])
            adjustments[team]["starting_qb"] = qb

    # 4) explicit per-player shares / efficiency
    for i, ov in player_ov.items():
        if ov.get("active") is False:
            continue
        team, pid = df.at[i, "team"], df.at[i, "gsis_id"]
        for c in ("target_share", "carry_share"):
            if c in ov:
                df.at[i, c] = float(ov[c])
                fixed[(team, c)].add(pid)
        for c in ("catch_rate", "ypr", "ypc"):
            if c in ov:
                df.at[i, c] = float(ov[c])
                explicit_eff.add(pid)
    for (team, c), total in totals.items():
        _renormalize(df, c, team, fixed[(team, c)], total)

    # 5) passing efficiency: backup QB and missing starting receivers
    for team in teams:
        tov = team_over.get(team, {})
        starter = df.loc[(df["team"] == team) & df["is_starting_qb"] & df["active"], "gsis_id"].tolist()
        f_qb = 1.0
        if depth_adjust or "starting_qb" in tov:
            qb1 = team_qb1(df, team)
            if starter and qb1 is not None and starter[0] != qb1:
                f_qb = float(tov.get("qb_efficiency", BACKUP_QB_EFFICIENCY))
        f_miss, missing = 1.0, []
        if depth_adjust:
            share, missing = missing_starter_share(df, team)
            f_miss = max(1.0 - MISSING_STARTER_PENALTY * share / MISSING_STARTER_UNIT, MIN_PASS_EFFICIENCY)
        factor = max(f_qb * f_miss, MIN_PASS_EFFICIENCY)
        _pass_efficiency(df, team, factor, explicit_eff)
        adjustments[team].update({"qb_efficiency": f_qb, "missing_starter_factor": f_miss,
                                  "missing_starters": missing, "pass_efficiency": factor,
                                  "qb_attempt_share": float(tov.get("qb_attempt_share", 1.0)),
                                  "changed_players": changed.get(team, [])})
        if starter:
            adjustments[team].setdefault("starting_qb", starter[0])
    for team in teams:
        team_over.setdefault(team, {})["_adjustments"] = adjustments[team]
    return df, team_over


def weather_factors(weather: Mapping[str, Any] | None) -> dict[str, float]:
    """Multipliers implied by a weather dict: ``pass_efficiency`` (yards per attempt), its
    ``wet_cold`` part, and ``pass_rate``. Neutral (all 1.0) for domes or missing forecasts."""
    out = {"wind_pass_efficiency": 1.0, "wet_cold": 1.0, "pass_rate": 1.0}
    if not weather or weather.get("is_dome"):
        return out
    wind = float(weather.get("wind_mph") or 0.0)
    excess = min(max(wind - WIND_THRESHOLD_MPH, 0.0), WIND_MAX_EXCESS_MPH)
    out["wind_pass_efficiency"] = 1.0 - WIND_PASS_EFF_PER_MPH * excess
    out["pass_rate"] = 1.0 - WIND_PASS_RATE_PER_MPH * excess
    wet = float(weather.get("precip_prob") or 0) >= PRECIP_PROB_THRESHOLD
    temp = weather.get("temp_f")
    cold = temp is not None and float(temp) < COLD_TEMP_F
    if wet and cold:
        out["wet_cold"] = 1.0 - WET_AND_COLD_PENALTY
    elif wet or cold:
        out["wet_cold"] = 1.0 - WET_OR_COLD_PENALTY
    return out


def apply_weather(players: pd.DataFrame, params: Mapping[str, "TeamParams"], weather: Mapping[str, Any] | None
                  ) -> tuple[pd.DataFrame, dict[str, float]]:
    """Scale both teams' receiving efficiency and pass rate for the game's weather (copies)."""
    f = weather_factors(weather)
    df = players.copy()
    if f == {"wind_pass_efficiency": 1.0, "wet_cold": 1.0, "pass_rate": 1.0}:
        return df, f
    for team, p in params.items():
        _pass_efficiency(df, team, f["wind_pass_efficiency"], set())
        if f["wet_cold"] != 1.0:
            m = df["team"] == team
            w = f["wet_cold"]
            df.loc[m, "catch_rate"] = np.clip(df.loc[m, "catch_rate"] * w ** WEATHER_CATCH_EXPONENT, 0.3, 0.95)
            df.loc[m, "ypr"] = np.clip(df.loc[m, "ypr"] * w ** (1 - WEATHER_CATCH_EXPONENT), 3.0, 25.0)
        p.pred_pass_rate = float(np.clip(p.pred_pass_rate * f["pass_rate"], 0.30, 0.80))
    return df, f


def depth_overrides(players: pd.DataFrame, inactive: Sequence[str] = (), reactivate: Sequence[str] = (),
                    starting_qbs: Mapping[str, str] | None = None, qb_efficiency: float = BACKUP_QB_EFFICIENCY,
                    qb_attempt_share: float = 1.0) -> dict[str, dict[str, Any]]:
    """Dashboard depth spec (global lists of gsis_ids, team -> QB) -> overrides for one game.

    Ids that are not in this game are ignored, so one spec can be applied to every game."""
    ids = set(players["gsis_id"])
    ov: dict[str, dict[str, Any]] = {}
    for pid in reactivate:
        if pid in ids:
            ov[pid] = {"active": True}
    for pid in inactive:
        if pid in ids:
            ov[pid] = {"active": False}
    teams = set(players["team"])
    qbs = set(players.loc[players["position"] == "QB", "gsis_id"])
    for team, qb in (starting_qbs or {}).items():
        if team in teams and qb in qbs and qb not in set(inactive):
            ov[team] = {"starting_qb": qb, "qb_efficiency": float(qb_efficiency),
                        "qb_attempt_share": float(qb_attempt_share)}
    for team in teams:      # efficiency factor also applies to auto-promoted backups
        ov.setdefault(team, {}).setdefault("qb_efficiency", float(qb_efficiency))
        ov[team].setdefault("qb_attempt_share", float(qb_attempt_share))
    return ov


# ======================================================================================
# Simulation
# ======================================================================================
def _nb(rng: np.random.Generator, mean: np.ndarray, k: float) -> np.ndarray:
    """Negative Binomial via Gamma-Poisson mixture: E = mean, Var = mean + mean^2 / k."""
    mean = np.maximum(mean, 0.0)
    return rng.poisson(rng.gamma(k, 1.0 / k, size=mean.shape) * mean)


def _per_event(rng: np.random.Generator, counts: np.ndarray, sampler) -> tuple[np.ndarray, np.ndarray]:
    """Draw one value per event (masked to ``counts``) -> (sum, max) per trial (0 if no events)."""
    nmax = int(counts.max()) if counts.size else 0
    if nmax == 0:
        z = np.zeros(counts.shape, dtype=float)
        return z, z.copy()
    draws = np.rint(sampler((counts.shape[0], nmax)))
    mask = np.arange(nmax)[None, :] < counts[:, None]
    total = np.where(mask, draws, 0.0).sum(axis=1)
    longest = np.where(mask, draws, -np.inf).max(axis=1)
    return total, np.where(counts > 0, longest, 0.0)


def _allocate(rng: np.random.Generator, n: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Multinomial split of ``n`` (T,) across columns of ``weights`` (T, P); last column = 'other'."""
    w = np.maximum(weights, 0.0)
    s = w.sum(axis=1, keepdims=True)
    p = np.where(s > 0, w / np.where(s > 0, s, 1.0), 0.0)
    p[s[:, 0] == 0, -1] = 1.0
    p[:, :-1] *= 1 - 1e-12
    p[:, -1] = np.clip(1.0 - p[:, :-1].sum(axis=1), 0.0, 1.0)
    return rng.multinomial(n, p)


@dataclass
class SimResult:
    game_id: str
    n_trials: int
    players: pd.DataFrame
    stats: dict[str, dict[str, np.ndarray]]
    teams: dict[str, dict[str, np.ndarray]]
    inputs: dict[str, TeamParams] = field(default_factory=dict)
    adjustments: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get(self, gsis_id: str, stat: str) -> np.ndarray:
        if gsis_id not in self.stats:
            raise KeyError(f"{gsis_id} not simulated in {self.game_id}")
        if stat not in self.stats[gsis_id]:
            raise KeyError(f"stat {stat!r} not available for {gsis_id}; have {sorted(self.stats[gsis_id])}")
        return self.stats[gsis_id][stat]

    def _p(self, hits: int, smooth: bool) -> float:
        return (hits + 0.5) / (self.n_trials + 1.0) if smooth else hits / self.n_trials

    def prob_over(self, gsis_id: str, stat: str, line: float, smooth: bool = True) -> float:
        """P(stat > line). With ``smooth`` (default) the result is strictly inside (0, 1)."""
        return self._p(int((self.get(gsis_id, stat) > line).sum()), smooth)

    def prob_under(self, gsis_id: str, stat: str, line: float, smooth: bool = True) -> float:
        return self._p(int((self.get(gsis_id, stat) < line).sum()), smooth)

    def prob_at_least(self, gsis_id: str, stat: str, k: int, smooth: bool = True) -> float:
        """Alt 'X+' lines: P(stat >= k)."""
        return self._p(int((self.get(gsis_id, stat) >= k).sum()), smooth)

    def price_market(self, market_key: str, gsis_id: str, line: float | None = None, side: str = "over") -> float:
        stat = MARKET_STATS[market_key]
        if stat in ("anytime_td", "first_td"):
            return self.prob_at_least(gsis_id, stat, 1)
        if line is None:
            raise ValueError("line required")
        return self.prob_over(gsis_id, stat, line) if side == "over" else self.prob_under(gsis_id, stat, line)

    @staticmethod
    def fair_american(p: float) -> float:
        return -100 * p / (1 - p) if p >= 0.5 else 100 * (1 - p) / p

    def summary(self) -> pd.DataFrame:
        rows = []
        for pid, stats in self.stats.items():
            meta = self.players.set_index("gsis_id").loc[pid]
            for stat, arr in stats.items():
                rows.append({"gsis_id": pid, "player_name": meta["player_name"], "team": meta["team"],
                             "position": meta["position"], "stat": stat, "mean": arr.mean(),
                             "median": float(np.median(arr)), "p10": float(np.percentile(arr, 10)),
                             "p90": float(np.percentile(arr, 90))})
        return pd.DataFrame(rows)


def _points_var(n: int, lam_td: float, lam_fg: float) -> float:
    """Var of 6.95*TD + 3*FG with TD ~ Bin(n, lam_td/n), FG | TD ~ Bin(n - TD, lam_fg/(n - lam_td))."""
    p_td = lam_td / n
    p_fg = min(lam_fg / max(n - lam_td, 1e-9), 1.0)
    v_td = n * p_td * (1 - p_td)
    v_fg = (n - lam_td) * p_fg * (1 - p_fg) + p_fg ** 2 * v_td
    cov = -p_fg * v_td
    return POINTS_PER_TD ** 2 * v_td + 9 * v_fg + 2 * POINTS_PER_TD * 3 * cov


def scoring_opportunities(lam_td: float, lam_fg: float, sd_points: float | None) -> int | None:
    """Effective number of scoring chances whose binomial scoring variance best matches
    ``sd_points``; ``None`` (Poisson) when the target is at or above Poisson variance."""
    if sd_points is None or not np.isfinite(sd_points):
        return None
    target = sd_points ** 2
    poisson_var = POINTS_PER_TD ** 2 * lam_td + 9 * lam_fg
    if target >= poisson_var:
        return None
    lo = int(np.ceil(lam_td + lam_fg)) + 1
    ns = np.arange(lo, 61)
    return int(ns[np.argmin([abs(_points_var(n, lam_td, lam_fg) - target) for n in ns])])


def simulate_points(team_params: Mapping[str, TeamParams], n_trials: int, rng: np.random.Generator
                    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Team TDs, FGs and points per trial. Mean = predicted points. Dispersion is matched to
    ``sd_points`` (from the game model's residuals) with binomial scoring over an effective
    number of chances, because plain Poisson scoring overstates NFL score variance."""
    tds, fgs, pts = {}, {}, {}
    for t, p in team_params.items():
        mean_pts = max(p.pred_points, 0.5)
        lam_td = TD_POINT_SHARE * mean_pts / POINTS_PER_TD
        lam_fg = (1 - TD_POINT_SHARE) * mean_pts / 3.0
        n = scoring_opportunities(lam_td, lam_fg, p.sd_points)
        if n is None:
            tds[t] = rng.poisson(lam_td, n_trials)
            fgs[t] = rng.poisson(lam_fg, n_trials)
        else:
            tds[t] = rng.binomial(n, lam_td / n, n_trials)
            fgs[t] = rng.binomial(n - tds[t], min(lam_fg / (n - lam_td), 1.0))
        pts[t] = np.rint(tds[t] * POINTS_PER_TD + 3 * fgs[t])
    return tds, fgs, pts


def simulate(players: pd.DataFrame, team_params: Mapping[str, TeamParams], n_trials: int = N_TRIALS,
             seed: int | None = None, game_id: str = "", allocation: str = "constrained") -> SimResult:
    """Pure vectorized simulator (no DB access)."""
    if allocation not in ("constrained", "independent_nb"):
        raise ValueError("allocation must be 'constrained' or 'independent_nb'")
    rng = np.random.default_rng(seed)
    teams = list(team_params)
    if len(teams) != 2:
        raise ValueError("simulate needs exactly two teams")
    T = n_trials
    tds, fgs, pts = simulate_points(team_params, T, rng)

    stats: dict[str, dict[str, np.ndarray]] = {}
    team_out: dict[str, dict[str, np.ndarray]] = {}
    td_counts: list[np.ndarray] = []   # per-player total TDs (both teams, incl. 'other') for first-TD draw
    td_owner: list[str | None] = []
    for t, p in team_params.items():
        o = teams[1 - teams.index(t)]
        margin_resid = (pts[t] - pts[o]) - (p.pred_points - team_params[o].pred_points)
        plays = np.clip(np.rint(p.pred_plays + p.beta_plays * margin_resid + rng.normal(0, p.sd_plays, T)), 30, 100).astype(int)
        pass_rate = expit(logit(p.pred_pass_rate) + p.beta_pass_rate * margin_resid + rng.normal(0, p.sd_logit_pass_rate, T))
        dropbacks = rng.binomial(plays, pass_rate)
        sacks = rng.binomial(dropbacks, p.sack_rate)
        scrambles = rng.binomial(dropbacks - sacks, p.scramble_rate)
        attempts = dropbacks - sacks - scrambles
        team_carries = (plays - dropbacks) + scrambles
        target_mean = attempts * p.target_rate

        roster = players[(players["team"] == t) & players["active"]].reset_index(drop=True)
        tshare = roster["target_share"].to_numpy(float)
        cshare = roster["carry_share"].to_numpy(float)
        other_t = max(1.0 - tshare.sum(), 0.0)
        other_c = max(1.0 - cshare.sum(), 0.0)
        P = len(roster)

        positions = roster["position"].tolist() + ["OTHER"]
        t_shares = np.append(tshare, other_t)
        c_shares = np.append(cshare, other_c)
        kt = np.array([NB_K_TARGETS.get(x, 15.0) for x in positions])
        kc = np.array([NB_K_CARRIES.get(x, 8.0) for x in positions])
        if allocation == "constrained":
            team_targets = rng.binomial(attempts, p.target_rate)
            targets = _allocate(rng, team_targets, rng.gamma(kt, 1.0 / kt, (T, P + 1)) * t_shares)
            carries = _allocate(rng, team_carries, rng.gamma(kc, 1.0 / kc, (T, P + 1)) * c_shares)
        else:
            targets = np.stack([_nb(rng, t_shares[j] * target_mean, kt[j]) for j in range(P + 1)], axis=1)
            carries = np.stack([_nb(rng, c_shares[j] * team_carries, kc[j]) for j in range(P + 1)], axis=1)
        receptions = np.zeros_like(targets)
        rec_yds = np.zeros((T, P + 1))
        rec_long = np.zeros((T, P + 1))
        rush_yds = np.zeros((T, P + 1))
        rush_long = np.zeros((T, P + 1))
        for j in range(P + 1):
            if j < P:
                r = roster.iloc[j]
                ts, cs, cr, ypr, ypc = tshare[j], cshare[j], r["catch_rate"], r["ypr"], r["ypc"]
                pos = r["position"]
            else:
                ts, cs, cr, ypr, ypc, pos = other_t, other_c, OTHER_PARAMS["catch_rate"], OTHER_PARAMS["ypr"], 4.3, "OTHER"
            receptions[:, j] = rng.binomial(targets[:, j], cr)
            sig = LOGNORMAL_SIGMA_REC.get(pos, 0.8)
            mu = np.log(ypr) - sig ** 2 / 2
            rec_yds[:, j], rec_long[:, j] = _per_event(rng, receptions[:, j], lambda s, m=mu, sg=sig: rng.lognormal(m, sg, s))
            sd = RUSH_SD.get(pos, 6.0)
            scale = sd ** 2 / (ypc + RUSH_SHIFT)
            shape = (ypc + RUSH_SHIFT) ** 2 / sd ** 2
            rush_yds[:, j], rush_long[:, j] = _per_event(
                rng, carries[:, j], lambda s, a=shape, sc=scale: rng.gamma(a, sc, s) - RUSH_SHIFT)

        # Touchdowns: team TDs -> pass/rush -> players by this trial's opportunities x RZ factor.
        pass_tds_team = rng.binomial(tds[t], p.pass_td_share)
        rush_tds_team = tds[t] - pass_tds_team
        rz_t = roster["rz_targets_ewm5"].fillna(0).to_numpy(float)
        rz_c = roster["rz_carries_ewm5"].fillna(0).to_numpy(float)
        rz_t_factor = np.append(_rz_factor(rz_t, tshare), 1.0)
        rz_c_factor = np.append(_rz_factor(rz_c, cshare), 1.0)
        rec_tds = _allocate(rng, pass_tds_team, (receptions + 0.05) * rz_t_factor)
        rush_tds = _allocate(rng, rush_tds_team, (carries + 0.05) * rz_c_factor)

        qb_idx = roster.index[roster["is_starting_qb"]].tolist() if "is_starting_qb" in roster else []
        untargeted = rng.binomial(attempts, 1 - p.target_rate)
        qb_attempts = targets.sum(axis=1) + untargeted
        qb_int_rate = float(roster.loc[qb_idx[0], "int_rate"]) if qb_idx else p.int_rate
        for j in range(P):
            pid = roster.at[j, "gsis_id"]
            s = {"targets": targets[:, j], "receptions": receptions[:, j], "reception_yds": rec_yds[:, j],
                 "reception_longest": rec_long[:, j], "rush_attempts": carries[:, j], "rush_yds": rush_yds[:, j],
                 "rush_longest": rush_long[:, j], "rec_tds": rec_tds[:, j], "rush_tds": rush_tds[:, j],
                 "anytime_td": rec_tds[:, j] + rush_tds[:, j]}
            if j in qb_idx:
                completions = receptions.sum(axis=1)
                incompletions = qb_attempts - completions
                # INTs come out of incompletions; per-trial rate keeps E[INT] = int_rate x attempts.
                p_int = np.clip(qb_int_rate * qb_attempts / np.maximum(incompletions, 1), 0.0, 1.0)
                q = {"pass_attempts": qb_attempts, "pass_completions": completions,
                     "pass_yds": rec_yds.sum(axis=1), "pass_tds": pass_tds_team,
                     "pass_interceptions": rng.binomial(incompletions, p_int)}
                share = float(np.clip(p.qb_attempt_share, 0.0, 1.0))
                if share < 1.0:     # another QB takes the rest: thin his counts, yards per completion kept
                    comp = rng.binomial(completions, share)
                    ypc_trial = np.where(completions > 0, q["pass_yds"] / np.maximum(completions, 1), 0.0)
                    q = {"pass_attempts": comp + rng.binomial(qb_attempts - completions, share),
                         "pass_completions": comp, "pass_yds": np.rint(comp * ypc_trial),
                         "pass_tds": rng.binomial(pass_tds_team, share),
                         "pass_interceptions": rng.binomial(q["pass_interceptions"], share)}
                s.update(q)
            stats[pid] = s
            td_counts.append(s["anytime_td"])
            td_owner.append(pid)
        td_counts.append(rec_tds[:, P] + rush_tds[:, P])
        td_owner.append(None)
        team_out[t] = {"points": pts[t], "tds": tds[t], "fgs": fgs[t], "plays": plays, "pass_rate": pass_rate,
                       "dropbacks": dropbacks, "sacks": sacks, "scrambles": scrambles, "pass_attempts": attempts,
                       "carries": team_carries, "targets": targets.sum(axis=1)}

    # First TD scorer: uniform over this trial's offensive TDs.
    counts = np.stack(td_counts, axis=1)
    total = counts.sum(axis=1)
    u = np.floor(rng.random(T) * np.maximum(total, 1)).astype(int)
    first_idx = (np.cumsum(counts, axis=1) > u[:, None]).argmax(axis=1)
    for j, pid in enumerate(td_owner):
        if pid is not None:
            stats[pid]["first_td"] = ((first_idx == j) & (total > 0)).astype(int)
    return SimResult(game_id=game_id, n_trials=T, players=players.reset_index(drop=True), stats=stats,
                     teams=team_out, inputs=dict(team_params))


def _rz_factor(rz: np.ndarray, share: np.ndarray) -> np.ndarray:
    """Relative red-zone involvement vs overall share (1 = proportional), clipped to [0.5, 2]."""
    if rz.sum() <= 0 or share.sum() <= 0:
        return np.ones_like(share)
    rz_share = rz / rz.sum()
    base = share / share.sum()
    return np.clip(np.where(base > 0, (0.5 * rz_share + 0.5 * base) / np.where(base > 0, base, 1), 1.0), 0.5, 2.0)


# ======================================================================================
# Public entry point
# ======================================================================================
_MODEL_CACHE: dict[tuple[str, str, str], GameModel] = {}


def _db_key(con: duckdb.DuckDBPyConnection) -> str:
    try:
        return str(con.execute("SELECT current_database()").fetchall()[0][0]) + str(id(con))
    except Exception:  # noqa: BLE001
        return str(id(con))


def game_inputs(con: duckdb.DuckDBPyConnection, game_id: str, model: GameModel | None = None,
                kind: str = "ridge", spread: float | None = None, total: float | None = None
                ) -> tuple[dict[str, TeamParams], GameModel]:
    """Team params for ``game_id``. Model is trained on games strictly before its date (cached).
    ``spread`` (home expected margin) / ``total`` override the schedule's market lines, e.g. with
    devigged live odds."""
    gd = build_game_dataset(con)
    rows = gd[gd["game_id"] == game_id].copy()
    if len(rows) != 2:
        raise ValueError(f"game {game_id!r} not found in team_features/schedules")
    gdate = pd.to_datetime(rows["game_date"].iloc[0])
    if spread is not None or total is not None:
        home = con.execute("SELECT home_team FROM schedules WHERE game_id = ?", [game_id]).fetchall()[0][0]
        if spread is not None:
            rows["team_spread"] = np.where(rows["team"] == home, spread, -spread)
        if total is not None:
            rows["total_line"] = total
        rows["implied_points"] = (rows["total_line"] + rows["team_spread"]) / 2
    if rows["implied_points"].isna().any():
        raise ValueError(f"No market spread/total for {game_id}; pass spread= and total=")
    if model is None:
        key = (_db_key(con), str(gdate.date()), kind)
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = GameModel(kind=kind).fit(gd[pd.to_datetime(gd["game_date"]) < gdate])
        model = _MODEL_CACHE[key]
    elif model.trained_through is not None and model.trained_through >= gdate:
        raise ValueError("Supplied GameModel was trained on games on/after this game's date (leakage)")
    pred = model.predict(rows)
    params = {}
    for _, r in pred.iterrows():
        params[r["team"]] = TeamParams(
            team=r["team"], pred_points=float(r["pred_points"]), pred_plays=float(r["pred_plays"]),
            pred_pass_rate=float(r["pred_pass_rate"]), sd_plays=model.resid["sd_plays"],
            sd_logit_pass_rate=model.resid["sd_logit_pass_rate"], beta_plays=model.resid["beta_plays"],
            beta_pass_rate=model.resid["beta_pass_rate"], sack_rate=model.league["sack_rate"],
            scramble_rate=model.league["scramble_rate"], target_rate=model.league["target_rate"],
            int_rate=model.league["int_rate"], pass_td_share=model.league["pass_td_share"],
            sd_points=model.resid.get("sd_points"))
    return params, model


def simulate_game(game_id: str, usage_overrides: Mapping[str, Mapping[str, Any]] | None = None, *,
                  n_trials: int = N_TRIALS, seed: int | None = None, con: duckdb.DuckDBPyConnection | None = None,
                  db_path: Path = DEFAULT_DB_PATH, model: GameModel | None = None, kind: str = "ridge",
                  spread: float | None = None, total: float | None = None,
                  allocation: str = "constrained", depth: Mapping[str, Any] | None = None,
                  weather: Mapping[str, Any] | None = None) -> SimResult:
    """Simulate ``game_id`` ``n_trials`` times and return full per-player stat arrays.

    ``usage_overrides`` example::

        {"00-0036900": {"target_share": 0.30},           # bump a WR (teammates rescaled)
         "00-0038120": {"active": False},                  # rule a player out (share redistributed)
         "DET": {"pass_rate": 0.65, "points": 27.5}}       # team-level overrides

    ``depth`` (dashboard): keyword arguments of :func:`depth_overrides` (``inactive``,
    ``reactivate``, ``starting_qbs``, ``qb_efficiency``, ``qb_attempt_share``). Its overrides are
    merged under ``usage_overrides`` and the depth-chart adjustments are enabled.

    ``weather``: dict from ``weather_client.get_game_weather`` (see :func:`weather_factors`),
    applied after the team-level overrides.
    """
    own = con is None
    con = con or duckdb.connect(str(db_path), read_only=True)
    try:
        params, _ = game_inputs(con, game_id, model=model, kind=kind, spread=spread, total=total)
        players = load_players(con, game_id)
    finally:
        if own:
            con.close()
    overrides: dict[str, dict[str, Any]] = {}
    if depth is not None:
        overrides = depth_overrides(players, **dict(depth))
    for key, ov in (usage_overrides or {}).items():
        overrides[key] = {**overrides.get(key, {}), **ov}
    players, team_over = apply_usage_overrides(players, overrides, list(params), depth_adjust=depth is not None)
    adjustments = {t: ov.pop("_adjustments") for t, ov in team_over.items() if "_adjustments" in ov}
    for team, ov in team_over.items():
        p = params[team]
        if "qb_attempt_share" in ov:
            p.qb_attempt_share = float(ov["qb_attempt_share"])
        if "points" in ov:
            p.pred_points = float(ov["points"])
        if "plays" in ov:
            p.pred_plays = float(ov["plays"])
        if "pass_rate" in ov:
            p.pred_pass_rate = float(ov["pass_rate"])
    players, wf = apply_weather(players, params, weather)
    if weather:
        for team in params:
            adjustments.setdefault(team, {})["weather"] = {**wf, "summary": weather.get("summary")}
    res = simulate(players, params, n_trials=n_trials, seed=seed, game_id=game_id, allocation=allocation)
    res.adjustments = adjustments
    return res


# ======================================================================================
# Evaluation / CLI
# ======================================================================================
def evaluate_game_model(con: duckdb.DuckDBPyConnection, season: int, kind: str = "ridge") -> dict[str, float]:
    """Walk-forward by week within ``season``: points MAE of model vs market-implied."""
    gd = build_game_dataset(con)
    errs_m, errs_k = [], []
    for week in sorted(gd.loc[gd["season"] == season, "week"].unique()):
        test = gd[(gd["season"] == season) & (gd["week"] == week)].dropna(subset=["points", "implied_points"])
        if test.empty:
            continue
        cutoff = pd.to_datetime(test["game_date"]).min()
        m = GameModel(kind=kind).fit(gd[pd.to_datetime(gd["game_date"]) < cutoff])
        pred = m.predict(test)
        errs_m.append(np.abs(test["points"] - pred["pred_points"]))
        errs_k.append(np.abs(test["points"] - test["implied_points"]))
    em, ek = pd.concat(errs_m), pd.concat(errs_k)
    return {"n": len(em), "model_points_mae": float(em.mean()), "market_points_mae": float(ek.mean())}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Simulate an NFL game or evaluate the game model.")
    p.add_argument("--game-id")
    p.add_argument("--evaluate-season", type=int)
    p.add_argument("--kind", choices=["ridge", "lgbm"], default="ridge")
    p.add_argument("--trials", type=int, default=N_TRIALS)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with duckdb.connect(str(args.db_path), read_only=True) as con:
        if args.evaluate_season:
            print(evaluate_game_model(con, args.evaluate_season, args.kind))
        if args.game_id:
            res = simulate_game(args.game_id, n_trials=args.trials, seed=args.seed, con=con, kind=args.kind)
            for t, d in res.teams.items():
                print(f"{t}: points {d['points'].mean():.1f}  plays {d['plays'].mean():.1f}  "
                      f"pass rate {d['pass_rate'].mean():.3f}")
            s = res.summary()
            keep = s["stat"].isin(["pass_yds", "rush_yds", "reception_yds", "receptions", "anytime_td"])
            with pd.option_context("display.width", 200, "display.max_rows", 200):
                print(s[keep & (s["mean"] > 0.05)].sort_values(["team", "stat", "mean"], ascending=[True, True, False])
                      .to_string(index=False, float_format="%.2f"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
