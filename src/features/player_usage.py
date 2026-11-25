"""Player usage & efficiency features (point-in-time) -> DuckDB ``player_features``.

Grain: one row per (player, team-game) for every scheduled game, including upcoming ones. Rows
come from two sources:

* **eligible** players: their most recent *prior* game was for this team, within the team's last
  ``ELIGIBLE_LOOKBACK_GAMES`` games. This pool depends only on history, so it exists for
  upcoming games and drives the injury spillover.
* players who actually appeared in a past game but were not eligible (debuts, returns from long
  absences, trades). They get features but take no part in redistribution.

Point-in-time guarantee
-----------------------
Per-game stats are first computed per player-game. Exponentially weighted averages (span 3 and
5) are then taken over each player's own game sequence, and each row picks up the EWMA state
from the player's latest game **strictly before** the row's ``game_date`` (``merge_asof`` with
``allow_exact_matches=False``). Same-game stats never reach that game's features. The only
inputs about the current game are the schedule and that week's injury report, both published
before kickoff.

Cold-start usage prior
----------------------
Players with little meaningful history (fewer than ``COLD_START_GAMES`` games above
``QUAL_SNAP_SHARE`` snaps) have EWMAs dominated by spot duty, which badly understates a promoted
rookie. Each row therefore also carries a role-based prior: expected snap share (latest game
weighted over the EWMA) x the league's target/carry share per snap for that position and snap
tier (rotational / starter / full-time). The league rates are cumulative over games on strictly
earlier dates and shrunk toward fixed defaults, so the prior is point-in-time too. The simulator
blends own usage with this prior (``game_sim.load_players``).

RYOE uses an expected-yards baseline per (down, distance, field-position) bucket built only from
carries on earlier dates, shrunk toward the prior league mean.

Route participation = routes / team dropbacks, where a route is a WR/TE/RB/FB on the field
(``participation.offense_players``) for a dropback. When participation data is missing for a game
(nflverse publishes it with a lag), snap share stands in and ``routes_from_participation = 0``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "nfl_model.duckdb"

SPANS: tuple[int, ...] = (3, 5)
ELIGIBLE_LOOKBACK_GAMES = 6
OUT_STATUSES = ("Out",)
TOP_N = 2
SAME_GROUP_WEIGHT = 2.0        # vacated targets flow preferentially to the same position group
PARTICIPATION_MIN_COVERAGE = 0.8
XYARDS_SHRINK_K = 50.0
XYARDS_DEFAULT = 4.3
QUAL_SNAP_SHARE = 0.20         # a game counts toward established usage above this snap share
COLD_START_GAMES = 3           # fewer qualifying games than this -> blend with the role prior
RECENT_SNAP_WEIGHT = 0.6       # expected snap share = w * latest game + (1 - w) * EWMA(3)
SNAP_TIER_EDGES = (0.5, 0.8)   # tiers: rotational < 0.5 <= starter < 0.8 <= full-time
USAGE_PRIOR_K = 50.0           # pseudo snap-units of the default rate behind the PIT league rate
# (target share per snap share, carry share per snap share) by position and tier: long-run league
# values, used as the shrinkage target and before any history exists
USAGE_PRIOR_DEFAULTS = {
    "WR": ((0.20, 0.015), (0.235, 0.012), (0.25, 0.009)),
    "TE": ((0.125, 0.004), (0.18, 0.002), (0.20, 0.002)),
    "RB": ((0.17, 0.70), (0.175, 0.83), (0.175, 0.84)),
    "FB": ((0.10, 0.20), (0.10, 0.20), (0.10, 0.20)),
    "QB": ((0.01, 0.20), (0.002, 0.13), (0.001, 0.12)),
}
SKILL_POSITIONS = ("QB", "RB", "FB", "WR", "TE")
POSITION_GROUP = {"QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}

# Per-game metrics averaged directly with EWMA
SHARE_METRICS = ("snap_share", "route_participation", "target_share", "air_yards_share", "carry_share",
                 "rz_targets", "rz_carries", "rz_touches", "routes_from_participation")
# EWMA ratio metrics: name -> (numerator, denominator)
RATIO_METRICS = {"yprr": ("rec_yards", "routes"), "ryoe_per_carry": ("ryoe", "designed_carries")}


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                            [table]).fetchall()[0][0])


# --------------------------------------------------------------------------------------
# Expected rushing yards (point-in-time)
# --------------------------------------------------------------------------------------
def _rush_bucket_sql() -> str:
    return """
        CAST(down AS INT) || '|' ||
        CASE WHEN ydstogo <= 2 THEN 'a' WHEN ydstogo <= 5 THEN 'b' WHEN ydstogo <= 10 THEN 'c' ELSE 'd' END || '|' ||
        CASE WHEN yardline_100 <= 5 THEN 'a' WHEN yardline_100 <= 10 THEN 'b' WHEN yardline_100 <= 20 THEN 'c'
             WHEN yardline_100 <= 50 THEN 'd' WHEN yardline_100 <= 80 THEN 'e' ELSE 'f' END"""


def expected_rush_yards_sql() -> str:
    """Per designed carry: x_yards from carries on strictly earlier dates only."""
    return f"""
    WITH carries AS (
        SELECT p.game_id, p.play_id, s.game_date, p.yards_gained, {_rush_bucket_sql()} AS bucket
        FROM pbp p JOIN schedules s USING (game_id)
        WHERE p.season_type IN ('REG', 'POST') AND p.play_type = 'run' AND p.rush_attempt = 1
          AND COALESCE(p.qb_scramble, 0) = 0 AND COALESCE(p.two_point_attempt, 0) = 0
          AND p.rusher_player_id IS NOT NULL AND p.down IS NOT NULL AND s.game_date IS NOT NULL
    ),
    daily AS (SELECT bucket, game_date, SUM(yards_gained) AS s, COUNT(*) AS n FROM carries GROUP BY ALL),
    bucket_prior AS (
        SELECT bucket, game_date,
               SUM(s) OVER w AS prior_s, SUM(n) OVER w AS prior_n
        FROM daily WINDOW w AS (PARTITION BY bucket ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    ),
    gdaily AS (SELECT game_date, SUM(s) AS s, SUM(n) AS n FROM daily GROUP BY 1),
    global_prior AS (
        SELECT game_date,
               SUM(s) OVER w / NULLIF(SUM(n) OVER w, 0) AS prior_mean
        FROM gdaily WINDOW w AS (ORDER BY game_date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
    )
    SELECT c.game_id, c.play_id, c.yards_gained,
           (COALESCE(b.prior_s, 0) + {XYARDS_SHRINK_K} * COALESCE(g.prior_mean, {XYARDS_DEFAULT}))
             / (COALESCE(b.prior_n, 0) + {XYARDS_SHRINK_K}) AS x_yards
    FROM carries c
    JOIN bucket_prior b USING (bucket, game_date)
    JOIN global_prior g USING (game_date)
    """


# --------------------------------------------------------------------------------------
# Per player-game stats
# --------------------------------------------------------------------------------------
def player_game_stats_sql(has_participation: bool) -> str:
    part_ctes = """
    part AS (
        SELECT game_id, CAST(play_id AS DOUBLE) AS play_id, offense_players
        FROM participation WHERE offense_players IS NOT NULL AND offense_players <> ''
    ),
    part_cov AS (
        SELECT b.game_id, b.posteam AS team, COUNT(*) AS part_dropbacks
        FROM base b JOIN part p USING (game_id, play_id) WHERE b.qb_dropback = 1 GROUP BY ALL
    ),
    routes AS (
        SELECT game_id, team, gsis_id, COUNT(*) AS routes_part FROM (
            SELECT b.game_id, b.posteam AS team, UNNEST(string_split(p.offense_players, ';')) AS gsis_id
            FROM base b JOIN part p USING (game_id, play_id) WHERE b.qb_dropback = 1)
        GROUP BY ALL
    ),""" if has_participation else """
    part_cov AS (SELECT NULL::VARCHAR AS game_id, NULL::VARCHAR AS team, NULL::BIGINT AS part_dropbacks WHERE FALSE),
    routes AS (SELECT NULL::VARCHAR AS game_id, NULL::VARCHAR AS team, NULL::VARCHAR AS gsis_id,
                      NULL::BIGINT AS routes_part WHERE FALSE),"""
    return f"""
    WITH base AS (
        SELECT * FROM pbp
        WHERE season_type IN ('REG', 'POST') AND posteam IS NOT NULL AND play_type IN ('pass', 'run')
          AND COALESCE(two_point_attempt, 0) = 0
    ),
    xy AS ({expected_rush_yards_sql()}),
    {part_ctes}
    team_game AS (
        SELECT game_id, posteam AS team,
               COUNT(*) FILTER (WHERE qb_dropback = 1) AS team_dropbacks,
               COUNT(*) FILTER (WHERE pass_attempt = 1 AND COALESCE(sack, 0) = 0 AND receiver_player_id IS NOT NULL) AS team_targets,
               SUM(air_yards) FILTER (WHERE pass_attempt = 1 AND COALESCE(sack, 0) = 0 AND receiver_player_id IS NOT NULL) AS team_air_yards,
               COUNT(*) FILTER (WHERE rush_attempt = 1 AND rusher_player_id IS NOT NULL) AS team_carries
        FROM base GROUP BY ALL
    ),
    rec AS (
        SELECT game_id, posteam AS team, receiver_player_id AS gsis_id,
               COUNT(*) AS targets, SUM(COALESCE(air_yards, 0)) AS air_yards,
               SUM(COALESCE(complete_pass, 0)) AS receptions,
               SUM(CASE WHEN complete_pass = 1 THEN COALESCE(yards_gained, 0) ELSE 0 END) AS rec_yards,
               COUNT(*) FILTER (WHERE yardline_100 <= 20) AS rz_targets,
               SUM(COALESCE(complete_pass, 0)) FILTER (WHERE yardline_100 <= 20) AS rz_receptions
        FROM base WHERE pass_attempt = 1 AND COALESCE(sack, 0) = 0 AND receiver_player_id IS NOT NULL
        GROUP BY ALL
    ),
    rush AS (
        SELECT b.game_id, b.posteam AS team, b.rusher_player_id AS gsis_id,
               COUNT(*) AS carries,
               COUNT(*) FILTER (WHERE b.yardline_100 <= 20) AS rz_carries,
               COUNT(xy.x_yards) AS designed_carries,
               SUM(xy.yards_gained - xy.x_yards) AS ryoe
        FROM base b LEFT JOIN xy USING (game_id, play_id)
        WHERE b.rush_attempt = 1 AND b.rusher_player_id IS NOT NULL
        GROUP BY ALL
    ),
    snaps AS (
        SELECT game_id, team, gsis_id, MAX(offense_snaps) AS offense_snaps, MAX(offense_pct) AS offense_pct
        FROM snap_counts WHERE gsis_id IS NOT NULL GROUP BY ALL
    ),
    keys AS (
        SELECT game_id, team, gsis_id FROM rec
        UNION SELECT game_id, team, gsis_id FROM rush
        UNION SELECT game_id, team, gsis_id FROM snaps WHERE offense_snaps > 0
    ),
    pos AS (
        SELECT gsis_id, season, ANY_VALUE(position) AS position FROM rosters
        WHERE gsis_id IS NOT NULL GROUP BY ALL
    ),
    dir AS (SELECT gsis_id, ANY_VALUE(position) AS position, ANY_VALUE(display_name) AS player_name
            FROM players GROUP BY 1)
    SELECT k.gsis_id, k.team, k.game_id, s.season, s.week, s.game_date,
           COALESCE(pos.position, dir.position) AS position, dir.player_name,
           tg.team_dropbacks, tg.team_targets, tg.team_air_yards, tg.team_carries,
           COALESCE(rec.targets, 0) AS targets, COALESCE(rec.air_yards, 0) AS air_yards,
           COALESCE(rec.receptions, 0) AS receptions, COALESCE(rec.rec_yards, 0) AS rec_yards,
           COALESCE(rec.rz_targets, 0) AS rz_targets, COALESCE(rec.rz_receptions, 0) AS rz_receptions,
           COALESCE(rush.carries, 0) AS carries, COALESCE(rush.rz_carries, 0) AS rz_carries,
           COALESCE(rush.designed_carries, 0) AS designed_carries, COALESCE(rush.ryoe, 0) AS ryoe,
           sn.offense_snaps, sn.offense_pct,
           pc.part_dropbacks, r.routes_part
    FROM keys k
    JOIN schedules s ON s.game_id = k.game_id
    JOIN team_game tg ON tg.game_id = k.game_id AND tg.team = k.team
    LEFT JOIN rec ON rec.game_id = k.game_id AND rec.team = k.team AND rec.gsis_id = k.gsis_id
    LEFT JOIN rush ON rush.game_id = k.game_id AND rush.team = k.team AND rush.gsis_id = k.gsis_id
    LEFT JOIN snaps sn ON sn.game_id = k.game_id AND sn.team = k.team AND sn.gsis_id = k.gsis_id
    LEFT JOIN part_cov pc ON pc.game_id = k.game_id AND pc.team = k.team
    LEFT JOIN routes r ON r.game_id = k.game_id AND r.team = k.team AND r.gsis_id = k.gsis_id
    LEFT JOIN pos ON pos.gsis_id = k.gsis_id AND pos.season = s.season
    LEFT JOIN dir ON dir.gsis_id = k.gsis_id
    """


def compute_player_game_stats(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(player_game_stats_sql(_table_exists(con, "participation"))).df()
    df = df[df["position"].isin(SKILL_POSITIONS)].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["position_group"] = df["position"].map(POSITION_GROUP)

    def share(num: str, den: str) -> pd.Series:
        d = df[den].astype(float)
        return (df[num].astype(float) / d.where(d > 0)).astype(float)

    covered = (df["part_dropbacks"].fillna(0) >= PARTICIPATION_MIN_COVERAGE * df["team_dropbacks"]) & \
              (df["team_dropbacks"] > 0)
    is_route_runner = df["position"].isin(["RB", "FB", "WR", "TE"])
    part_rp = (df["routes_part"].fillna(0) / df["part_dropbacks"].where(df["part_dropbacks"] > 0)).where(is_route_runner, 0.0)
    df["routes_from_participation"] = covered.astype(float)
    df["snap_share"] = df["offense_pct"].astype(float)
    df["route_participation"] = np.where(covered, part_rp, df["snap_share"].where(is_route_runner, 0.0))
    df["routes"] = df["route_participation"] * df["team_dropbacks"].astype(float)
    df["target_share"] = share("targets", "team_targets")
    df["air_yards_share"] = share("air_yards", "team_air_yards")
    df["carry_share"] = share("carries", "team_carries")
    df["rz_touches"] = df["rz_carries"] + df["rz_receptions"]
    return df.sort_values(["gsis_id", "game_date", "game_id"]).reset_index(drop=True)


# --------------------------------------------------------------------------------------
# EWMA state after each game
# --------------------------------------------------------------------------------------
def ewma_state(stats: pd.DataFrame, spans: Sequence[int] = SPANS) -> pd.DataFrame:
    """Post-game EWMA per player *including* that game. Callers must only attach it to strictly
    later games (see ``attach_state``)."""
    stats = stats.sort_values(["gsis_id", "game_date", "game_id"]).reset_index(drop=True)
    g = stats.groupby("gsis_id", sort=False)
    out = stats[["gsis_id", "game_date", "team"]].rename(columns={"team": "last_game_team",
                                                                  "game_date": "state_date"}).copy()
    out["prior_games"] = g.cumcount() + 1
    qual = (stats["snap_share"].fillna(0.0) > QUAL_SNAP_SHARE).astype(int)
    out["qual_games"] = qual.groupby(stats["gsis_id"], sort=False).cumsum().to_numpy()
    out["last_snap_share"] = stats["snap_share"].astype(float).to_numpy()
    ratio_cols = sorted({c for pair in RATIO_METRICS.values() for c in pair})
    cols = list(SHARE_METRICS) + ratio_cols
    values = stats[cols].astype(float)
    values["gsis_id"] = stats["gsis_id"]
    for span in spans:
        ew = (values.groupby("gsis_id", sort=False)[cols].ewm(span=span, min_periods=1).mean()
              .reset_index(level=0, drop=True).sort_index())
        for m in SHARE_METRICS:
            out[f"{m}_ewm{span}"] = ew[m].to_numpy()
        for name, (num, den) in RATIO_METRICS.items():
            d = ew[den].to_numpy()
            out[f"{name}_ewm{span}"] = np.where(d > 0, ew[num].to_numpy() / np.where(d > 0, d, 1), np.nan)
    return out


def attach_state(panel: pd.DataFrame, state: pd.DataFrame) -> pd.DataFrame:
    """As-of join: each panel row gets the state from the player's latest game strictly before it."""
    panel = panel.assign(game_date=pd.to_datetime(panel["game_date"]).astype("datetime64[ns]"))
    state = state.assign(state_date=pd.to_datetime(state["state_date"]).astype("datetime64[ns]"))
    panel = panel.sort_values("game_date").reset_index(drop=True)
    state = state.sort_values("state_date").reset_index(drop=True)
    merged = pd.merge_asof(panel, state, left_on="game_date", right_on="state_date", by="gsis_id",
                           allow_exact_matches=False, direction="backward")
    merged["prior_games"] = merged["prior_games"].fillna(0).astype(int)
    if "qual_games" in merged:
        merged["qual_games"] = merged["qual_games"].fillna(0).astype(int)
    merged["days_since_last_game"] = (merged["game_date"] - merged["state_date"]).dt.days
    return merged.drop(columns=["state_date"])


# --------------------------------------------------------------------------------------
# Cold-start role prior
# --------------------------------------------------------------------------------------
def snap_tier(snap) -> np.ndarray:
    return np.digitize(np.nan_to_num(np.asarray(snap, float)), SNAP_TIER_EDGES)


def _default_rate(position: str, tier: int, j: int) -> float:
    return USAGE_PRIOR_DEFAULTS.get(position, USAGE_PRIOR_DEFAULTS["WR"])[int(tier)][j]


def positional_usage_rates(stats: pd.DataFrame, k: float = USAGE_PRIOR_K) -> pd.DataFrame:
    """League target/carry share per snap share by (position, tier), cumulative through each date.

    Built from qualifying player-games only and shrunk toward ``USAGE_PRIOR_DEFAULTS`` with ``k``
    pseudo snap-units. Each row is the state *after* ``rate_date``: attach it with
    ``allow_exact_matches=False``.
    """
    d = stats[(stats["snap_share"] > QUAL_SNAP_SHARE) & stats["position"].isin(list(USAGE_PRIOR_DEFAULTS))].copy()
    d["usage_tier"] = snap_tier(d["snap_share"])
    agg = (d.groupby(["position", "usage_tier", "game_date"])
           .agg(snap=("snap_share", "sum"), ts=("target_share", "sum"), cs=("carry_share", "sum"))
           .reset_index().sort_values("game_date").reset_index(drop=True))
    cum = agg.groupby(["position", "usage_tier"], sort=False)[["snap", "ts", "cs"]].cumsum()
    d_t = np.array([_default_rate(pos, t, 0) for pos, t in zip(agg["position"], agg["usage_tier"])])
    d_c = np.array([_default_rate(pos, t, 1) for pos, t in zip(agg["position"], agg["usage_tier"])])
    agg["pos_target_per_snap"] = (cum["ts"].fillna(0) + k * d_t) / (cum["snap"] + k)
    agg["pos_carry_per_snap"] = (cum["cs"].fillna(0) + k * d_c) / (cum["snap"] + k)
    return agg[["position", "usage_tier", "game_date", "pos_target_per_snap", "pos_carry_per_snap"]].rename(
        columns={"game_date": "rate_date"})


def attach_usage_prior(panel: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Expected snap share, snap tier and the role-based target/carry share prior for every row.

    Uses only the as-of state already attached to the row (latest prior game, EWMAs) and league
    rates from strictly earlier dates.
    """
    p = panel.copy()
    last = p["last_snap_share"].astype(float)
    ewm3 = p["snap_share_ewm3"].astype(float).fillna(last)
    p["snap_share_expected"] = (RECENT_SNAP_WEIGHT * last + (1 - RECENT_SNAP_WEIGHT) * ewm3).fillna(0.0)
    p["usage_tier"] = snap_tier(p["snap_share_expected"])
    rates = positional_usage_rates(stats)
    rates["rate_date"] = pd.to_datetime(rates["rate_date"]).astype("datetime64[ns]")
    p["game_date"] = pd.to_datetime(p["game_date"]).astype("datetime64[ns]")
    p["_row"] = np.arange(len(p))
    known = p.loc[p["position"].isin(list(USAGE_PRIOR_DEFAULTS)), ["_row", "game_date", "position", "usage_tier"]]
    m = pd.merge_asof(known.sort_values("game_date"), rates.sort_values("rate_date"), left_on="game_date",
                      right_on="rate_date", by=["position", "usage_tier"], allow_exact_matches=False,
                      direction="backward").set_index("_row")
    for j, c in ((0, "pos_target_per_snap"), (1, "pos_carry_per_snap")):
        fallback = pd.Series([_default_rate(pos, t, j) for pos, t in zip(p["position"], p["usage_tier"])],
                             index=p.index)
        p[c] = p["_row"].map(m[c]).astype(float).fillna(fallback)
    p["target_share_prior"] = p["pos_target_per_snap"] * p["snap_share_expected"]
    p["carry_share_prior"] = p["pos_carry_per_snap"] * p["snap_share_expected"]
    p["cold_start"] = p["qual_games"] < COLD_START_GAMES
    return p.drop(columns=["_row"])


# --------------------------------------------------------------------------------------
# Panel of (player, team-game)
# --------------------------------------------------------------------------------------
def team_schedule(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute("""
        SELECT game_id, season, week, game_date, home_team AS team, away_team AS opponent FROM schedules
        WHERE game_date IS NOT NULL
        UNION ALL
        SELECT game_id, season, week, game_date, away_team, home_team FROM schedules WHERE game_date IS NOT NULL
    """).df()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)
    df["team_game_idx"] = df.groupby("team").cumcount()
    return df


def build_panel(stats: pd.DataFrame, sched: pd.DataFrame, lookback: int = ELIGIBLE_LOOKBACK_GAMES) -> pd.DataFrame:
    """Eligible rows (history-only pool) plus rows for players who appeared but weren't eligible."""
    pg = stats[["gsis_id", "team", "game_id", "game_date"]].merge(
        sched[["game_id", "team", "team_game_idx"]], on=["game_id", "team"], how="inner")
    pg = pg.sort_values(["gsis_id", "game_date"]).reset_index(drop=True)
    pg["next_game_date"] = pg.groupby("gsis_id")["game_date"].shift(-1)

    # Each player-game makes the player eligible for that team's next `lookback` games, up to
    # (and including) the date of his next appearance anywhere, which supersedes it.
    targets = sched[["team", "team_game_idx", "game_id", "game_date"]].rename(
        columns={"team_game_idx": "target_idx", "game_id": "target_game_id", "game_date": "target_date"})
    ahead = pd.concat([pg.assign(target_idx=pg["team_game_idx"] + k) for k in range(1, lookback + 1)],
                      ignore_index=True).merge(targets, on=["team", "target_idx"], how="inner")
    ahead = ahead[ahead["next_game_date"].isna() | (ahead["target_date"] <= ahead["next_game_date"])]
    eligible = ahead[["gsis_id", "team", "target_game_id"]].rename(columns={"target_game_id": "game_id"})
    eligible = eligible.drop_duplicates(["gsis_id", "game_id"]).assign(eligible=True)

    appeared = stats[["gsis_id", "team", "game_id"]].assign(eligible=False)
    panel = pd.concat([eligible, appeared], ignore_index=True)
    panel = panel.sort_values("eligible", ascending=False).drop_duplicates(["gsis_id", "game_id"], keep="first")
    return panel.merge(sched[["game_id", "team", "season", "week", "game_date", "opponent"]],
                       on=["game_id", "team"], how="inner")


def attach_injuries(con: duckdb.DuckDBPyConnection, panel: pd.DataFrame) -> pd.DataFrame:
    if not _table_exists(con, "injuries"):
        return panel.assign(injury_status=None, is_out=False)
    inj = con.execute("""
        SELECT CAST(season AS INT) AS season, CAST(week AS INT) AS week, team, gsis_id,
               ANY_VALUE(report_status) AS injury_status
        FROM injuries WHERE gsis_id IS NOT NULL GROUP BY ALL""").df()
    panel = panel.merge(inj, on=["season", "week", "team", "gsis_id"], how="left")
    panel["is_out"] = panel["injury_status"].isin(OUT_STATUSES)
    return panel


def player_meta(stats: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Latest known position/name per player (positions rarely change; used for upcoming rows)."""
    latest = (stats.sort_values("game_date").groupby("gsis_id")[["position", "player_name"]].last())
    return latest.reset_index()


# --------------------------------------------------------------------------------------
# Injury spillover
# --------------------------------------------------------------------------------------
def apply_injury_spillover(panel: pd.DataFrame, spans: Sequence[int] = SPANS, top_n: int = TOP_N,
                           same_group_weight: float = SAME_GROUP_WEIGHT) -> pd.DataFrame:
    """For each team-game, if any of the team's top-``top_n`` RB/WR/TE (by pre-game EWMA usage)
    is Out, give their target and carry share to the remaining eligible players.

    * targets go to RB/WR/TE/FB in proportion to their own target share, with same-group players
      weighted ``same_group_weight``x;
    * carries go to RB/FB in proportion to carry share.
    Only eligible rows (a history-based pool) take part, so totals per team-game are preserved.
    """
    panel = panel.copy()
    panel["is_top2"] = False
    for span in spans:
        ts, cs = f"target_share_ewm{span}", f"carry_share_ewm{span}"
        panel[f"{ts}_adj"] = panel[ts]
        panel[f"{cs}_adj"] = panel[cs]
        panel[f"vacated_target_share_ewm{span}"] = 0.0
        panel[f"vacated_carry_share_ewm{span}"] = 0.0
    panel["team_top2_out"] = 0

    pool_mask = panel["eligible"] & panel["position_group"].isin(["RB", "WR", "TE"])
    team_index = panel.groupby(["game_id", "team"]).groups
    for (game_id, team), idx in panel[pool_mask].groupby(["game_id", "team"]).groups.items():
        grp = panel.loc[idx]
        for span in spans:
            ts, cs = f"target_share_ewm{span}", f"carry_share_ewm{span}"
            t = grp[ts].fillna(0.0)
            c = grp[cs].fillna(0.0)
            usage = pd.Series(np.where(grp["position_group"] == "RB", t + c, t), index=grp.index)
            # Deterministic tie-break on gsis_id: row order must never influence the result.
            order = (pd.DataFrame({"g": grp["position_group"], "u": usage, "id": grp["gsis_id"]})
                     .sort_values(["g", "u", "id"], ascending=[True, False, True]))
            rank = order.groupby("g").cumcount().add(1).reindex(grp.index)
            top = (rank <= top_n) & (usage > 0)
            if span == max(spans):
                panel.loc[grp.index, "is_top2"] = top.to_numpy()
            out_top = grp.index[top & grp["is_out"]]
            if len(out_top) == 0:
                continue
            if span == max(spans):
                panel.loc[team_index[(game_id, team)], "team_top2_out"] = len(out_top)
            healthy = grp.index[~grp["is_out"]]
            add_t = pd.Series(0.0, index=healthy)
            add_c = pd.Series(0.0, index=healthy)
            for pid in out_top:
                g_out = grp.at[pid, "position_group"]
                w = t[healthy] * np.where(grp.loc[healthy, "position_group"] == g_out, same_group_weight, 1.0)
                if w.sum() > 0:
                    add_t += t[pid] * w / w.sum()
                rb = grp.loc[healthy, "position_group"] == "RB"
                wc = c[healthy].where(rb, 0.0)
                if wc.sum() > 0:
                    add_c += c[pid] * wc / wc.sum()
            panel.loc[healthy, f"{ts}_adj"] = t[healthy] + add_t
            panel.loc[healthy, f"{cs}_adj"] = c[healthy] + add_c
            panel.loc[out_top, f"{ts}_adj"] = 0.0
            panel.loc[out_top, f"{cs}_adj"] = 0.0
            team_rows = team_index[(game_id, team)]  # incl. QBs / non-pool rows
            panel.loc[team_rows, f"vacated_target_share_ewm{span}"] = float(t[out_top].sum())
            panel.loc[team_rows, f"vacated_carry_share_ewm{span}"] = float(c[out_top].sum())
    return panel


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def compute_player_features(con: duckdb.DuckDBPyConnection, spans: Sequence[int] = SPANS) -> pd.DataFrame:
    stats = compute_player_game_stats(con)
    sched = team_schedule(con)
    state = ewma_state(stats, spans)
    panel = build_panel(stats, sched)
    panel = panel.merge(player_meta(stats, con), on="gsis_id", how="left")
    panel["position_group"] = panel["position"].map(POSITION_GROUP)
    panel = attach_state(panel, state)
    panel = attach_usage_prior(panel, stats)
    panel = attach_injuries(con, panel)
    panel = apply_injury_spillover(panel, spans)
    lead = ["season", "week", "game_id", "game_date", "team", "opponent", "gsis_id", "player_name", "position",
            "position_group", "eligible", "prior_games", "qual_games", "cold_start", "days_since_last_game",
            "last_game_team",
            "injury_status", "is_out", "is_top2", "team_top2_out"]
    rest = sorted(c for c in panel.columns if c not in lead)
    return panel[lead + rest].sort_values(["game_date", "game_id", "team", "gsis_id"]).reset_index(drop=True)


def build_player_features(con: duckdb.DuckDBPyConnection, spans: Sequence[int] = SPANS,
                          table: str = "player_features") -> int:
    df = compute_player_features(con, spans)
    df["game_date"] = df["game_date"].dt.date
    con.register("_player_features_df", df)
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM _player_features_df")
    con.unregister("_player_features_df")
    logger.info("Wrote %d rows to %s", len(df), table)
    return len(df)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build point-in-time player usage features.")
    p.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with duckdb.connect(str(args.db_path)) as con:
        build_player_features(con)
    return 0


if __name__ == "__main__":
    sys.exit(main())
