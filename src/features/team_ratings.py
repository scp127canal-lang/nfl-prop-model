"""Team-level rolling features (point-in-time) -> DuckDB ``team_features``.

One row per (team, scheduled game), including games not yet played, so the latest row for each
team is the feature vector for its next game.

Point-in-time guarantee
-----------------------
Every rolling metric for a game is computed over that team's last ``w`` *played* games with
``game_date`` strictly before the game (rolling over played games, then an ``ASOF`` join on
``game_date >``); the current game and anything after it never enter. Rest days / home-away come from the schedule, which is known
before kickoff.

Play filter
-----------
Regular + post season scrimmage plays (``play_type`` in pass/run), no two-point tries, EPA present,
and garbage time removed: 4th-quarter plays with offense win probability < 0.05 or > 0.95.
Dropbacks are ``qb_dropback = 1`` (incl. sacks & scrambles); rushes are designed runs (``rush = 1``).

Pace = mean seconds of game clock between consecutive snaps by the same offense within a drive
(non-garbage snaps, gaps outside 1..60s dropped).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "nfl_model.duckdb"

GARBAGE_WP_LOW = 0.05
GARBAGE_WP_HIGH = 0.95
WINDOWS: tuple[int, ...] = (5, 10)

# metric name -> (numerator column, denominator column) in the team-game stats table
_RATE_METRICS: dict[str, tuple[str, str]] = {
    "off_epa_per_dropback": ("off_db_epa", "off_db_n"),
    "off_epa_per_rush": ("off_rush_epa", "off_rush_n"),
    "off_success_rate": ("off_success", "off_plays"),
    "def_epa_per_dropback": ("def_db_epa", "def_db_n"),
    "def_epa_per_rush": ("def_rush_epa", "def_rush_n"),
    "def_success_rate": ("def_success", "def_plays"),
    "pace_sec_per_play": ("pace_secs", "pace_n"),
    "off_plays_per_game": ("off_plays", "played"),
}


def _not_garbage(alias: str = "") -> str:
    a = f"{alias}." if alias else ""
    return (f"NOT ({a}qtr = 4 AND COALESCE({a}wp < {GARBAGE_WP_LOW} OR {a}wp > {GARBAGE_WP_HIGH}, FALSE))")


def team_game_stats_sql() -> str:
    """Per (game_id, team) offensive/defensive sums for played games (garbage time removed)."""
    return f"""
    WITH plays AS (
        SELECT game_id, play_id, posteam, defteam, epa, success, qb_dropback, rush
        FROM pbp
        WHERE season_type IN ('REG', 'POST') AND posteam IS NOT NULL AND defteam IS NOT NULL
          AND play_type IN ('pass', 'run') AND COALESCE(two_point_attempt, 0) = 0
          AND epa IS NOT NULL AND {_not_garbage()}
    ),
    snaps AS (
        SELECT game_id, posteam, qtr, wp, game_seconds_remaining AS gsr,
               LEAD(game_seconds_remaining) OVER (
                   PARTITION BY game_id, posteam, fixed_drive ORDER BY play_id) AS next_gsr
        FROM pbp
        WHERE season_type IN ('REG', 'POST') AND posteam IS NOT NULL
          AND play_type IN ('pass', 'run', 'no_play')
    ),
    pace AS (
        SELECT game_id, posteam AS team, SUM(gsr - next_gsr) AS pace_secs, COUNT(*) AS pace_n
        FROM snaps
        WHERE next_gsr IS NOT NULL AND gsr - next_gsr BETWEEN 1 AND 60 AND {_not_garbage()}
        GROUP BY ALL
    ),
    off AS (
        SELECT game_id, posteam AS team,
               SUM(epa) FILTER (WHERE qb_dropback = 1) AS off_db_epa,
               COUNT(*) FILTER (WHERE qb_dropback = 1) AS off_db_n,
               SUM(epa) FILTER (WHERE rush = 1 AND COALESCE(qb_dropback, 0) = 0) AS off_rush_epa,
               COUNT(*) FILTER (WHERE rush = 1 AND COALESCE(qb_dropback, 0) = 0) AS off_rush_n,
               SUM(success) AS off_success, COUNT(*) AS off_plays
        FROM plays GROUP BY ALL
    ),
    def AS (
        SELECT game_id, defteam AS team,
               SUM(epa) FILTER (WHERE qb_dropback = 1) AS def_db_epa,
               COUNT(*) FILTER (WHERE qb_dropback = 1) AS def_db_n,
               SUM(epa) FILTER (WHERE rush = 1 AND COALESCE(qb_dropback, 0) = 0) AS def_rush_epa,
               COUNT(*) FILTER (WHERE rush = 1 AND COALESCE(qb_dropback, 0) = 0) AS def_rush_n,
               SUM(success) AS def_success, COUNT(*) AS def_plays
        FROM plays GROUP BY ALL
    )
    SELECT off.game_id, off.team, 1 AS played,
           off.* EXCLUDE (game_id, team), def.* EXCLUDE (game_id, team), pace.* EXCLUDE (game_id, team)
    FROM off
    LEFT JOIN def USING (game_id, team)
    LEFT JOIN pace USING (game_id, team)
    """


def schedule_long_sql() -> str:
    """One row per (team, scheduled game) with home/away, neutral and rest days."""
    return """
    WITH g AS (
        SELECT game_id, season, week, game_type, game_date, home_team, away_team,
               COALESCE(location = 'Neutral', FALSE) AS is_neutral
        FROM schedules WHERE game_date IS NOT NULL
    ),
    long AS (
        SELECT game_id, season, week, game_type, game_date, home_team AS team, away_team AS opponent,
               CASE WHEN is_neutral THEN 0 ELSE 1 END AS is_home, is_neutral::INT AS is_neutral FROM g
        UNION ALL
        SELECT game_id, season, week, game_type, game_date, away_team, home_team, 0, is_neutral::INT FROM g
    )
    SELECT *,
           date_diff('day', LAG(game_date) OVER (PARTITION BY team ORDER BY game_date, game_id), game_date)
               AS rest_days
    FROM long
    """


def team_features_sql(windows: Sequence[int] = WINDOWS) -> str:
    """Rolling windows run over *played* games only (inclusive of each played game), then every
    scheduled game takes the window ending at the team's latest played game strictly before it
    (``ASOF`` join on ``game_date >``). A game can therefore never see its own stats or later ones,
    and games several weeks out still get the last ``w`` played games."""
    roll_cols = []
    for w in windows:
        frame = f"(PARTITION BY team ORDER BY game_date, game_id ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW)"
        for name, (num, den) in _RATE_METRICS.items():
            roll_cols.append(f"SUM({num}) OVER {frame} / NULLIF(SUM({den}) OVER {frame}, 0) AS {name}_{w}")
        roll_cols.append(f"COUNT(*) OVER {frame} AS games_in_window_{w}")
    metric_names = [f"{n}_{w}" for w in windows for n in _RATE_METRICS] + [f"games_in_window_{w}" for w in windows]
    opp_cols = ", ".join(f"o.{c} AS opp_{c}" for c in metric_names)
    return f"""
    WITH sched AS ({schedule_long_sql()}),
    stats AS ({team_game_stats_sql()}),
    played AS (
        SELECT s.team, s.game_id, s.game_date, t.* EXCLUDE (game_id, team)
        FROM sched s JOIN stats t ON t.game_id = s.game_id AND t.team = s.team
    ),
    rolled AS (
        SELECT team, game_date AS last_played_date, {', '.join(roll_cols)} FROM played
    ),
    feat AS (
        SELECT s.season, s.week, s.game_type, s.game_id, s.game_date, s.team, s.opponent,
               s.is_home, s.is_neutral, s.rest_days, r.* EXCLUDE (team)
        FROM sched s ASOF LEFT JOIN rolled r ON s.team = r.team AND s.game_date > r.last_played_date
    )
    SELECT f.*, o.rest_days AS opp_rest_days, f.rest_days - o.rest_days AS rest_advantage, {opp_cols}
    FROM feat f
    LEFT JOIN feat o ON o.game_id = f.game_id AND o.team = f.opponent
    ORDER BY f.game_date, f.game_id, f.team
    """


def compute_team_features(con: duckdb.DuckDBPyConnection, windows: Sequence[int] = WINDOWS) -> pd.DataFrame:
    return con.execute(team_features_sql(windows)).df()


def build_team_features(con: duckdb.DuckDBPyConnection, windows: Sequence[int] = WINDOWS,
                        table: str = "team_features") -> int:
    con.execute(f"CREATE OR REPLACE TABLE {table} AS {team_features_sql(windows)}")
    n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchall()[0][0]
    logger.info("Wrote %d rows to %s", n, table)
    return n


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build point-in-time team features.")
    p.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    with duckdb.connect(str(args.db_path)) as con:
        build_team_features(con)
    return 0


if __name__ == "__main__":
    sys.exit(main())
