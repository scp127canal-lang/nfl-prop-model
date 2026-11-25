"""nflverse ingestion (via nflreadpy) into DuckDB.

Tables written to ``data/nfl_model.duckdb``:

* ``pbp``          play-by-play, ``game_date`` cast to DATE
* ``player_weekly`` weekly player stats; ``player_id`` renamed to ``gsis_id``; ``game_date`` joined from schedules
* ``snap_counts``  PFR snap counts with ``gsis_id`` (via players.pfr_id) and ``game_date``
* ``injuries``     weekly injury reports with ``game_id`` / ``game_date`` of that team's game
* ``rosters``      seasonal rosters (gsis_id, team, position, ...)
* ``schedules``    games with ``game_date`` DATE
* ``participation`` players on the field per play (route participation); nflverse publishes it
                   with a lag, so the newest season may be "unavailable" (logged, not a failure)
* ``players``      nflverse player directory (id crosswalk; used by entity resolution)
* ``ingestion_log`` one row per (dataset, season) load attempt

Every season-partitioned table is written with *replace-by-season* semantics, so re-running is
idempotent and ``--update-current`` only rewrites the active season. ``game_date`` is attached to
every per-game table so downstream features can enforce ``game_date < target_game_date``.

Usage
-----
    python -m src.ingestion.nfl_data                     # full backfill 2018..current
    python -m src.ingestion.nfl_data --update-current    # weekly refresh of the active season only
    python -m src.ingestion.nfl_data --datasets pbp,player_weekly --start-season 2022
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import duckdb
import polars as pl

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "nfl_model.duckdb"
START_SEASON = 2018

SEASONAL_DATASETS: tuple[str, ...] = ("schedules", "pbp", "player_weekly", "snap_counts", "injuries", "rosters",
                                      "participation")
DATASETS: tuple[str, ...] = ("players",) + SEASONAL_DATASETS

Loader = Callable[..., pl.DataFrame]


def default_loaders() -> dict[str, Loader]:
    import nflreadpy as nfl  # imported lazily so tests never touch the network

    return {
        "players": nfl.load_players,
        "schedules": nfl.load_schedules,
        "pbp": nfl.load_pbp,
        "player_weekly": lambda season: nfl.load_player_stats(season, summary_level="week"),
        "snap_counts": nfl.load_snap_counts,
        "injuries": nfl.load_injuries,
        "rosters": nfl.load_rosters,
        "participation": nfl.load_participation,
    }


def current_season() -> int:
    import nflreadpy as nfl

    return int(nfl.get_current_season())


# --------------------------------------------------------------------------------------
# Cleaning (pure polars; unit-testable)
# --------------------------------------------------------------------------------------
def _to_date(df: pl.DataFrame, src: str, dst: str) -> pl.DataFrame:
    if src not in df.columns:
        return df.with_columns(pl.lit(None, dtype=pl.Date).alias(dst))
    dtype = df.schema[src]
    if dtype == pl.Date:
        expr = pl.col(src)
    elif dtype == pl.Utf8:
        expr = pl.col(src).str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False)
    else:
        expr = pl.col(src).cast(pl.Date, strict=False)
    return df.with_columns(expr.alias(dst))


def clean_schedules(df: pl.DataFrame) -> pl.DataFrame:
    df = _to_date(df, "gameday", "game_date")
    return df.filter(pl.col("game_id").is_not_null()).unique(subset=["game_id"], keep="last")


def schedule_team_games(schedules: pl.DataFrame) -> pl.DataFrame:
    """Long form: one row per (season, week, team) -> game_id, game_date, opponent, is_home."""
    base = ["season", "week", "game_id", "game_date"]
    home = schedules.select(base + [pl.col("home_team").alias("team"), pl.col("away_team").alias("opponent"),
                                    pl.lit(True).alias("is_home")])
    away = schedules.select(base + [pl.col("away_team").alias("team"), pl.col("home_team").alias("opponent"),
                                    pl.lit(False).alias("is_home")])
    return pl.concat([home, away]).unique(subset=["season", "week", "team"], keep="first")


def _game_dates(schedules: pl.DataFrame) -> pl.DataFrame:
    return schedules.select("game_id", "game_date").unique(subset=["game_id"])


def _join_game_date(df: pl.DataFrame, schedules: pl.DataFrame | None) -> pl.DataFrame:
    df = df.drop("game_date", strict=False)
    if schedules is None or schedules.is_empty() or "game_id" not in df.columns:
        return df.with_columns(pl.lit(None, dtype=pl.Date).alias("game_date"))
    return df.join(_game_dates(schedules), on="game_id", how="left")


def clean_pbp(df: pl.DataFrame) -> pl.DataFrame:
    df = _to_date(df, "game_date", "game_date")
    return df.filter(pl.col("game_id").is_not_null())


def clean_player_weekly(df: pl.DataFrame, schedules: pl.DataFrame | None) -> pl.DataFrame:
    if "player_id" in df.columns and "gsis_id" not in df.columns:
        df = df.rename({"player_id": "gsis_id"})
    if "recent_team" in df.columns and "team" not in df.columns:  # older nflverse schema
        df = df.rename({"recent_team": "team"})
    df = df.filter(pl.col("gsis_id").is_not_null())
    return _join_game_date(df, schedules)


def clean_snap_counts(df: pl.DataFrame, schedules: pl.DataFrame | None, players: pl.DataFrame | None) -> pl.DataFrame:
    df = df.drop("gsis_id", strict=False)
    if players is not None and {"pfr_id", "gsis_id"} <= set(players.columns):
        xwalk = (players.select(pl.col("pfr_id").alias("pfr_player_id"), "gsis_id")
                 .drop_nulls().unique(subset=["pfr_player_id"], keep="first"))
        df = df.join(xwalk, on="pfr_player_id", how="left")
    else:
        df = df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("gsis_id"))
    return _join_game_date(df, schedules)


def clean_injuries(df: pl.DataFrame, schedules: pl.DataFrame | None) -> pl.DataFrame:
    df = df.drop(["game_id", "game_date"], strict=False)
    if schedules is None or schedules.is_empty():
        return df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("game_id"),
                               pl.lit(None, dtype=pl.Date).alias("game_date"))
    games = schedule_team_games(schedules).select("season", "week", "team", "game_id", "game_date")
    games = games.with_columns(pl.col("season").cast(df.schema["season"]), pl.col("week").cast(df.schema["week"]))
    return df.join(games, on=["season", "week", "team"], how="left")


def clean_rosters(df: pl.DataFrame) -> pl.DataFrame:
    return df


def clean_participation(df: pl.DataFrame, season: int) -> pl.DataFrame:
    if "nflverse_game_id" in df.columns:
        df = df.rename({"nflverse_game_id": "game_id"})
    return (df.filter(pl.col("game_id").is_not_null() & pl.col("play_id").is_not_null())
              .with_columns(pl.col("play_id").cast(pl.Int64), pl.lit(season, dtype=pl.Int32).alias("season")))


def _int_keys(df: pl.DataFrame) -> pl.DataFrame:
    """season/week must be integers in every table (some nflverse files ship them as floats)."""
    return df.with_columns([pl.col(c).cast(pl.Int32, strict=False) for c in ("season", "week") if c in df.columns])


def clean_players(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("gsis_id").is_not_null()).unique(subset=["gsis_id"], keep="first")


# --------------------------------------------------------------------------------------
# DuckDB writing with schema reconciliation
# --------------------------------------------------------------------------------------
_INT_TYPES = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT"}
_FLOAT_TYPES = {"FLOAT", "DOUBLE", "REAL"}


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return bool(con.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?", [table]).fetchall()[0][0])


def _describe(con: duckdb.DuckDBPyConnection, relation: str) -> dict[str, str]:
    return {row[0]: row[1] for row in con.execute(f'DESCRIBE "{relation}"').fetchall()}


def _widen(existing: str, incoming: str) -> str | None:
    """Type the table column must become to hold both; ``None`` if no change needed."""
    if existing == incoming or incoming == '"NULL"' or incoming == "NULL":
        return None
    numeric = _INT_TYPES | _FLOAT_TYPES
    if existing.startswith("DECIMAL") or incoming.startswith("DECIMAL"):
        return "DOUBLE" if existing != "DOUBLE" else None
    if existing in numeric and incoming in numeric:
        if existing in _FLOAT_TYPES or incoming in _FLOAT_TYPES:
            return "DOUBLE" if existing != "DOUBLE" else None
        return "BIGINT" if existing != "BIGINT" else None
    if existing == "VARCHAR":
        return None
    return "VARCHAR"


def _reconcile_schema(con: duckdb.DuckDBPyConnection, table: str, stage: str) -> None:
    existing = _describe(con, table)
    for col, typ in _describe(con, stage).items():
        if col not in existing:
            add_type = "VARCHAR" if typ in ('"NULL"', "NULL") else typ
            con.execute(f'ALTER TABLE "{table}" ADD COLUMN "{col}" {add_type}')
        else:
            target = _widen(existing[col], typ)
            if target:
                logger.info("Widening %s.%s %s -> %s", table, col, existing[col], target)
                con.execute(f'ALTER TABLE "{table}" ALTER COLUMN "{col}" TYPE {target}')


def write_table(con: duckdb.DuckDBPyConnection, table: str, df: pl.DataFrame,
                replace_seasons: Sequence[int] | None = None) -> int:
    """Write ``df`` to ``table``. With ``replace_seasons`` rows for those seasons are replaced;
    without it the whole table is replaced."""
    stage = f"_stage_{table}"
    con.register(stage, df.to_arrow())
    try:
        con.execute("BEGIN TRANSACTION")
        if replace_seasons is None or not _table_exists(con, table):
            if replace_seasons is None:
                con.execute(f'CREATE OR REPLACE TABLE "{table}" AS SELECT * FROM "{stage}"')
            else:
                con.execute(f'CREATE TABLE "{table}" AS SELECT * FROM "{stage}"')
        else:
            _reconcile_schema(con, table, stage)
            placeholders = ",".join("?" * len(replace_seasons))
            con.execute(f'DELETE FROM "{table}" WHERE season IN ({placeholders})', list(replace_seasons))
            con.execute(f'INSERT INTO "{table}" BY NAME SELECT * FROM "{stage}"')
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister(stage)
    return df.height


def _log(con: duckdb.DuckDBPyConnection, dataset: str, season: int | None, rows: int, status: str, message: str = "") -> None:
    con.execute("""CREATE TABLE IF NOT EXISTS ingestion_log (
        dataset VARCHAR, season INTEGER, rows BIGINT, status VARCHAR, message VARCHAR, loaded_at_utc TIMESTAMP)""")
    con.execute("INSERT INTO ingestion_log VALUES (?, ?, ?, ?, ?, ?)",
                [dataset, season, rows, status, message[:500],
                 datetime.now(timezone.utc).replace(tzinfo=None)])


# --------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------
@dataclass
class IngestSummary:
    seasons: list[int]
    rows: dict[str, int] = field(default_factory=dict)
    failures: dict[str, str] = field(default_factory=dict)
    unavailable: list[str] = field(default_factory=list)


def _read_existing(con: duckdb.DuckDBPyConnection, table: str, seasons: Sequence[int] | None = None) -> pl.DataFrame | None:
    if not _table_exists(con, table):
        return None
    if seasons is None:
        return con.execute(f'SELECT * FROM "{table}"').pl()
    placeholders = ",".join("?" * len(seasons))
    return con.execute(f'SELECT * FROM "{table}" WHERE season IN ({placeholders})', list(seasons)).pl()


def ingest(con: duckdb.DuckDBPyConnection, seasons: Sequence[int], datasets: Sequence[str] = DATASETS,
           loaders: Mapping[str, Loader] | None = None) -> IngestSummary:
    """Download and write the requested datasets for ``seasons``.

    ``players`` and ``schedules`` are always refreshed because other tables join against them.
    A failure for one (dataset, season) is logged and recorded but does not stop the run.
    """
    loaders = dict(loaders or default_loaders())
    seasons = sorted(set(int(s) for s in seasons))
    wanted = set(datasets) | {"players", "schedules"}
    unknown = wanted - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown datasets: {sorted(unknown)}")
    summary = IngestSummary(seasons=seasons)

    def record(dataset: str, season: int | None, rows: int | None, exc: Exception | None = None) -> None:
        key = dataset if season is None else f"{dataset}:{season}"
        if exc is None:
            summary.rows[key] = rows or 0
            _log(con, dataset, season, rows or 0, "ok")
            logger.info("%-14s %s -> %d rows", dataset, season if season is not None else "-", rows or 0)
        else:
            summary.failures[key] = f"{type(exc).__name__}: {exc}"
            _log(con, dataset, season, 0, "error", summary.failures[key])
            logger.warning("%-14s %s FAILED: %s", dataset, season if season is not None else "-", summary.failures[key])

    # players (season-less directory) -------------------------------------------------
    players: pl.DataFrame | None
    try:
        players = clean_players(loaders["players"]())
        record("players", None, write_table(con, "players", players))
    except Exception as exc:  # noqa: BLE001 - keep going, fall back to existing table
        record("players", None, None, exc)
        players = _read_existing(con, "players")

    # schedules (one call for all seasons, written per season) -------------------------
    schedules: pl.DataFrame | None
    try:
        schedules = _int_keys(clean_schedules(loaders["schedules"](seasons)))
        schedules = schedules.filter(pl.col("season").is_in(seasons))
        record("schedules", None, write_table(con, "schedules", schedules, replace_seasons=seasons))
    except Exception as exc:  # noqa: BLE001
        record("schedules", None, None, exc)
        schedules = _read_existing(con, "schedules", seasons)

    # per-season datasets ---------------------------------------------------------------
    for dataset in (d for d in SEASONAL_DATASETS if d in wanted and d != "schedules"):
        for season in seasons:
            try:
                raw = loaders[dataset](season)
                season_sched = schedules.filter(pl.col("season") == season) if schedules is not None else None
                if dataset == "pbp":
                    df = clean_pbp(raw)
                elif dataset == "player_weekly":
                    df = clean_player_weekly(raw, season_sched)
                elif dataset == "snap_counts":
                    df = clean_snap_counts(raw, season_sched, players)
                elif dataset == "injuries":
                    df = clean_injuries(raw, season_sched)
                elif dataset == "participation":
                    df = clean_participation(raw, season)
                else:
                    df = clean_rosters(raw)
                df = _int_keys(df)
                if df.is_empty():
                    record(dataset, season, 0)
                    continue
                record(dataset, season, write_table(con, dataset, df, replace_seasons=[season]))
            except ValueError as exc:
                if "season" not in str(exc).lower():
                    record(dataset, season, None, exc)
                    continue
                summary.unavailable.append(f"{dataset}:{season}")
                _log(con, dataset, season, 0, "unavailable", str(exc))
                logger.info("%-14s %s not published yet: %s", dataset, season, exc)
            except Exception as exc:  # noqa: BLE001
                record(dataset, season, None, exc)
    return summary


def resolve_seasons(update_current: bool, start_season: int = START_SEASON, end_season: int | None = None,
                    current: int | None = None) -> list[int]:
    current = current if current is not None else current_season()
    if update_current:
        return [current]
    end = end_season if end_season is not None else current
    if start_season > end:
        raise ValueError(f"start season {start_season} is after end season {end}")
    return list(range(start_season, end + 1))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ingest nflverse data into DuckDB via nflreadpy.")
    p.add_argument("--update-current", action="store_true", help="Refresh only the active season.")
    p.add_argument("--start-season", type=int, default=START_SEASON)
    p.add_argument("--end-season", type=int, default=None)
    p.add_argument("--datasets", default=",".join(DATASETS),
                   help=f"Comma list from {','.join(DATASETS)} (players & schedules always refresh).")
    p.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Sequence[str] | None = None, loaders: Mapping[str, Loader] | None = None,
         current: int | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    seasons = resolve_seasons(args.update_current, args.start_season, args.end_season, current)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    args.db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Ingesting %s for seasons %s", datasets, seasons)
    with duckdb.connect(str(args.db_path)) as con:
        summary = ingest(con, seasons, datasets, loaders)
    logger.info("Done: %d loads ok, %d failed", len(summary.rows), len(summary.failures))
    for key, err in summary.failures.items():
        logger.warning("  failed %s: %s", key, err)
    return 0 if not summary.failures else 2


if __name__ == "__main__":
    sys.exit(main())
