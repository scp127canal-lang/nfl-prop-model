"""Build the compact ``data/deploy_model.duckdb`` shipped to Streamlit Community Cloud.

Usage
-----
    python scripts/export_deploy_db.py                      # data/nfl_model.duckdb -> data/deploy_model.duckdb
    python scripts/export_deploy_db.py --src X.duckdb --dst Y.duckdb --max-mb 40

What is copied
--------------
* Full copies of the tables the UI and simulations read: ``team_features``, ``player_features``,
  ``player_id_map``, ``schedules``, ``backtest_results``, ``backtest_bets``, ``raw_odds``,
  ``td_predictions``, ``rosters`` and ``players`` (the last two feed the entity resolver, rule 4).
* ``player_weekly`` slimmed to the id/date/stat columns used by the simulator's efficiency priors
  and prop settlement.
* ``team_game_actuals``: the per-(game, offense) aggregate the game model trains on, computed
  from ``pbp`` with the exact same SQL as ``game_sim.build_game_dataset``.
* ``pbp`` itself (and the other raw nflverse tables) is excluded.

Missing optional tables are skipped with a warning. Makes no network calls.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.backtest import SETTLE_COLUMNS  # noqa: E402
from src.models.game_sim import TEAM_GAME_ACTUALS_SQL, TEAM_GAME_ACTUALS_TABLE  # noqa: E402

logger = logging.getLogger("export_deploy_db")

DEFAULT_SRC = PROJECT_ROOT / "data" / "nfl_model.duckdb"
DEFAULT_DST = PROJECT_ROOT / "data" / "deploy_model.duckdb"
DEFAULT_MAX_MB = 40.0

FULL_TABLES = ["team_features", "player_features", "player_id_map", "schedules", "backtest_results",
               "backtest_bets", "raw_odds", "td_predictions", "rosters", "players"]
REQUIRED_TABLES = {"team_features", "player_features", "schedules"}
PLAYER_WEEKLY_COLUMNS = ["gsis_id", "player_name", "player_display_name", "position", "team", "game_id",
                         "season", "week", "season_type", "game_date", "targets", "receptions",
                         "receiving_yards", "receiving_tds", "carries", "rushing_yards", "rushing_tds",
                         "attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions",
                         *SETTLE_COLUMNS.values()]


def _tables(con: duckdb.DuckDBPyConnection, schema: str) -> set[str]:
    return {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_catalog = ? AND table_schema = 'main'",
        [schema]).fetchall()}


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    return [r[0] for r in con.execute(f"DESCRIBE src.{table}").fetchall()]


def export(src: Path = DEFAULT_SRC, dst: Path = DEFAULT_DST, max_mb: float = DEFAULT_MAX_MB) -> dict[str, int]:
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Source DB not found: {src}")
    if src.resolve() == dst.resolve():
        raise ValueError("--dst must differ from --src")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    for p in (tmp, tmp.with_name(tmp.name + ".wal")):
        p.unlink(missing_ok=True)

    counts: dict[str, int] = {}
    con = duckdb.connect(str(tmp))
    try:
        con.execute(f"ATTACH '{src.as_posix()}' AS src (READ_ONLY)")
        have = _tables(con, "src")
        missing = REQUIRED_TABLES - have
        if missing:
            raise RuntimeError(f"Source DB lacks required tables: {sorted(missing)}")

        for t in FULL_TABLES:
            if t not in have:
                logger.warning("skipping %s (not in source DB)", t)
                continue
            con.execute(f'CREATE TABLE main."{t}" AS SELECT * FROM src."{t}"')

        if "player_weekly" in have:
            src_cols = _columns(con, "player_weekly")
            keep = [c for c in dict.fromkeys(PLAYER_WEEKLY_COLUMNS) if c in src_cols]
            con.execute(f'CREATE TABLE main.player_weekly AS SELECT {", ".join(keep)} FROM src.player_weekly')
        else:
            logger.warning("skipping player_weekly (not in source DB)")

        if "pbp" in have:
            con.execute(f"CREATE TABLE main.{TEAM_GAME_ACTUALS_TABLE} AS "
                        f"WITH pbp AS (SELECT * FROM src.pbp) {TEAM_GAME_ACTUALS_SQL}")
        elif TEAM_GAME_ACTUALS_TABLE in have:
            con.execute(f"CREATE TABLE main.{TEAM_GAME_ACTUALS_TABLE} AS SELECT * FROM src.{TEAM_GAME_ACTUALS_TABLE}")
        else:
            raise RuntimeError("Source DB has neither pbp nor team_game_actuals; the game model can't be fit")

        con.execute("DETACH src")
        for t in sorted(_tables(con, con.execute("SELECT current_database()").fetchone()[0])):
            counts[t] = con.execute(f'SELECT COUNT(*) FROM main."{t}"').fetchone()[0]
        con.execute("CHECKPOINT")
    finally:
        con.close()

    size_mb = tmp.stat().st_size / 1e6
    if size_mb > max_mb:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Deploy DB would be {size_mb:.1f} MB (> {max_mb} MB limit); not written")
    for p in (dst, dst.with_name(dst.name + ".wal")):
        p.unlink(missing_ok=True)
    tmp.replace(dst)
    logger.info("wrote %s (%.1f MB)", dst, size_mb)
    return counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--dst", type=Path, default=DEFAULT_DST)
    p.add_argument("--max-mb", type=float, default=DEFAULT_MAX_MB)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    counts = export(args.src, args.dst, args.max_mb)
    for t, n in sorted(counts.items()):
        print(f"  {t:<20} {n:>9,} rows")
    print(f"{args.dst}  {args.dst.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
