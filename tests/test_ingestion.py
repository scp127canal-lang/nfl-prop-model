"""Tests for src/ingestion/nfl_data.py and src/utils/entity_resolution.py.

nflreadpy is never called: every test injects fake loaders that return small polars frames."""

from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd
import polars as pl
import pytest

from src.ingestion import nfl_data as nd
from src.ingestion import odds_client as oc
from src.utils import entity_resolution as er


# ======================================================================================
# Fake nflverse data
# ======================================================================================
def _schedules(seasons):
    rows = []
    for s in seasons:
        rows += [
            {"game_id": f"{s}_01_GB_DET", "season": s, "game_type": "REG", "week": 1, "gameday": f"{s}-09-10",
             "away_team": "GB", "home_team": "DET", "away_score": 20, "home_score": 27, "spread_line": 3.0,
             "total_line": 48.5},
            {"game_id": f"{s}_02_DET_CHI", "season": s, "game_type": "REG", "week": 2, "gameday": f"{s}-09-17",
             "away_team": "DET", "home_team": "CHI", "away_score": 31, "home_score": 17, "spread_line": -4.5,
             "total_line": 45.0},
        ]
    return pl.DataFrame(rows).with_columns(pl.col("season").cast(pl.Int32), pl.col("week").cast(pl.Int32))


def _players():
    return pl.DataFrame({
        "gsis_id": ["00-0001", "00-0002", "00-0003", None],
        "display_name": ["Jared Goff", "Amon-Ra St. Brown", "Jordan Love", "Nobody"],
        "pfr_id": ["GoffJa00", "StBrAm00", "LoveJo03", "Nobod00"],
        "position": ["QB", "WR", "QB", "WR"],
        "latest_team": ["DET", "DET", "GB", "DET"],
        "last_season": [2026, 2026, 2026, 2026],
    })


def _pbp(season, extra_col=False):
    df = pl.DataFrame({
        "play_id": [1, 2, 3], "game_id": [f"{season}_01_GB_DET"] * 2 + [None],
        "season": [season] * 3, "week": [1, 1, 1], "game_date": [f"{season}-09-10"] * 3,
        "posteam": ["DET", "GB", "DET"], "epa": [0.5, -0.2, 0.1], "weather": [None, None, None],
    })
    if extra_col:  # schema drift: new column + previously all-null column now has strings
        df = df.with_columns(pl.lit(1.5).alias("new_metric"), pl.lit("Sunny").alias("weather"))
    return df


def _player_weekly(season):
    return pl.DataFrame({
        "player_id": ["00-0001", "00-0002", "00-0003"], "player_display_name": ["Jared Goff", "Amon-Ra St. Brown", "Jordan Love"],
        "position": ["QB", "WR", "QB"], "season": [season] * 3, "week": [1, 1, 1], "season_type": ["REG"] * 3,
        "game_id": [f"{season}_01_GB_DET"] * 3, "team": ["DET", "DET", "GB"], "passing_yards": [280, 0, 240],
    })


def _snap_counts(season):
    return pl.DataFrame({
        "game_id": [f"{season}_01_GB_DET"] * 2, "season": [season] * 2, "week": [1, 1], "player": ["Jared Goff", "Unknown Guy"],
        "pfr_player_id": ["GoffJa00", "UnknGu00"], "position": ["QB", "TE"], "team": ["DET", "DET"],
        "offense_snaps": [65.0, 10.0], "offense_pct": [1.0, 0.15],
    })


def _injuries(season):
    return pl.DataFrame({
        "season": pl.Series([season] * 2, dtype=pl.Int32), "week": pl.Series([2, 2], dtype=pl.Int32),
        "game_type": ["REG", "REG"], "team": ["DET", "CHI"], "gsis_id": ["00-0002", "00-0099"],
        "full_name": ["Amon-Ra St. Brown", "Some Bear"], "report_status": ["Questionable", "Out"],
    })


def _rosters(season):
    return pl.DataFrame({
        "season": [season] * 3, "team": ["DET", "DET", "GB"], "position": ["QB", "WR", "QB"],
        "full_name": ["Jared Goff", "Amon-Ra St. Brown", "Jordan Love"], "gsis_id": ["00-0001", "00-0002", "00-0003"],
        "week": [1, 1, 1],
    })


def _participation(season):
    if season >= 2026:
        raise ValueError("Season must be between 2016 and 2025")
    return pl.DataFrame({"nflverse_game_id": [f"{season}_01_GB_DET"], "play_id": [1.0],
                         "possession_team": ["DET"], "offense_players": ["00-0001;00-0002"]})


class FakeLoaders(dict):
    def __init__(self, fail=None, drift_seasons=(), pbp_offset=0.0):
        self.calls: list[tuple[str, object]] = []
        fail = fail or set()

        def wrap(name, fn):
            def _inner(*args):
                arg = args[0] if args else None
                self.calls.append((name, arg))
                if (name, tuple(arg) if isinstance(arg, list) else arg) in fail:
                    raise ConnectionError(f"simulated failure {name}")
                return fn(*args)
            return _inner

        super().__init__({
            "players": wrap("players", _players),
            "schedules": wrap("schedules", _schedules),
            "pbp": wrap("pbp", lambda s: _pbp(s, s in drift_seasons).with_columns(pl.col("epa") + pbp_offset)),
            "player_weekly": wrap("player_weekly", _player_weekly),
            "snap_counts": wrap("snap_counts", _snap_counts),
            "injuries": wrap("injuries", _injuries),
            "rosters": wrap("rosters", _rosters),
            "participation": wrap("participation", _participation),
        })


@pytest.fixture
def con(tmp_path):
    c = duckdb.connect(str(tmp_path / "test.duckdb"))
    yield c
    c.close()


def _q(con, sql, params=None):
    return con.execute(sql, params or []).fetchall()


# ======================================================================================
# nfl_data
# ======================================================================================
def test_full_ingest_writes_all_tables(con):
    summary = nd.ingest(con, [2024, 2025], loaders=FakeLoaders())
    assert not summary.failures
    tables = {r[0] for r in _q(con, "SELECT table_name FROM information_schema.tables")}
    assert {"pbp", "player_weekly", "snap_counts", "injuries", "rosters", "schedules", "players", "ingestion_log"} <= tables

    # schedules / pbp: game_date is a real DATE
    assert _q(con, "SELECT DISTINCT typeof(game_date) FROM schedules") == [("DATE",)]
    assert _q(con, "SELECT DISTINCT typeof(game_date) FROM pbp") == [("DATE",)]
    assert _q(con, "SELECT COUNT(*) FROM pbp")[0][0] == 4  # null game_id rows dropped

    # player_weekly: player_id -> gsis_id and game_date joined
    cols = {r[0] for r in _q(con, "DESCRIBE player_weekly")}
    assert "gsis_id" in cols and "player_id" not in cols
    assert _q(con, "SELECT DISTINCT game_date FROM player_weekly WHERE season=2024") == [(date(2024, 9, 10),)]

    # snap_counts: gsis_id via pfr crosswalk (unknown players stay NULL)
    assert dict(_q(con, "SELECT player, gsis_id FROM snap_counts WHERE season=2025")) == {
        "Jared Goff": "00-0001", "Unknown Guy": None}

    # injuries joined to that team's game in that week
    assert _q(con, "SELECT team, game_id, game_date FROM injuries WHERE season=2025 ORDER BY team") == [
        ("CHI", "2025_02_DET_CHI", date(2025, 9, 17)), ("DET", "2025_02_DET_CHI", date(2025, 9, 17))]

    # players directory drops null gsis_id
    assert _q(con, "SELECT COUNT(*) FROM players")[0][0] == 3


def test_ingest_is_idempotent(con):
    nd.ingest(con, [2024, 2025], loaders=FakeLoaders())
    before = {t: _q(con, f"SELECT COUNT(*) FROM {t}")[0][0] for t in nd.SEASONAL_DATASETS}
    nd.ingest(con, [2024, 2025], loaders=FakeLoaders())
    after = {t: _q(con, f"SELECT COUNT(*) FROM {t}")[0][0] for t in nd.SEASONAL_DATASETS}
    assert before == after


def test_update_current_only_touches_active_season(con):
    nd.ingest(con, [2024, 2025, 2026], loaders=FakeLoaders())
    loaders = FakeLoaders(pbp_offset=10.0)
    nd.ingest(con, nd.resolve_seasons(update_current=True, current=2026), loaders=loaders)

    loaded_seasons = {s for name, s in loaders.calls if name not in ("players", "schedules")}
    assert loaded_seasons == {2026}
    assert ("schedules", [2026]) in loaders.calls
    epa = dict(_q(con, "SELECT season, MAX(epa) FROM pbp GROUP BY 1"))
    assert epa[2024] == pytest.approx(0.5) and epa[2025] == pytest.approx(0.5)
    assert epa[2026] == pytest.approx(10.5)
    assert _q(con, "SELECT COUNT(DISTINCT season) FROM schedules")[0][0] == 3


def test_resolve_seasons():
    assert nd.resolve_seasons(False, current=2026) == list(range(2018, 2027))
    assert nd.resolve_seasons(True, current=2026) == [2026]
    assert nd.resolve_seasons(False, start_season=2023, end_season=2024, current=2026) == [2023, 2024]
    with pytest.raises(ValueError):
        nd.resolve_seasons(False, start_season=2027, current=2026)


def test_schema_drift_adds_and_widens_columns(con):
    nd.ingest(con, [2024], datasets=["pbp"], loaders=FakeLoaders())
    nd.ingest(con, [2025], datasets=["pbp"], loaders=FakeLoaders(drift_seasons={2025}))
    rows = dict(_q(con, "SELECT season, MAX(weather) FROM pbp GROUP BY 1"))
    assert rows == {2024: None, 2025: "Sunny"}
    assert _q(con, "SELECT MAX(new_metric) FROM pbp WHERE season = 2025")[0][0] == 1.5


def test_failed_season_is_logged_and_others_continue(con):
    summary = nd.ingest(con, [2024, 2025], loaders=FakeLoaders(fail={("injuries", 2025)}))
    assert set(summary.failures) == {"injuries:2025"}
    assert _q(con, "SELECT DISTINCT season FROM injuries") == [(2024,)]
    assert _q(con, "SELECT COUNT(*) FROM pbp WHERE season = 2025")[0][0] == 2
    assert _q(con, "SELECT status FROM ingestion_log WHERE dataset='injuries' AND season=2025") == [("error",)]


def test_dataset_subset_still_refreshes_players_and_schedules(con):
    loaders = FakeLoaders()
    nd.ingest(con, [2025], datasets=["rosters"], loaders=loaders)
    assert {name for name, _ in loaders.calls} == {"players", "schedules", "rosters"}
    with pytest.raises(ValueError):
        nd.ingest(con, [2025], datasets=["bogus"], loaders=loaders)


def test_cli_update_current(tmp_path):
    db = tmp_path / "cli.duckdb"
    loaders = FakeLoaders()
    assert nd.main(["--update-current", "--db-path", str(db)], loaders=loaders, current=2026) == 0
    with duckdb.connect(str(db)) as c:
        assert _q(c, "SELECT DISTINCT season FROM pbp") == [(2026,)]
    assert nd.main(["--update-current", "--db-path", str(db)],
                   loaders=FakeLoaders(fail={("pbp", 2026)}), current=2026) == 2


# ======================================================================================
# entity_resolution
# ======================================================================================
@pytest.mark.parametrize("a,b", [
    ("A.J. Brown", "AJ Brown"), ("A. J. Brown", "aj brown"), ("Kenneth Walker III", "Kenneth Walker"),
    ("Odell Beckham Jr.", "Odell Beckham"), ("Amon-Ra St. Brown", "Amon Ra St Brown"),
    ("Ja'Marr Chase", "JaMarr Chase"), ("Matt Stafford", "Matthew Stafford"), ("Damián Lillard", "Damian Lillard"),
    ("D.K. Metcalf", "DK Metcalf"), ("Marvin Harrison Jr", "Marvin Harrison"),
])
def test_normalize_name_equivalences(a, b):
    assert er.normalize_name(a) == er.normalize_name(b)


def test_normalize_name_keeps_short_names():
    assert er.normalize_name("Kenneth Walker III") == "kenneth walker"
    assert er.normalize_name("Ben V") == "benjamin v"  # suffix not stripped from 2-token names
    assert er.normalize_name(None) == ""


def test_team_to_abbr():
    assert er.team_to_abbr("Los Angeles Rams") == "LA"
    assert er.team_to_abbr("Washington Commanders") == "WAS"
    assert er.team_to_abbr("lar") == "LA"
    assert er.team_to_abbr("Nowhere FC") is None


def test_market_positions():
    assert er.market_positions("player_pass_yds") == {"QB"}
    assert "LB" in er.market_positions("player_sacks")
    assert er.market_positions("player_field_goals") == {"K", "PK"}
    assert er.market_positions("h2h") is None


@pytest.fixture
def roster_con(con):
    rosters = pd.DataFrame([
        ("00-AJB", "A.J. Brown", "PHI", "WR"), ("00-KW3", "Kenneth Walker III", "SEA", "RB"),
        ("00-JAQ", "Josh Allen", "BUF", "QB"), ("00-JAL", "Josh Allen", "JAX", "LB"),
        ("00-MHJ", "Marvin Harrison Jr.", "ARI", "WR"), ("00-MBR", "Marquise Brown", "KC", "WR"),
        ("00-DKM", "DK Metcalf", "PIT", "WR"), ("00-CEE", "CeeDee Lamb", "DAL", "WR"),
        ("00-GDA", "Gabriel Davis", "BUF", "WR"), ("00-OLD", "A.J. Brown", "PHI", "WR"),
    ], columns=["gsis_id", "full_name", "team", "position"])
    rosters["season"] = [2026] * 9 + [2015]  # the 2015 duplicate must be excluded (too old)
    rosters["week"] = 10
    rosters["football_name"] = rosters["full_name"].str.split().str[0]
    rosters.loc[rosters.gsis_id == "00-MBR", "football_name"] = "Hollywood"
    rosters["last_name"] = rosters["full_name"].str.split().str[1]
    con.register("_r", rosters)
    con.execute("CREATE TABLE rosters AS SELECT * FROM _r")
    con.unregister("_r")
    return con


def test_load_candidates_latest_roster_and_aliases(roster_con):
    cands = {c.gsis_id: c for c in er.load_candidates(roster_con)}
    assert "00-OLD" not in cands
    assert "hollywood brown" in cands["00-MBR"].aliases
    assert cands["00-JAL"].team == "JAX"


def test_resolve_examples(roster_con):
    r = er.PlayerResolver(roster_con)
    ajb = r.resolve("A.J. Brown", ["Philadelphia Eagles", "Dallas Cowboys"], "player_reception_yds")
    assert (ajb.status, ajb.gsis_id, ajb.team, ajb.position, ajb.method) == ("matched", "00-AJB", "PHI", "WR", "exact")
    kw = r.resolve("Kenneth Walker III", ["Seattle Seahawks", "San Francisco 49ers"], "player_rush_yds")
    assert kw.matched and kw.gsis_id == "00-KW3" and kw.score == 100
    mhj = r.resolve("Marvin Harrison", ["Arizona Cardinals", "Los Angeles Rams"], None)
    assert mhj.matched and mhj.gsis_id == "00-MHJ"
    # Nickname alias via football_name
    assert r.resolve("Hollywood Brown", ["Kansas City Chiefs", "Denver Broncos"]).gsis_id == "00-MBR"


def test_same_name_disambiguation(roster_con):
    r = er.PlayerResolver(roster_con)
    assert r.resolve("Josh Allen", ["Buffalo Bills", "Miami Dolphins"], "player_pass_yds").gsis_id == "00-JAQ"
    assert r.resolve("Josh Allen", ["Jacksonville Jaguars", "Tennessee Titans"]).gsis_id == "00-JAL"
    sack = r.resolve("Josh Allen", None, "player_sacks")
    assert sack.matched and sack.gsis_id == "00-JAL"
    ambiguous = r.resolve("Josh Allen", None, None)
    assert not ambiguous.matched and ambiguous.reason.startswith("ambiguous")


def test_low_score_and_team_mismatch_go_to_review(roster_con):
    r = er.PlayerResolver(roster_con)
    gabe = r.resolve("Gabe Davis", ["Buffalo Bills", "New York Jets"], "player_reception_yds")
    assert gabe.matched  # nickname table: gabe -> gabriel
    low = r.resolve("Gabby Davies", ["Buffalo Bills", "New York Jets"])
    assert not low.matched and low.reason == "low_score" and low.score < 90
    assert low.alternatives and low.alternatives[0][0] == "00-GDA"
    moved = r.resolve("CeeDee Lamb", ["New York Giants", "Philadelphia Eagles"])
    assert not moved.matched and moved.reason.startswith("team_mismatch") and moved.score == 100


def test_cache_round_trip(roster_con):
    r = er.PlayerResolver(roster_con)
    first = r.resolve("A.J. Brown", ["Philadelphia Eagles", "Dallas Cowboys"])
    assert r.save([first]) == 1
    rows = _q(roster_con, "SELECT odds_name_norm, team, gsis_id, match_method FROM player_id_map")
    assert rows == [("aj brown", "PHI", "00-AJB", "exact")]
    # Cache hit even if the candidate pool is gone
    r2 = er.PlayerResolver(roster_con, candidates=[])
    again = r2.resolve("AJ Brown", ["Philadelphia Eagles", "New York Giants"])
    assert again.matched and again.gsis_id == "00-AJB" and again.method == "cache:exact"
    assert r2.save([again]) == 0  # cache hits are not re-written


def test_export_and_import_reviewed(roster_con, tmp_path):
    csv = tmp_path / "unmatched_players.csv"
    r = er.PlayerResolver(roster_con)
    results = r.resolve_many([
        ("A.J. Brown", ["PHI", "DAL"], None),
        ("Gabby Davies", ["BUF", "NYJ"], "player_receptions"),
        ("Zzyzx Qwerty", ["BUF", "NYJ"], None),
    ])
    assert r.export_unmatched(results, csv) == 2
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    assert set(df["odds_name"]) == {"Gabby Davies", "Zzyzx Qwerty"}
    assert list(df.columns) == er.UNMATCHED_COLUMNS
    assert (df["best_score"].astype(float) < 90).all()

    # Reviewer approves one row; re-export must keep the approval
    df.loc[df.odds_name == "Gabby Davies", "approved_gsis_id"] = "00-GDA"
    df.to_csv(csv, index=False)
    r.export_unmatched(results, csv)
    df2 = pd.read_csv(csv, dtype=str, keep_default_na=False)
    assert df2.set_index("odds_name").loc["Gabby Davies", "approved_gsis_id"] == "00-GDA"

    assert r.import_reviewed(csv) == 1
    hit = r.resolve("Gabby Davies", ["BUF", "NYJ"])
    assert hit.matched and hit.gsis_id == "00-GDA" and hit.method == "cache:manual"


def test_resolve_odds_players_end_to_end(roster_con, tmp_path):
    wrapper = {"fetched_at_utc": "2026-11-23T12:00:00+00:00", "request_params": {"oddsFormat": "american"},
               "data": {"id": "evt1", "home_team": "Philadelphia Eagles", "away_team": "Buffalo Bills",
                        "commence_time": "2026-11-29T18:00:00Z", "bookmakers": [{"key": "draftkings", "markets": [
                            {"key": "player_reception_yds", "outcomes": [
                                {"name": "Over", "description": "A.J. Brown", "price": -110, "point": 70.5},
                                {"name": "Over", "description": "Gabby Davies", "price": -110, "point": 30.5}]},
                            {"key": "player_pass_yds", "outcomes": [
                                {"name": "Over", "description": "Josh Allen", "price": -110, "point": 250.5}]},
                            {"key": "player_anytime_td", "outcomes": [
                                {"name": "Yes", "description": "Josh Allen", "price": 250}]},
                        ]}]}}
    oc.ingest_rows(roster_con, oc.parse_event_odds(wrapper, "f.json"))
    records = er.odds_player_records(roster_con)
    assert ("Josh Allen", ("Philadelphia Eagles", "Buffalo Bills"), "player_pass_yds") in records

    csv = tmp_path / "unmatched.csv"
    results = er.resolve_odds_players(roster_con, unmatched_path=csv)
    by_name = {r.odds_name: r for r in results}
    assert by_name["A.J. Brown"].gsis_id == "00-AJB"
    assert by_name["Josh Allen"].gsis_id == "00-JAQ"
    assert not by_name["Gabby Davies"].matched
    assert {r[0] for r in _q(roster_con, "SELECT gsis_id FROM player_id_map")} == {"00-AJB", "00-JAQ"}
    assert list(pd.read_csv(csv)["odds_name"]) == ["Gabby Davies"]


def test_participation_and_unavailable_season(con):
    summary = nd.ingest(con, [2025, 2026], datasets=["participation"], loaders=FakeLoaders())
    assert summary.unavailable == ["participation:2026"] and not summary.failures
    assert _q(con, "SELECT season, game_id, play_id FROM participation") == [(2025, "2025_01_GB_DET", 1)]
    assert _q(con, "SELECT status FROM ingestion_log WHERE dataset='participation' AND season=2026") == [("unavailable",)]


def test_season_week_forced_to_integer(con):
    loaders = FakeLoaders()
    loaders["injuries"] = lambda s: _injuries(s).with_columns(pl.col("season").cast(pl.Float64))
    nd.ingest(con, [2024], datasets=["injuries"], loaders=loaders)
    assert _q(con, "SELECT DISTINCT typeof(season) FROM injuries") == [("INTEGER",)]
