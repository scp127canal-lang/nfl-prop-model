"""Mocked tests for src/ingestion/odds_client.py. No real network traffic is possible here:
an autouse fixture makes the real httpx transport raise, and all HTTP goes through MockTransport."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import httpx
import pytest

from src.ingestion import odds_client as oc

API_KEY = "test-secret-key-123"
EVENT_A = "aaaa1111"
EVENT_B = "bbbb2222"


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch):
    async def _boom(*args, **kwargs):
        raise AssertionError("Real network access attempted in tests!")
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _boom)


@pytest.fixture
def paths(tmp_path):
    return tmp_path / "raw_odds", tmp_path / "test.duckdb"


def _events_payload():
    return [
        {"id": EVENT_B, "sport_key": oc.SPORT_KEY, "commence_time": "2026-11-29T21:25:00Z",
         "home_team": "Kansas City Chiefs", "away_team": "Buffalo Bills"},
        {"id": EVENT_A, "sport_key": oc.SPORT_KEY, "commence_time": "2026-11-29T18:00:00Z",
         "home_team": "Detroit Lions", "away_team": "Green Bay Packers"},
    ]


def _odds_payload(event_id: str):
    return {
        "id": event_id, "sport_key": oc.SPORT_KEY, "commence_time": "2026-11-29T18:00:00Z",
        "home_team": "Detroit Lions", "away_team": "Green Bay Packers",
        "bookmakers": [
            {"key": "draftkings", "title": "DraftKings", "last_update": "2026-11-23T20:00:00Z",
             "markets": [
                 {"key": "h2h", "last_update": "2026-11-23T20:00:00Z", "outcomes": [
                     {"name": "Detroit Lions", "price": -150},
                     {"name": "Green Bay Packers", "price": 130}]},
                 {"key": "spreads", "last_update": "2026-11-23T20:00:00Z", "outcomes": [
                     {"name": "Detroit Lions", "price": -110, "point": -3.5},
                     {"name": "Green Bay Packers", "price": -110, "point": 3.5}]},
                 {"key": "player_pass_yds", "last_update": "2026-11-23T20:00:00Z", "outcomes": [
                     {"name": "Over", "description": "Jared Goff", "price": -115, "point": 265.5},
                     {"name": "Under", "description": "Jared Goff", "price": -105, "point": 265.5}]},
                 {"key": "player_anytime_td", "last_update": "2026-11-23T20:00:00Z", "outcomes": [
                     {"name": "Yes", "description": "Jahmyr Gibbs", "price": -140}]},
             ]},
        ],
    }


class FakeOddsAPI:
    """MockTransport handler emulating the two endpoints and quota headers."""

    def __init__(self, remaining: int = 10_000, odds_status: dict[str, int] | None = None,
                 events_status: int = 200):
        self.remaining = remaining
        self.used = 0
        self.requests: list[httpx.Request] = []
        self.odds_status = odds_status or {}
        self.events_status = events_status

    def _headers(self, last: int) -> dict[str, str]:
        self.remaining -= last
        self.used += last
        return {"x-requests-remaining": str(self.remaining), "x-requests-used": str(self.used),
                "x-requests-last": str(last)}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == f"/v4/sports/{oc.SPORT_KEY}/events":
            if self.events_status != 200:
                return httpx.Response(self.events_status, json={"message": "bad key"}, headers=self._headers(0))
            return httpx.Response(200, json=_events_payload(), headers=self._headers(0))
        if path.startswith(f"/v4/sports/{oc.SPORT_KEY}/events/") and path.endswith("/odds"):
            event_id = path.split("/")[-2]
            status = self.odds_status.get(event_id, 200)
            if status != 200:
                return httpx.Response(status, json={"message": "Invalid market"}, headers=self._headers(0))
            markets = request.url.params["markets"].split(",")
            regions = request.url.params.get("regions", "us").split(",")
            return httpx.Response(200, json=_odds_payload(event_id),
                                  headers=self._headers(len(markets) * len(regions)))
        return httpx.Response(404, json={"message": "not found"})

    @property
    def odds_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path.endswith("/odds")]

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


async def _run(api: FakeOddsAPI, paths, **kw):
    raw_dir, db_path = paths
    kw.setdefault("markets", oc.resolve_markets("main"))
    return await oc.run_live(API_KEY, raw_dir=raw_dir, db_path=db_path, transport=api.transport(), **kw)


def _db_rows(db_path: Path, sql: str):
    with duckdb.connect(str(db_path)) as con:
        return con.execute(sql).fetchall()


# --------------------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------------------
def test_resolve_markets_groups_and_raw_keys():
    assert oc.resolve_markets("main") == ["h2h", "spreads", "totals"]
    allm = oc.resolve_markets("all")
    assert len(oc.PLAYER_PROP_MARKETS) == 18
    for m in ("team_totals", "alternate_spreads", "alternate_totals", "player_sacks", "player_1st_td"):
        assert m in allm
    assert len(allm) == len(set(allm))
    assert oc.resolve_markets("main,h2h,player_receptions") == ["h2h", "spreads", "totals", "player_receptions"]
    with pytest.raises(ValueError):
        oc.resolve_markets(" , ")


def test_estimate_event_cost():
    assert oc.estimate_event_cost(["h2h", "spreads"], ["us"]) == 2
    assert oc.estimate_event_cost(["h2h", "spreads"], ["us", "eu"]) == 4
    assert oc.estimate_event_cost(["h2h"], ["us", "eu"], bookmakers=["pinnacle"] * 11) == 2


@pytest.mark.parametrize("american,decimal", [(-110, 1.909090909), (150, 2.5), (100, 2.0), (-200, 1.5)])
def test_odds_conversion(american, decimal):
    assert oc.american_to_decimal(american) == pytest.approx(decimal)
    assert oc.decimal_to_american(decimal) == pytest.approx(american if american != -100 else 100)


def test_redact_hides_api_key():
    url = f"https://api.the-odds-api.com/v4/sports/x/events?apiKey={API_KEY}&dateFormat=iso"
    assert API_KEY not in oc.redact(url)
    assert "apiKey=***" in oc.redact(url)


def test_load_api_key_from_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("ODDS_API_KEY=abc123\n", encoding="utf-8")
    assert oc.load_api_key(env) == "abc123"


def test_load_api_key_missing_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ODDS_API_KEY"):
        oc.load_api_key(tmp_path / "missing.env")


# --------------------------------------------------------------------------------------
# Live-call guards
# --------------------------------------------------------------------------------------
async def test_client_refuses_without_live_flag():
    with pytest.raises(oc.LiveCallBlockedError):
        async with oc.OddsClient(API_KEY, live=False, transport=FakeOddsAPI().transport()):
            pass


async def test_client_refuses_real_transport_under_pytest():
    with pytest.raises(oc.LiveCallBlockedError, match="pytest"):
        async with oc.OddsClient(API_KEY, live=True):
            pass


# --------------------------------------------------------------------------------------
# Live pipeline (mocked)
# --------------------------------------------------------------------------------------
async def test_run_live_fetches_caches_and_ingests(paths, caplog):
    raw_dir, db_path = paths
    api = FakeOddsAPI()
    with caplog.at_level(logging.INFO, logger=oc.__name__):
        summary = await _run(api, paths, markets=oc.resolve_markets("all"))

    # Endpoints & params
    assert api.requests[0].url.path == f"/v4/sports/{oc.SPORT_KEY}/events"
    assert len(api.odds_requests) == 2
    sent_markets = api.odds_requests[0].url.params["markets"].split(",")
    assert set(oc.PLAYER_PROP_MARKETS) <= set(sent_markets)
    assert {"h2h", "spreads", "totals", "team_totals", "alternate_spreads"} <= set(sent_markets)
    assert all(r.url.params["apiKey"] == API_KEY for r in api.requests)

    # Quota headers logged for every live request
    quota_logs = [r.getMessage() for r in caplog.records if "x-requests-remaining" in r.getMessage()]
    assert len(quota_logs) == len(api.requests) == 3
    assert all("x-requests-last=" in m for m in quota_logs)

    # Raw JSON cached with UTC timestamps and without the API key
    files = sorted(p.name for p in raw_dir.glob("*.json"))
    assert len(files) == 3
    assert sum("__events" in f for f in files) == 1
    assert {f"{EVENT_A}", f"{EVENT_B}"} == {f.split("__")[-1][:-5] for f in files if "__odds__" in f}
    for p in raw_dir.glob("*.json"):
        text = p.read_text(encoding="utf-8")
        assert API_KEY not in text
        wrapper = json.loads(text)
        assert datetime.fromisoformat(wrapper["fetched_at_utc"]).utcoffset() == timedelta(0)
        assert "quota" in wrapper and "data" in wrapper

    # DuckDB rows
    assert summary.rows_ingested == 2 * 7
    assert sorted(summary.fetched) == [EVENT_A, EVENT_B]
    assert summary.credits_spent == 2 * len(oc.ALL_MARKETS)
    rows = _db_rows(db_path, """
        SELECT market_key, market_group, is_player_prop, outcome_name, outcome_description,
               outcome_point, price_american, price_decimal
        FROM raw_odds WHERE event_id = 'aaaa1111' AND market_key = 'player_pass_yds' AND outcome_name = 'Over'""")
    assert rows == [("player_pass_yds", "player_props", True, "Over", "Jared Goff", 265.5, -115.0,
                     pytest.approx(1 + 100 / 115))]
    groups = dict(_db_rows(db_path, "SELECT market_key, ANY_VALUE(market_group) FROM raw_odds GROUP BY 1"))
    assert groups["h2h"] == "main" and groups["spreads"] == "main"


async def test_run_live_reuses_fresh_cache(paths):
    api = FakeOddsAPI()
    await _run(api, paths)
    api2 = FakeOddsAPI()
    summary = await _run(api2, paths, cache_ttl=timedelta(minutes=30))
    assert api2.odds_requests == []  # events call only (0 credits)
    assert sorted(summary.reused_from_cache) == [EVENT_A, EVENT_B]
    assert summary.credits_spent == 0
    # Re-ingesting the same files must not duplicate rows
    assert _db_rows(paths[1], "SELECT COUNT(*) FROM raw_odds")[0][0] == 2 * 7


async def test_cache_not_reused_for_wider_markets_or_zero_ttl(paths):
    await _run(FakeOddsAPI(), paths, markets=["h2h"])
    api = FakeOddsAPI()
    await _run(api, paths, markets=["h2h", "totals"], cache_ttl=timedelta(hours=1))
    assert len(api.odds_requests) == 2
    api3 = FakeOddsAPI()
    await _run(api3, paths, markets=["h2h"], cache_ttl=timedelta(0))
    assert len(api3.odds_requests) == 2


async def test_max_credits_budget_blocks_requests(paths):
    api = FakeOddsAPI()
    # 3 markets x 1 region = 3 credits/event; budget 4 allows exactly one event.
    summary = await _run(api, paths, max_credits=4, concurrency=1)
    assert len(api.odds_requests) == 1
    assert len(summary.skipped) == 1
    assert "max-credits" in next(iter(summary.skipped.values()))


async def test_min_remaining_reserve_blocks_requests(paths):
    api = FakeOddsAPI(remaining=52)  # events call reports 52 remaining; reserve 50 -> only 2 available
    summary = await _run(api, paths, min_remaining=50)
    assert api.odds_requests == []
    assert set(summary.skipped) == {EVENT_A, EVENT_B}


async def test_event_error_is_skipped_but_others_ingest(paths):
    api = FakeOddsAPI(odds_status={EVENT_B: 422})
    summary = await _run(api, paths)
    assert summary.fetched == [EVENT_A]
    assert "422" in summary.skipped[EVENT_B]
    assert {r[0] for r in _db_rows(paths[1], "SELECT DISTINCT event_id FROM raw_odds")} == {EVENT_A}


async def test_unauthorized_raises(paths):
    with pytest.raises(oc.OddsAPIError) as exc:
        await _run(FakeOddsAPI(events_status=401), paths)
    assert exc.value.status_code == 401
    assert API_KEY not in str(exc.value)


async def test_max_events_and_event_filter(paths):
    api = FakeOddsAPI()
    summary = await _run(api, paths, max_events=1)
    assert summary.fetched == [EVENT_A]  # earliest kickoff first
    api2 = FakeOddsAPI()
    await _run(api2, paths, event_ids=[EVENT_B], cache_ttl=timedelta(0))
    assert [r.url.path.split("/")[-2] for r in api2.odds_requests] == [EVENT_B]


# --------------------------------------------------------------------------------------
# Dry-run / cache
# --------------------------------------------------------------------------------------
def _write_snapshot(raw_dir: Path, event_id: str, ts: datetime, price: int) -> Path:
    payload = _odds_payload(event_id)
    payload["bookmakers"][0]["markets"][0]["outcomes"][0]["price"] = price
    return oc.RawOddsCache(raw_dir).save("odds", payload, event_id=event_id, fetched_at=ts,
                                         params={"markets": "h2h", "regions": "us", "oddsFormat": "american",
                                                 "apiKey": API_KEY}, quota={})


def test_run_dry_uses_latest_snapshot_per_event(paths):
    raw_dir, db_path = paths
    t0 = datetime(2026, 11, 23, 12, tzinfo=timezone.utc)
    _write_snapshot(raw_dir, EVENT_A, t0, -150)
    latest = _write_snapshot(raw_dir, EVENT_A, t0 + timedelta(hours=1), -170)
    _write_snapshot(raw_dir, EVENT_B, t0, -120)

    summary = oc.run_dry(raw_dir=raw_dir, db_path=db_path)
    assert summary.mode == "dry-run" and summary.events_seen == 2
    price = _db_rows(db_path, f"""SELECT price_american FROM raw_odds WHERE event_id='{EVENT_A}'
                                  AND market_key='h2h' AND outcome_name='Detroit Lions'""")
    assert price == [(-170.0,)]
    assert {r[0] for r in _db_rows(db_path, f"SELECT DISTINCT source_file FROM raw_odds WHERE event_id='{EVENT_A}'")} == {latest.name}
    assert API_KEY not in latest.read_text(encoding="utf-8")

    oc.run_dry(raw_dir=raw_dir, db_path=db_path)  # idempotent
    assert _db_rows(db_path, "SELECT COUNT(*) FROM raw_odds")[0][0] == 14


def test_run_dry_empty_cache(paths):
    summary = oc.run_dry(raw_dir=paths[0], db_path=paths[1])
    assert summary.rows_ingested == 0


def test_parse_decimal_odds_format():
    wrapper = {"fetched_at_utc": "2026-11-23T12:00:00+00:00", "request_params": {"oddsFormat": "decimal"},
               "data": {"id": "x", "bookmakers": [{"key": "pinnacle", "markets": [
                   {"key": "h2h", "outcomes": [{"name": "A", "price": 2.5}]}]}]}}
    [row] = oc.parse_event_odds(wrapper, source_file="f.json")
    assert row["price_decimal"] == 2.5 and row["price_american"] == 150
    assert row["implied_prob_vigged"] == pytest.approx(0.4)
    assert row["is_player_prop"] is False


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def test_cli_defaults_to_dry_run_without_network(paths, monkeypatch):
    raw_dir, db_path = paths
    _write_snapshot(raw_dir, EVENT_A, datetime(2026, 11, 23, tzinfo=timezone.utc), -150)

    def _no_live(*a, **k):
        raise AssertionError("run_live must not be called in dry-run")
    monkeypatch.setattr(oc, "run_live", _no_live)
    assert oc.main(["--raw-dir", str(raw_dir), "--db-path", str(db_path)]) == 0
    assert oc.main(["--dry-run", "--raw-dir", str(raw_dir), "--db-path", str(db_path)]) == 0
    assert _db_rows(db_path, "SELECT COUNT(*) FROM raw_odds")[0][0] == 7


def test_cli_live_blocked_under_pytest(paths, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", API_KEY)
    # No transport injected -> client refuses because we are under pytest; exits non-zero.
    assert oc.main(["--live", "--raw-dir", str(paths[0]), "--db-path", str(paths[1])]) == 1
    assert not list(paths[0].glob("*.json")) if paths[0].exists() else True


def test_cli_rejects_both_flags():
    with pytest.raises(SystemExit):
        oc.main(["--live", "--dry-run"])
