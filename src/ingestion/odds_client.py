"""Async client for The Odds API (v4) with credit guards, raw-JSON caching and DuckDB ingestion.

Usage
-----
    python -m src.ingestion.odds_client --dry-run            # default: re-ingest latest cached JSONs, 0 credits
    python -m src.ingestion.odds_client --live               # hit the API (costs credits)
    python -m src.ingestion.odds_client --live --markets main,player_props --max-credits 200

Credit safety (CLAUDE.md rule 1)
--------------------------------
* No HTTP request is ever sent unless the client was built with ``live=True`` (only set by the
  ``--live`` CLI flag or the Streamlit "Pull Live Odds" button).
* Under pytest, live requests are refused unless a mock ``transport`` is injected.
* Before each per-event request the cache in ``data/raw_odds/`` is inspected; a fresh snapshot
  that already covers the requested markets/regions is reused instead of re-fetching.
* An estimated per-event cost (markets x regions) is reserved against ``--max-credits`` and the
  last known ``x-requests-remaining`` (minus ``--min-remaining`` reserve) before every request.
* ``x-requests-remaining`` / ``x-requests-used`` / ``x-requests-last`` are logged after every
  live response, including error responses.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import duckdb
import httpx
import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw_odds"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "nfl_model.duckdb"

BASE_URL = "https://api.the-odds-api.com"
SPORT_KEY = "americanfootball_nfl"

MAIN_MARKETS: tuple[str, ...] = ("h2h", "spreads", "totals")
TEAM_TOTAL_MARKETS: tuple[str, ...] = ("team_totals",)
ALTERNATE_MARKETS: tuple[str, ...] = ("alternate_spreads", "alternate_totals", "alternate_team_totals")
PLAYER_PROP_MARKETS: tuple[str, ...] = (
    "player_pass_yds",
    "player_pass_tds",
    "player_pass_completions",
    "player_pass_attempts",
    "player_pass_interceptions",
    "player_rush_yds",
    "player_rush_attempts",
    "player_rush_longest",
    "player_reception_yds",
    "player_receptions",
    "player_reception_longest",
    "player_anytime_td",
    "player_1st_td",
    "player_last_td",
    "player_kicking_points",
    "player_field_goals",
    "player_tackles_assists",
    "player_sacks",
)

MARKET_GROUPS: dict[str, tuple[str, ...]] = {
    "main": MAIN_MARKETS,
    "team_totals": TEAM_TOTAL_MARKETS,
    "alternates": ALTERNATE_MARKETS,
    "player_props": PLAYER_PROP_MARKETS,
}
ALL_MARKETS: tuple[str, ...] = tuple(m for group in MARKET_GROUPS.values() for m in group)
_MARKET_TO_GROUP: dict[str, str] = {m: g for g, ms in MARKET_GROUPS.items() for m in ms}

QUOTA_HEADERS = ("x-requests-remaining", "x-requests-used", "x-requests-last")
_TS_FORMAT = "%Y%m%dT%H%M%S%fZ"
_FILENAME_RE = re.compile(r"^(?P<ts>\d{8}T\d{12}Z)__(?P<kind>events|odds)(?:__(?P<event_id>[A-Za-z0-9_-]+))?\.json$")
_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_-]")
_APIKEY_RE = re.compile(r"(apiKey=)[^&\s'\"]+", re.IGNORECASE)

RAW_ODDS_DDL = """
CREATE TABLE IF NOT EXISTS raw_odds (
    snapshot_ts_utc          TIMESTAMP NOT NULL,
    event_id                 VARCHAR   NOT NULL,
    sport_key                VARCHAR,
    commence_time_utc        TIMESTAMP,
    home_team                VARCHAR,
    away_team                VARCHAR,
    bookmaker_key            VARCHAR   NOT NULL,
    bookmaker_title          VARCHAR,
    bookmaker_last_update_utc TIMESTAMP,
    market_key               VARCHAR   NOT NULL,
    market_group             VARCHAR,
    market_last_update_utc   TIMESTAMP,
    is_player_prop           BOOLEAN,
    outcome_name             VARCHAR,
    outcome_description      VARCHAR,   -- player name for player props (pre entity-resolution)
    outcome_point            DOUBLE,
    price_american           DOUBLE,
    price_decimal            DOUBLE,
    implied_prob_vigged      DOUBLE,    -- raw 1/decimal; MUST be devigged before use (rule 5)
    source_file              VARCHAR   NOT NULL,
    ingested_at_utc          TIMESTAMP NOT NULL
)
"""
RAW_ODDS_COLUMNS = [
    "snapshot_ts_utc", "event_id", "sport_key", "commence_time_utc", "home_team", "away_team",
    "bookmaker_key", "bookmaker_title", "bookmaker_last_update_utc", "market_key", "market_group",
    "market_last_update_utc", "is_player_prop", "outcome_name", "outcome_description",
    "outcome_point", "price_american", "price_decimal", "implied_prob_vigged", "source_file",
    "ingested_at_utc",
]
_TS_COLUMNS = ["snapshot_ts_utc", "commence_time_utc", "bookmaker_last_update_utc",
               "market_last_update_utc", "ingested_at_utc"]


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------
class LiveCallBlockedError(RuntimeError):
    """Raised when a live HTTP call is attempted without explicit live authorization."""


class CreditBudgetExceededError(RuntimeError):
    """Raised when a request would exceed the configured or remaining credit budget."""


class OddsAPIError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        super().__init__(f"Odds API error {status_code}: {redact(message)}")
        self.status_code = status_code


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def redact(text: str) -> str:
    """Strip the apiKey query value from any string (URLs, exception messages)."""
    return _APIKEY_RE.sub(r"\1***", str(text))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_ts(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).strftime(_TS_FORMAT)


def parse_ts(ts: str) -> datetime:
    return datetime.strptime(ts, _TS_FORMAT).replace(tzinfo=timezone.utc)


def load_api_key(env_path: Path | None = None) -> str:
    """``ODDS_API_KEY`` from the environment (``.env``) first, then ``st.secrets`` (Streamlit Cloud)."""
    from src.utils.secrets import get_secret

    load_dotenv(env_path or PROJECT_ROOT / ".env")
    key = get_secret("ODDS_API_KEY")
    if not key:
        raise RuntimeError("ODDS_API_KEY is not set. Add it to .env (see .env.example) or to Streamlit secrets.")
    return key


def resolve_markets(spec: str | Sequence[str] | None) -> list[str]:
    """Expand group names (main, team_totals, alternates, player_props, all) and raw market keys."""
    if spec is None:
        return list(ALL_MARKETS)
    tokens = [t.strip() for t in (spec.split(",") if isinstance(spec, str) else spec) if t.strip()]
    out: list[str] = []
    for tok in tokens:
        expanded = ALL_MARKETS if tok == "all" else MARKET_GROUPS.get(tok, (tok,))
        out.extend(m for m in expanded if m not in out)
    if not out:
        raise ValueError("No markets selected.")
    return out


def estimate_event_cost(markets: Sequence[str], regions: Sequence[str], bookmakers: Sequence[str] | None = None) -> int:
    """Upper-bound credit cost of one per-event odds call.

    The Odds API charges (unique markets returned) x (regions); when ``bookmakers`` is used,
    every group of 10 bookmakers counts as one region. Markets with no data are not charged,
    so the true cost can be lower than this estimate.
    """
    region_units = math.ceil(len(bookmakers) / 10) if bookmakers else len(regions)
    return len(markets) * max(region_units, 1)


def american_to_decimal(price: float) -> float | None:
    if price is None or price == 0:
        return None
    return 1.0 + (price / 100.0 if price > 0 else 100.0 / abs(price))


def decimal_to_american(price: float) -> float | None:
    if price is None or price <= 1.0:
        return None
    return round((price - 1.0) * 100.0) if price >= 2.0 else round(-100.0 / (price - 1.0))


def _to_int(value: str | None) -> int | None:
    try:
        return int(float(value)) if value is not None else None
    except ValueError:
        return None


@dataclass
class QuotaState:
    remaining: int | None = None
    used: int | None = None
    last: int | None = None

    def update(self, headers: httpx.Headers) -> None:
        remaining = _to_int(headers.get("x-requests-remaining"))
        used = _to_int(headers.get("x-requests-used"))
        last = _to_int(headers.get("x-requests-last"))
        self.remaining = remaining if remaining is not None else self.remaining
        self.used = used if used is not None else self.used
        self.last = last

    def as_dict(self) -> dict[str, int | None]:
        return {"remaining": self.remaining, "used": self.used, "last": self.last}


# --------------------------------------------------------------------------------------
# Raw JSON cache
# --------------------------------------------------------------------------------------
class RawOddsCache:
    """Timestamped raw-payload store under ``data/raw_odds/``.

    Files: ``<UTC ts>__events.json`` and ``<UTC ts>__odds__<event_id>.json``. Each file wraps the
    API payload with request metadata (never the API key) and the quota headers.
    """

    def __init__(self, raw_dir: Path = DEFAULT_RAW_DIR):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def save(self, kind: str, data: Any, *, params: dict[str, Any], quota: dict[str, int | None],
             event_id: str | None = None, fetched_at: datetime | None = None) -> Path:
        fetched_at = fetched_at or utc_now()
        name = f"{format_ts(fetched_at)}__{kind}"
        if event_id:
            name += f"__{_SAFE_ID_RE.sub('_', event_id)}"
        path = self.raw_dir / f"{name}.json"
        wrapper = {
            "fetched_at_utc": fetched_at.astimezone(timezone.utc).isoformat(),
            "kind": kind,
            "sport_key": SPORT_KEY,
            "event_id": event_id,
            "request_params": {k: v for k, v in params.items() if k.lower() != "apikey"},
            "quota": quota,
            "data": data,
        }
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(wrapper, indent=2), encoding="utf-8")
        tmp.replace(path)
        logger.debug("Saved raw payload -> %s", path.name)
        return path

    def _iter(self, kind: str) -> Iterable[tuple[datetime, str | None, Path]]:
        for p in self.raw_dir.glob(f"*__{kind}*.json"):
            m = _FILENAME_RE.match(p.name)
            if m and m.group("kind") == kind:
                yield parse_ts(m.group("ts")), m.group("event_id"), p

    def latest_events_file(self) -> Path | None:
        files = sorted(self._iter("events"), key=lambda t: t[0])
        return files[-1][2] if files else None

    def latest_odds_files(self) -> dict[str, Path]:
        """Latest cached odds snapshot per event_id."""
        latest: dict[str, tuple[datetime, Path]] = {}
        for ts, event_id, path in self._iter("odds"):
            if event_id and (event_id not in latest or ts > latest[event_id][0]):
                latest[event_id] = (ts, path)
        return {eid: p for eid, (_, p) in latest.items()}

    @staticmethod
    def load(path: Path) -> dict[str, Any]:
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def fresh_odds_file(self, event_id: str, markets: Sequence[str], regions: Sequence[str],
                        bookmakers: Sequence[str] | None, ttl: timedelta, now: datetime | None = None) -> Path | None:
        """Return a cached snapshot for ``event_id`` that is younger than ``ttl`` and was requested
        with a superset of ``markets`` and the same regions/bookmakers; else ``None``."""
        if ttl <= timedelta(0):
            return None
        now = now or utc_now()
        safe_id = _SAFE_ID_RE.sub("_", event_id)
        candidates = sorted((t for t in self._iter("odds") if t[1] == safe_id), key=lambda t: t[0], reverse=True)
        for ts, _, path in candidates:
            if now - ts > ttl:
                break
            try:
                params = self.load(path).get("request_params", {})
            except (OSError, json.JSONDecodeError):
                continue
            cached_markets = set(str(params.get("markets", "")).split(","))
            same_scope = (str(params.get("regions", "")) == ",".join(regions) and
                          str(params.get("bookmakers", "")) == ",".join(bookmakers or []))
            if set(markets) <= cached_markets and same_scope:
                return path
        return None


# --------------------------------------------------------------------------------------
# HTTP client
# --------------------------------------------------------------------------------------
class OddsClient:
    """Async Odds API client. Refuses every HTTP request unless ``live=True``."""

    def __init__(self, api_key: str | None, *, live: bool = False, max_credits: int = 500,
                 min_remaining: int = 50, concurrency: int = 4, timeout: float = 30.0,
                 transport: httpx.AsyncBaseTransport | None = None, max_retries: int = 2,
                 retry_backoff: float = 2.0):
        self.api_key = api_key
        self.live = live
        self.max_credits = max_credits
        self.min_remaining = min_remaining
        self.quota = QuotaState()
        self.credits_spent = 0       # actual, from x-requests-last
        self._reserved = 0           # estimated, for in-flight requests
        self._budget_lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(max(concurrency, 1))
        self._transport = transport
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "OddsClient":
        self._assert_live_allowed()
        self._client = httpx.AsyncClient(base_url=BASE_URL, timeout=self._timeout, transport=self._transport)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _assert_live_allowed(self) -> None:
        if not self.live:
            raise LiveCallBlockedError("Live Odds API calls require --live (or the 'Pull Live Odds' button).")
        if "PYTEST_CURRENT_TEST" in os.environ and self._transport is None:
            raise LiveCallBlockedError("Refusing real Odds API calls under pytest; inject a mock transport.")
        if not self.api_key:
            raise RuntimeError("ODDS_API_KEY is required for live calls.")

    # -- budget -------------------------------------------------------------------------
    async def _reserve(self, estimate: int) -> None:
        async with self._budget_lock:
            committed = self.credits_spent + self._reserved
            if committed + estimate > self.max_credits:
                raise CreditBudgetExceededError(
                    f"Estimated cost {estimate} would exceed --max-credits {self.max_credits} "
                    f"(already committed {committed}).")
            if self.quota.remaining is not None:
                available = self.quota.remaining - self._reserved - self.min_remaining
                if estimate > available:
                    raise CreditBudgetExceededError(
                        f"Estimated cost {estimate} exceeds remaining quota {self.quota.remaining} "
                        f"minus reserve {self.min_remaining} (in-flight {self._reserved}).")
            self._reserved += estimate

    async def _settle(self, estimate: int, actual: int) -> None:
        async with self._budget_lock:
            self._reserved -= estimate
            self.credits_spent += actual

    # -- core request -------------------------------------------------------------------
    def _log_quota(self, label: str, response: httpx.Response) -> None:
        self.quota.update(response.headers)
        logger.info("Odds API quota | %s | status=%s | x-requests-remaining=%s | x-requests-last=%s | x-requests-used=%s",
                    label, response.status_code, response.headers.get("x-requests-remaining"),
                    response.headers.get("x-requests-last"), response.headers.get("x-requests-used"))

    async def _get(self, path: str, params: dict[str, Any], label: str, estimate: int = 0) -> Any:
        self._assert_live_allowed()
        if self._client is None:
            raise RuntimeError("OddsClient must be used as an async context manager.")
        await self._reserve(estimate)
        actual = 0  # read per-response so concurrent requests can't clobber each other's cost
        try:
            for attempt in range(self._max_retries + 1):
                async with self._sem:
                    try:
                        resp = await self._client.get(path, params={**params, "apiKey": self.api_key})
                    except httpx.HTTPError as exc:
                        raise OddsAPIError(0, f"{type(exc).__name__}: {exc}") from None
                self._log_quota(label, resp)
                actual += _to_int(resp.headers.get("x-requests-last")) or 0
                if resp.status_code == 429 and attempt < self._max_retries:
                    wait = self._retry_backoff * (2 ** attempt)
                    logger.warning("429 rate-limited on %s; retrying in %.1fs", label, wait)
                    await asyncio.sleep(wait)
                    continue
                if resp.status_code >= 400:
                    try:
                        msg = resp.json().get("message", resp.text)
                    except ValueError:
                        msg = resp.text
                    raise OddsAPIError(resp.status_code, msg)
                return resp.json()
        finally:
            await self._settle(estimate, actual)

    # -- endpoints ----------------------------------------------------------------------
    async def get_events(self, commence_time_from: str | None = None, commence_time_to: str | None = None) -> tuple[list[dict], dict]:
        """GET /v4/sports/americanfootball_nfl/events  (0 credits)."""
        params: dict[str, Any] = {"dateFormat": "iso"}
        if commence_time_from:
            params["commenceTimeFrom"] = commence_time_from
        if commence_time_to:
            params["commenceTimeTo"] = commence_time_to
        data = await self._get(f"/v4/sports/{SPORT_KEY}/events", params, label="events", estimate=0)
        return data, params

    async def get_event_odds(self, event_id: str, markets: Sequence[str], regions: Sequence[str] = ("us",),
                             bookmakers: Sequence[str] | None = None, odds_format: str = "american") -> tuple[dict, dict]:
        """GET /v4/sports/americanfootball_nfl/events/{eventId}/odds  (markets x regions credits)."""
        params: dict[str, Any] = {"markets": ",".join(markets), "oddsFormat": odds_format, "dateFormat": "iso"}
        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)
        else:
            params["regions"] = ",".join(regions)
        estimate = estimate_event_cost(markets, regions, bookmakers)
        data = await self._get(f"/v4/sports/{SPORT_KEY}/events/{event_id}/odds", params,
                               label=f"event_odds:{event_id}", estimate=estimate)
        return data, params


# --------------------------------------------------------------------------------------
# Parsing & DuckDB
# --------------------------------------------------------------------------------------
def parse_event_odds(wrapper: dict[str, Any], source_file: str, ingested_at: datetime | None = None) -> list[dict[str, Any]]:
    """Flatten one cached per-event odds payload into normalized ``raw_odds`` rows."""
    data = wrapper.get("data") or {}
    snapshot = wrapper.get("fetched_at_utc")
    odds_format = (wrapper.get("request_params") or {}).get("oddsFormat", "american")
    ingested = (ingested_at or utc_now()).isoformat()
    rows: list[dict[str, Any]] = []
    for book in data.get("bookmakers") or []:
        for market in book.get("markets") or []:
            mkey = market.get("key")
            for oc in market.get("outcomes") or []:
                price = oc.get("price")
                if price is None:
                    continue
                price = float(price)
                if odds_format == "decimal":
                    dec, amer = price, decimal_to_american(price)
                else:
                    amer, dec = price, american_to_decimal(price)
                point = oc.get("point")
                rows.append({
                    "snapshot_ts_utc": snapshot,
                    "event_id": data.get("id") or wrapper.get("event_id"),
                    "sport_key": data.get("sport_key"),
                    "commence_time_utc": data.get("commence_time"),
                    "home_team": data.get("home_team"),
                    "away_team": data.get("away_team"),
                    "bookmaker_key": book.get("key"),
                    "bookmaker_title": book.get("title"),
                    "bookmaker_last_update_utc": book.get("last_update"),
                    "market_key": mkey,
                    "market_group": _MARKET_TO_GROUP.get(mkey, "other"),
                    "market_last_update_utc": market.get("last_update"),
                    "is_player_prop": bool(mkey and mkey.startswith("player_")),
                    "outcome_name": oc.get("name"),
                    "outcome_description": oc.get("description"),
                    "outcome_point": float(point) if point is not None else None,
                    "price_american": amer,
                    "price_decimal": dec,
                    "implied_prob_vigged": (1.0 / dec) if dec else None,
                    "source_file": source_file,
                    "ingested_at_utc": ingested,
                })
    return rows


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(RAW_ODDS_DDL)


def ingest_rows(con: duckdb.DuckDBPyConnection, rows: list[dict[str, Any]]) -> int:
    """Insert rows into ``raw_odds``. Idempotent per ``source_file`` (re-ingesting replaces)."""
    ensure_schema(con)
    if not rows:
        return 0
    df = pd.DataFrame(rows, columns=RAW_ODDS_COLUMNS)
    for col in _TS_COLUMNS:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce").dt.tz_localize(None)
    sources = sorted(df["source_file"].unique().tolist())
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(f"DELETE FROM raw_odds WHERE source_file IN ({','.join('?' * len(sources))})", sources)
        con.register("_raw_odds_df", df)
        con.execute(f"INSERT INTO raw_odds ({', '.join(RAW_ODDS_COLUMNS)}) "
                    f"SELECT {', '.join(RAW_ODDS_COLUMNS)} FROM _raw_odds_df")
        con.unregister("_raw_odds_df")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(df)


def ingest_files(paths: Iterable[Path], db_path: Path = DEFAULT_DB_PATH) -> int:
    rows: list[dict[str, Any]] = []
    for p in paths:
        try:
            rows.extend(parse_event_odds(RawOddsCache.load(p), source_file=Path(p).name))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Skipping unreadable cache file %s: %s", Path(p).name, exc)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(db_path)) as con:
        n = ingest_rows(con, rows)
    logger.info("Ingested %d raw_odds rows into %s", n, Path(db_path).name)
    return n


# --------------------------------------------------------------------------------------
# Pipelines
# --------------------------------------------------------------------------------------
@dataclass
class RunSummary:
    mode: str
    events_seen: int = 0
    fetched: list[str] = field(default_factory=list)
    reused_from_cache: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    rows_ingested: int = 0
    credits_spent: int = 0
    quota: dict[str, int | None] = field(default_factory=dict)


def run_dry(raw_dir: Path = DEFAULT_RAW_DIR, db_path: Path = DEFAULT_DB_PATH,
            event_ids: Sequence[str] | None = None) -> RunSummary:
    """Zero-credit path: ingest the latest cached odds snapshot per event."""
    cache = RawOddsCache(raw_dir)
    latest = cache.latest_odds_files()
    if event_ids:
        latest = {k: v for k, v in latest.items() if k in set(event_ids)}
    summary = RunSummary(mode="dry-run", events_seen=len(latest), reused_from_cache=sorted(latest))
    if not latest:
        logger.warning("No cached odds snapshots found in %s; run with --live to fetch.", raw_dir)
        return summary
    summary.rows_ingested = ingest_files(latest.values(), db_path)
    return summary


async def run_live(api_key: str, *, markets: Sequence[str], regions: Sequence[str] = ("us",),
                   bookmakers: Sequence[str] | None = None, event_ids: Sequence[str] | None = None,
                   max_events: int | None = None, max_credits: int = 500, min_remaining: int = 50,
                   cache_ttl: timedelta = timedelta(minutes=15), concurrency: int = 4,
                   raw_dir: Path = DEFAULT_RAW_DIR, db_path: Path = DEFAULT_DB_PATH,
                   transport: httpx.AsyncBaseTransport | None = None) -> RunSummary:
    """Fetch events (0 credits) then per-event odds, caching every payload, then ingest."""
    cache = RawOddsCache(raw_dir)
    summary = RunSummary(mode="live")
    files_to_ingest: list[Path] = []

    async with OddsClient(api_key, live=True, max_credits=max_credits, min_remaining=min_remaining,
                          concurrency=concurrency, transport=transport) as client:
        events, ev_params = await client.get_events()
        cache.save("events", events, params=ev_params, quota=client.quota.as_dict())
        if event_ids:
            wanted = set(event_ids)
            events = [e for e in events if e.get("id") in wanted]
        events = sorted(events, key=lambda e: e.get("commence_time") or "")
        if max_events is not None:
            events = events[:max_events]
        summary.events_seen = len(events)
        logger.info("%d NFL events selected; markets=%d regions=%s est. max cost/event=%d",
                    len(events), len(markets), ",".join(bookmakers or regions),
                    estimate_event_cost(markets, regions, bookmakers))

        async def fetch_one(event: dict) -> None:
            eid = event["id"]
            cached = cache.fresh_odds_file(eid, markets, regions, bookmakers, cache_ttl)
            if cached is not None:
                logger.info("Cache hit for %s (%s); skipping live call", eid, cached.name)
                summary.reused_from_cache.append(eid)
                files_to_ingest.append(cached)
                return
            try:
                data, params = await client.get_event_odds(eid, markets, regions, bookmakers)
            except CreditBudgetExceededError as exc:
                summary.skipped[eid] = str(exc)
                logger.warning("Skipping %s: %s", eid, exc)
                return
            except OddsAPIError as exc:
                if exc.status_code == 401:
                    raise
                summary.skipped[eid] = str(exc)
                logger.error("Failed %s: %s", eid, exc)
                return
            path = cache.save("odds", data, params=params, quota=client.quota.as_dict(), event_id=eid)
            summary.fetched.append(eid)
            files_to_ingest.append(path)

        await asyncio.gather(*(fetch_one(e) for e in events))
        summary.credits_spent = client.credits_spent
        summary.quota = client.quota.as_dict()

    summary.rows_ingested = ingest_files(files_to_ingest, db_path) if files_to_ingest else 0
    return summary


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NFL Odds API ingestion (cache-first, credit-guarded).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Ingest latest cached JSONs only (default, 0 credits).")
    mode.add_argument("--live", action="store_true", help="Make live Odds API calls (costs credits).")
    p.add_argument("--markets", default="all",
                   help="Comma list of groups (main, team_totals, alternates, player_props, all) or raw market keys.")
    p.add_argument("--regions", default="us", help="Comma list of regions (e.g. us,eu; pinnacle is in eu).")
    p.add_argument("--bookmakers", default=None, help="Comma list of bookmaker keys (overrides --regions).")
    p.add_argument("--event-ids", default=None, help="Comma list of event IDs to restrict to.")
    p.add_argument("--max-events", type=int, default=None)
    p.add_argument("--max-credits", type=int, default=500, help="Hard credit budget for this run.")
    p.add_argument("--min-remaining", type=int, default=50, help="Never dip x-requests-remaining below this.")
    p.add_argument("--cache-ttl-minutes", type=float, default=15.0,
                   help="Reuse cached snapshots younger than this instead of re-fetching (0 disables).")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    p.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _split(value: str | None) -> list[str] | None:
    return [v.strip() for v in value.split(",") if v.strip()] if value else None


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # httpx logs full request URLs (including apiKey) at INFO; keep it quiet.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    event_ids = _split(args.event_ids)
    try:
        if args.live:
            summary = asyncio.run(run_live(
                load_api_key(), markets=resolve_markets(args.markets), regions=_split(args.regions) or ["us"],
                bookmakers=_split(args.bookmakers), event_ids=event_ids, max_events=args.max_events,
                max_credits=args.max_credits, min_remaining=args.min_remaining,
                cache_ttl=timedelta(minutes=args.cache_ttl_minutes), concurrency=args.concurrency,
                raw_dir=args.raw_dir, db_path=args.db_path))
        else:
            summary = run_dry(raw_dir=args.raw_dir, db_path=args.db_path, event_ids=event_ids)
    except (OddsAPIError, LiveCallBlockedError, CreditBudgetExceededError, RuntimeError) as exc:
        logger.error(redact(str(exc)))
        return 1

    logger.info("Done [%s]: events=%d fetched=%d cached=%d skipped=%d rows=%d credits_spent=%d quota=%s",
                summary.mode, summary.events_seen, len(summary.fetched), len(summary.reused_from_cache),
                len(summary.skipped), summary.rows_ingested, summary.credits_spent, summary.quota)
    return 0


if __name__ == "__main__":
    sys.exit(main())
