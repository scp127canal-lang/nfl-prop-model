"""Game-day weather from Open-Meteo (free, no API key) for the simulator's weather adjustment.

* ``STADIUMS``: home stadium coordinates and roof type for all 32 teams (nflverse abbreviations).
  ``dome`` = fixed roof (incl. SoFi's canopy); ``retractable`` = treated as closed (controlled)
  unless the schedule says ``roof = 'open'``; ``outdoor`` = forecast needed.
* :func:`get_game_weather` (``st.cache_data``, 30 min) returns a weather dict. Domes / closed roofs
  answer immediately **without any HTTP call**. Outdoor games query the Open-Meteo forecast API
  and take the hour nearest kickoff. Any failure, a kickoff outside the forecast horizon
  (past games or > 16 days out) or a neutral site returns neutral weather (no adjustment).

Rerun safety (CLAUDE.md rule 2): the app only calls this from the "Re-run Match Simulations"
button callback; normal reruns read the result from ``st.session_state``. Under pytest a live
request is refused unless a client is injected (tests mock the HTTP layer).

Needs no ``.env`` / ``st.secrets`` entry.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY = "temperature_2m,precipitation_probability,wind_speed_10m"
FORECAST_HORIZON_DAYS = 16
TIMEOUT_S = 6.0

# team -> (lat, lon, roof)
STADIUMS: dict[str, tuple[float, float, str]] = {
    "ARI": (33.5276, -112.2626, "retractable"),   # State Farm Stadium
    "ATL": (33.7554, -84.4008, "retractable"),    # Mercedes-Benz Stadium
    "BAL": (39.2780, -76.6227, "outdoor"),        # M&T Bank Stadium
    "BUF": (42.7738, -78.7870, "outdoor"),        # Highmark Stadium
    "CAR": (35.2258, -80.8528, "outdoor"),        # Bank of America Stadium
    "CHI": (41.8623, -87.6167, "outdoor"),        # Soldier Field
    "CIN": (39.0955, -84.5161, "outdoor"),        # Paycor Stadium
    "CLE": (41.5061, -81.6995, "outdoor"),        # Huntington Bank Field
    "DAL": (32.7473, -97.0945, "retractable"),    # AT&T Stadium
    "DEN": (39.7439, -105.0201, "outdoor"),       # Empower Field at Mile High
    "DET": (42.3400, -83.0456, "dome"),           # Ford Field
    "GB": (44.5013, -88.0622, "outdoor"),         # Lambeau Field
    "HOU": (29.6847, -95.4107, "retractable"),    # NRG Stadium
    "IND": (39.7601, -86.1639, "retractable"),    # Lucas Oil Stadium
    "JAX": (30.3239, -81.6373, "outdoor"),        # EverBank Stadium
    "KC": (39.0489, -94.4839, "outdoor"),         # GEHA Field at Arrowhead
    "LV": (36.0909, -115.1833, "dome"),           # Allegiant Stadium
    "LAC": (33.9535, -118.3392, "dome"),          # SoFi Stadium (fixed canopy)
    "LA": (33.9535, -118.3392, "dome"),           # SoFi Stadium
    "MIA": (25.9580, -80.2389, "outdoor"),        # Hard Rock Stadium
    "MIN": (44.9737, -93.2581, "dome"),           # U.S. Bank Stadium
    "NE": (42.0909, -71.2643, "outdoor"),         # Gillette Stadium
    "NO": (29.9511, -90.0812, "dome"),            # Caesars Superdome
    "NYG": (40.8135, -74.0745, "outdoor"),        # MetLife Stadium
    "NYJ": (40.8135, -74.0745, "outdoor"),        # MetLife Stadium
    "PHI": (39.9008, -75.1675, "outdoor"),        # Lincoln Financial Field
    "PIT": (40.4468, -80.0158, "outdoor"),        # Acrisure Stadium
    "SF": (37.4030, -121.9700, "outdoor"),        # Levi's Stadium
    "SEA": (47.5952, -122.3316, "outdoor"),       # Lumen Field
    "TB": (27.9759, -82.5033, "outdoor"),         # Raymond James Stadium
    "TEN": (36.1665, -86.7713, "outdoor"),        # Nissan Stadium
    "WAS": (38.9077, -76.8645, "outdoor"),        # Northwest Stadium
}

DOME_WEATHER = {"is_dome": True, "temp_f": 70.0, "wind_mph": 0.0, "precip_prob": 0, "summary": "Dome (Controlled)"}


def neutral_weather(reason: str) -> dict[str, Any]:
    """No-adjustment weather (used whenever a forecast is unavailable)."""
    return {"is_dome": False, "temp_f": None, "wind_mph": 0.0, "precip_prob": 0, "summary": reason,
            "available": False}


def is_controlled(home_team: str, roof: str | None = None) -> bool:
    """Dome or (closed / unknown-state) retractable roof. ``roof`` is nflverse ``schedules.roof``."""
    r = (roof or "").strip().lower() if isinstance(roof, str) else ""
    if r in ("dome", "closed"):
        return True
    if r in ("outdoors", "outdoor", "open"):
        return False
    kind = STADIUMS.get(home_team, (0, 0, "outdoor"))[2]
    return kind in ("dome", "retractable")


def kickoff_utc(gameday: Any, gametime: Any) -> datetime | None:
    """nflverse ``gameday`` + ``gametime`` (US Eastern) -> aware UTC datetime."""
    try:
        t = str(gametime) if isinstance(gametime, str) and gametime else "13:00"
        ts = pd.Timestamp(f"{pd.Timestamp(gameday).date()} {t}").tz_localize("America/New_York")
        return ts.tz_convert("UTC").to_pydatetime()
    except (ValueError, TypeError):
        return None


def _live_allowed(client: httpx.Client | None) -> bool:
    return client is not None or "PYTEST_CURRENT_TEST" not in os.environ


def nearest_hour(payload: dict, kickoff: datetime) -> dict[str, Any] | None:
    """Values of the hourly forecast slot nearest ``kickoff`` (UTC), or None if absent."""
    h = (payload or {}).get("hourly") or {}
    times = h.get("time") or []
    if not times:
        return None
    ts = pd.to_datetime(pd.Series(times), utc=True)
    k = pd.Timestamp(kickoff).tz_convert("UTC") if pd.Timestamp(kickoff).tzinfo else pd.Timestamp(kickoff, tz="UTC")
    i = int((ts - k).abs().idxmin())
    if abs((ts.iloc[i] - k).total_seconds()) > 3 * 3600:
        return None

    def val(key):
        arr = h.get(key) or []
        return arr[i] if i < len(arr) else None
    return {"time": str(ts.iloc[i]), "temp_f": val("temperature_2m"), "wind_mph": val("wind_speed_10m"),
            "precip_prob": val("precipitation_probability")}


def fetch_game_weather(home_team: str, kickoff: datetime, roof: str | None = None, neutral_site: bool = False,
                       client: httpx.Client | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Uncached weather lookup (see module docstring). ``client`` lets tests mock HTTP."""
    if is_controlled(home_team, roof) and not neutral_site:
        return dict(DOME_WEATHER)
    if neutral_site:
        return neutral_weather("Neutral site (no forecast)")
    if home_team not in STADIUMS:
        return neutral_weather(f"Unknown stadium for {home_team}")
    if kickoff is None:
        return neutral_weather("Unknown kickoff time")
    kickoff = kickoff if kickoff.tzinfo else kickoff.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    if kickoff < now - timedelta(hours=6) or kickoff > now + timedelta(days=FORECAST_HORIZON_DAYS):
        return neutral_weather("Outside forecast horizon")
    if not _live_allowed(client):
        return neutral_weather("Live weather disabled under pytest")
    lat, lon, _ = STADIUMS[home_team]
    # forecast_days (not start/end dates): Open-Meteo rejects an end date past its horizon, and
    # nearest_hour() returns None when kickoff isn't covered -> neutral fallback.
    params = {"latitude": lat, "longitude": lon, "hourly": HOURLY, "temperature_unit": "fahrenheit",
              "wind_speed_unit": "mph", "timezone": "UTC", "forecast_days": FORECAST_HORIZON_DAYS}
    own = client is None
    client = client or httpx.Client(timeout=TIMEOUT_S)
    try:
        r = client.get(FORECAST_URL, params=params)
        r.raise_for_status()
        slot = nearest_hour(r.json(), kickoff)
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Open-Meteo request failed for %s: %s", home_team, exc)
        return neutral_weather("Forecast unavailable")
    finally:
        if own:
            client.close()
    if slot is None or slot["wind_mph"] is None:
        return neutral_weather("Outside forecast horizon")
    temp = None if slot["temp_f"] is None else round(float(slot["temp_f"]), 1)
    wind = round(float(slot["wind_mph"]), 1)
    precip = int(slot["precip_prob"] or 0)
    summary = f"Outdoor: {temp:.0f}°F | Wind: {wind:.0f} mph | Precip: {precip}%" if temp is not None else \
        f"Outdoor: Wind: {wind:.0f} mph | Precip: {precip}%"
    return {"is_dome": False, "temp_f": temp, "wind_mph": wind, "precip_prob": precip, "summary": summary,
            "available": True, "forecast_hour_utc": slot["time"]}


@st.cache_data(ttl=1800, show_spinner=False)
def get_game_weather(home_team: str, kickoff_utc: datetime, roof: str | None = None,
                     neutral_site: bool = False) -> dict:
    """Cached (30 min) :func:`fetch_game_weather`."""
    return fetch_game_weather(home_team, kickoff_utc, roof=roof, neutral_site=neutral_site)
