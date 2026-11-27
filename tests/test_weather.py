"""Tests for src/ingestion/weather_client.py and the weather adjustment in the simulator / app.

No real HTTP: outdoor lookups go through an ``httpx.MockTransport``; domes must never request."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from src.app import components as ui
from src.app import state
from src.ingestion import weather_client as wc
from src.models import game_sim as gs
from tests.test_app import _run_app, env, app_env, no_network  # noqa: F401 - fixtures
from tests.test_models import con, upcoming_game  # noqa: F401 - fixtures

NOW = datetime(2026, 12, 3, 12, 0, tzinfo=timezone.utc)
KICK = datetime(2026, 12, 6, 18, 0, tzinfo=timezone.utc)      # 1:00 pm ET


def _payload(wind: float = 22.0, temp: float = 38.0, precip: int = 40) -> dict:
    times = [f"2026-12-06T{h:02d}:00" for h in range(24)]
    return {"hourly": {"time": times,
                       "temperature_2m": [temp - 5 + (h == 18) * 5 for h in range(24)],
                       "wind_speed_10m": [5.0 if h != 18 else wind for h in range(24)],
                       "precipitation_probability": [0 if h != 18 else precip for h in range(24)]}}


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _no_http(request):
    raise AssertionError(f"unexpected HTTP request: {request.url}")


def test_all_32_teams_have_stadiums():
    assert len(wc.STADIUMS) == 32
    assert all(r in ("dome", "outdoor", "retractable") for _, _, r in wc.STADIUMS.values())
    assert wc.STADIUMS["DET"][2] == "dome" and wc.STADIUMS["GB"][2] == "outdoor"


def test_dome_returns_controlled_weather_without_http():
    w = wc.fetch_game_weather("DET", KICK, client=_client(_no_http), now=NOW)
    assert w == {"is_dome": True, "temp_f": 70.0, "wind_mph": 0.0, "precip_prob": 0, "summary": "Dome (Controlled)"}
    assert wc.fetch_game_weather("DAL", KICK, client=_client(_no_http), now=NOW)["is_dome"]     # retractable
    assert wc.fetch_game_weather("GB", KICK, roof="closed", client=_client(_no_http), now=NOW)["is_dome"]
    assert wc.get_game_weather("NO", KICK)["summary"] == "Dome (Controlled)"                  # cached wrapper


def test_outdoor_windy_game_queries_open_meteo_nearest_kickoff_hour():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=_payload(wind=22.4, temp=38.2, precip=40))
    w = wc.fetch_game_weather("CHI", KICK, client=_client(handler), now=NOW)
    assert len(seen) == 1
    req = seen[0]
    assert req.url.host == "api.open-meteo.com" and req.url.path == "/v1/forecast"
    q = dict(req.url.params)
    assert q["temperature_unit"] == "fahrenheit" and q["wind_speed_unit"] == "mph" and q["timezone"] == "UTC"
    assert float(q["latitude"]) == pytest.approx(41.8623) and "wind_speed_10m" in q["hourly"]
    assert (w["is_dome"], w["wind_mph"], w["temp_f"], w["precip_prob"]) == (False, 22.4, 38.2, 40)
    assert w["summary"] == "Outdoor: 38°F | Wind: 22 mph | Precip: 40%"
    text, color, high = ui.weather_label(w)
    assert high and color == "red" and text.startswith("🌬️ Outdoor: 38°F | Wind: 22 mph")
    # an open retractable roof is outdoor too
    assert not wc.fetch_game_weather("DAL", KICK, roof="open", client=_client(handler), now=NOW)["is_dome"]


@pytest.mark.parametrize("handler", [lambda r: httpx.Response(500), lambda r: httpx.Response(200, text="nope"),
                                     lambda r: httpx.Response(200, json={"hourly": {"time": []}})])
def test_api_failure_falls_back_to_neutral(handler):
    w = wc.fetch_game_weather("BUF", KICK, client=_client(handler), now=NOW)
    assert w["available"] is False and w["wind_mph"] == 0.0 and not w["is_dome"]
    assert gs.weather_factors(w) == {"wind_pass_efficiency": 1.0, "wet_cold": 1.0, "pass_rate": 1.0}


def test_outside_horizon_neutral_site_and_pytest_guard_make_no_request():
    far = wc.fetch_game_weather("GB", NOW + timedelta(days=30), client=_client(_no_http), now=NOW)
    past = wc.fetch_game_weather("GB", NOW - timedelta(days=3), client=_client(_no_http), now=NOW)
    assert far["summary"] == past["summary"] == "Outside forecast horizon"
    assert wc.fetch_game_weather("GB", KICK, neutral_site=True, client=_client(_no_http), now=NOW)["available"] is False
    # no injected client under pytest -> never a live call
    assert wc.fetch_game_weather("GB", KICK, now=NOW)["summary"] == "Live weather disabled under pytest"


def test_kickoff_eastern_to_utc():
    assert wc.kickoff_utc("2026-12-06", "13:00") == KICK
    assert wc.kickoff_utc("2026-09-13", "20:20") == datetime(2026, 9, 14, 0, 20, tzinfo=timezone.utc)   # EDT


# ======================================================================================
# Simulator impact
# ======================================================================================
def test_weather_factors():
    assert gs.weather_factors(wc.DOME_WEATHER) == {"wind_pass_efficiency": 1.0, "wet_cold": 1.0, "pass_rate": 1.0}
    calm = {"is_dome": False, "wind_mph": 15.0, "temp_f": 50, "precip_prob": 10}
    assert gs.weather_factors(calm)["wind_pass_efficiency"] == 1.0
    f = gs.weather_factors({**calm, "wind_mph": 20.0})
    assert f["wind_pass_efficiency"] == pytest.approx(1 - 5 * 0.012) and f["pass_rate"] == pytest.approx(1 - 5 * 0.008)
    assert gs.weather_factors({**calm, "precip_prob": 60})["wet_cold"] == pytest.approx(0.97)
    assert gs.weather_factors({**calm, "temp_f": 20})["wet_cold"] == pytest.approx(0.97)
    assert gs.weather_factors({**calm, "temp_f": 20, "precip_prob": 80})["wet_cold"] == pytest.approx(0.95)


def test_wind_moves_volume_from_passing_to_rushing(con, upcoming_game):  # noqa: F811
    calm = gs.simulate_game(upcoming_game, con=con, seed=9, n_trials=6000, weather=wc.DOME_WEATHER)
    storm = gs.simulate_game(upcoming_game, con=con, seed=9, n_trials=6000,
                             weather={"is_dome": False, "wind_mph": 28.0, "temp_f": 20.0, "precip_prob": 70,
                                      "summary": "storm"})
    for team in calm.teams:
        assert storm.teams[team]["dropbacks"].mean() < calm.teams[team]["dropbacks"].mean()
        assert storm.teams[team]["carries"].mean() > calm.teams[team]["carries"].mean()
        assert storm.adjustments[team]["weather"]["pass_rate"] == pytest.approx(1 - 13 * 0.008)
    qb = next(p for p, s in calm.stats.items() if "pass_yds" in s)
    ypa = lambda r: r.get(qb, "pass_yds").mean() / r.get(qb, "pass_attempts").mean()  # noqa: E731
    expect = (1 - 13 * 0.012) * 0.95
    assert ypa(storm) == pytest.approx(ypa(calm) * expect, rel=0.06)
    dome = gs.simulate_game(upcoming_game, con=con, seed=9, n_trials=6000)
    assert dome.get(qb, "pass_yds").mean() == pytest.approx(calm.get(qb, "pass_yds").mean())  # dome = no change


# ======================================================================================
# App: weather is fetched by the Re-run button only, then shown as a badge
# ======================================================================================
def test_weather_only_loaded_by_rerun_and_shown_in_player_sim(env, no_network, monkeypatch):  # noqa: F811
    calls = []
    windy = {"is_dome": False, "temp_f": 38.0, "wind_mph": 18.0, "precip_prob": 40, "available": True,
             "summary": "Outdoor: 38°F | Wind: 18 mph | Precip: 40%"}

    def fake_slate(con, today=None):
        calls.append(1)
        return {env["game_id"]: windy}
    monkeypatch.setattr(state, "fetch_slate_weather", fake_slate)
    at = _run_app()
    at.toggle(key="exp_past").set_value(True).run()
    at.selectbox(key="exp_game").set_value(env["game_id"]).run()
    assert calls == []                                                   # reruns never fetch
    badges = [m.value for m in at.markdown]                              # st.badge renders as markdown
    assert any("forecast not loaded" in b for b in badges)
    next(b for b in at.sidebar.button if b.label == "🔄 Re-run Match Simulations").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert calls == [1] and at.session_state["weather"][env["game_id"]]["wind_mph"] == 18.0
    badges = [m.value for m in at.markdown]
    assert any("Wind: 18 mph" in b and "high wind" in b and "red" in b for b in badges)
    assert any("weather: pass efficiency" in c.value for c in at.caption)
    assert no_network == []
