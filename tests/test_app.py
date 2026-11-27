"""Tests for the Streamlit UI: pure state helpers + headless AppTest runs of app.py.

Rerun safety is checked directly: the live odds entry point is replaced by a recorder and the
HTTP layer is booby-trapped, then the app is rerun and interacted with. Only clicking the
"Pull Live Odds" button may reach the (mocked) live path.
"""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.app import state
from src.evaluation import backtest as bt
from src.evaluation import devig as dv
from src.ingestion import odds_client as oc
from tests.test_evaluation import TEAM_FULL, two_season_league
from tests.test_models import build_db

TIMEOUT = 300
APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


# ======================================================================================
# Fixture DB: two seasons + raw_odds snapshot + a backtest run
# ======================================================================================
def _odds_wrapper(ts: datetime, kick: datetime, home: str, away: str) -> dict:
    wr = f"{home} WR1"

    def book(key, ml_home, ml_away, upd):
        return {"key": key, "title": key, "last_update": upd.isoformat(), "markets": [
            {"key": "h2h", "last_update": upd.isoformat(), "outcomes": [
                {"name": TEAM_FULL[home], "price": ml_home}, {"name": TEAM_FULL[away], "price": ml_away}]},
            {"key": "spreads", "last_update": upd.isoformat(), "outcomes": [
                {"name": TEAM_FULL[home], "price": -110, "point": -2.5},
                {"name": TEAM_FULL[away], "price": -110, "point": 2.5}]},
            {"key": "totals", "last_update": upd.isoformat(), "outcomes": [
                {"name": "Over", "price": -108, "point": 41.5}, {"name": "Under", "price": -112, "point": 41.5}]},
            {"key": "player_receptions", "last_update": upd.isoformat(), "outcomes": [
                {"name": "Over", "description": wr, "price": -120, "point": 2.5},
                {"name": "Under", "description": wr, "price": -105, "point": 2.5},
                {"name": "Over", "description": "Totally Unknown Guy", "price": -110, "point": 1.5},
                {"name": "Under", "description": "Totally Unknown Guy", "price": -110, "point": 1.5}]},
            {"key": "player_receptions_alternate", "last_update": upd.isoformat(), "outcomes": [
                {"name": "Over", "description": wr, "price": 210, "point": 4.5}]},
        ]}
    books = [book("pinnacle", -150, 135, ts), book("draftkings", -145, 125, ts),
             book("stalebook", -190, 165, ts - timedelta(hours=3))]    # stale *and* off-market
    return {"fetched_at_utc": ts.isoformat(), "quota": {"remaining": 4321, "used": 179, "last": 12},
            "data": {"id": "evt_app", "home_team": TEAM_FULL[home], "away_team": TEAM_FULL[away],
                     "commence_time": kick.isoformat(), "bookmakers": books}}


@pytest.fixture(scope="module")
def app_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("app")
    path, raw = root / "app.duckdb", root / "raw_odds"
    raw.mkdir()
    con = build_db(two_season_league(), path)
    gid = con.execute("SELECT game_id FROM schedules WHERE season = 2025 AND week = 3 ORDER BY game_id LIMIT 1").fetchall()[0][0]
    home, away, gdate = con.execute("SELECT home_team, away_team, game_date FROM schedules WHERE game_id = ?",
                                    [gid]).fetchall()[0]
    kick = datetime.combine(pd.Timestamp(gdate).date(), datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=18)
    w = _odds_wrapper(kick - timedelta(hours=2), kick, home, away)
    (raw / "20250921T160000000000Z__odds__evt_app.json").write_text(__import__("json").dumps(w), encoding="utf-8")
    oc.ingest_rows(con, oc.parse_event_odds(w, "snap.json"))
    bt.run_backtest(con, bt.BacktestConfig(eval_seasons=[2025], n_trials=2000, prop_trials=1000, min_edge_pct=0.0,
                                           include_td=False))
    con.close()
    return {"db": path, "raw": raw, "game_id": gid, "home": home, "away": away, "wr": f"{home}_WR1",
            "now": kick - timedelta(hours=1)}


@pytest.fixture
def env(app_env, monkeypatch):
    monkeypatch.setenv("NFL_DB_PATH", str(app_env["db"]))
    monkeypatch.setenv("NFL_RAW_ODDS_DIR", str(app_env["raw"]))
    return app_env


@pytest.fixture
def con(env):
    with state.connect() as c:
        yield c


@pytest.fixture
def no_network(monkeypatch):
    """Record live entry points and make any real HTTP attempt fail loudly."""
    calls = []

    async def fake_run_live(api_key, **kw):
        calls.append(kw)
        return oc.RunSummary(mode="live", events_seen=1, fetched=["evt_app"], rows_ingested=0, credits_spent=0,
                             quota={"remaining": 4321, "last": 0})

    async def boom(*a, **k):
        raise AssertionError("real HTTP attempted from the app")

    monkeypatch.setattr(oc, "run_live", fake_run_live)
    monkeypatch.setattr(httpx.AsyncClient, "send", boom)
    monkeypatch.setattr(httpx.Client, "send", boom)
    monkeypatch.setenv("ODDS_API_KEY", "test-key-not-real")
    return calls


# ======================================================================================
# Pure state helpers
# ======================================================================================
def test_latest_odds_maps_game_and_filters_started(con, env):
    assert state.latest_odds(con, include_started=False).empty            # kickoff is in 2025
    o = state.latest_odds(con, include_started=False, now=env["now"])
    assert set(o["game_id"]) == {env["game_id"]}
    assert {"pinnacle", "draftkings", "stalebook"} == set(o["bookmaker_key"])
    assert "player_receptions" in set(o["base_market"]) and "player_receptions_alternate" in set(o["market_key"])


def test_price_markets_devigs_resolves_and_prices(con, env):
    priced, diag = state.price_markets(con, n_trials=3000, include_started=True)
    assert {"h2h", "spreads", "totals", "player_receptions", "player_receptions_alternate"} <= set(priced["market_key"])
    assert diag["unmatched_players"] == ["Totally Unknown Guy"]          # never guessed (rule 4)
    props = priced[priced["market_key"].str.startswith("player_")]
    assert set(props["gsis_id"]) == {env["wr"]}
    assert priced["model_prob"].between(0, 1, inclusive="neither").all()
    # fair probs come from the sharp book and sum to one within each two-way group (rule 5)
    two_way = priced[priced["market_key"].isin(["h2h", "spreads", "totals", "player_receptions"])]
    assert (two_way["fair_source"] == "pinnacle").all()
    for mk, g in two_way[two_way["selection"] != "Totally Unknown Guy"].groupby("market_key"):
        assert len(g) == 2 and g["fair_prob"].sum() == pytest.approx(1.0), mk
    ml = priced[priced["market_key"] == "h2h"].set_index("selection")
    assert ml.loc[env["home"], "fair_prob"] == pytest.approx(dv.devig_american([-150, 135], "shin")[0])
    assert ml.loc[env["home"], "best_book"] == "draftkings"               # best price shopped
    alt = priced[priced["market_key"] == "player_receptions_alternate"].iloc[0]
    assert alt["fair_source"] == "pinnacle" and alt["devig_method"] == "one_way_assumed_margin"   # Over-only quote


def test_sharp_book_selector_changes_reference(con):
    auto, _ = state.price_markets(con, n_trials=1000, include_started=True)
    dk, _ = state.price_markets(con, sharp_priority=state.Settings(sharp_book="draftkings").sharp_priority(),
                                n_trials=1000, include_started=True)
    assert (auto.loc[auto.market_key == "h2h", "fair_source"] == "pinnacle").all()
    assert (dk.loc[dk.market_key == "h2h", "fair_source"] == "draftkings").all()


def test_apply_bankroll_zero_stake_for_negative_ev_and_threshold(con):
    priced, _ = state.price_markets(con, n_trials=2000, include_started=True)
    ev = state.apply_bankroll(priced, state.Settings(bankroll=5000, kelly=0.5, min_edge_pct=1.0))
    assert (ev.loc[ev["ev_pct"] <= 0, "kelly_stake"] == 0).all()
    assert (ev.loc[ev["is_bet"], "edge_pct"] >= 1.0).all() and (ev.loc[ev["is_bet"], "ev_pct"] > 0).all()
    assert (ev["kelly_stake"] <= 5000 * dv.DEFAULT_MAX_STAKE_PCT + 1e-9).all()
    hi = state.apply_bankroll(priced, state.Settings(min_edge_pct=99.0))
    assert not hi["is_bet"].any()
    table = state.scanner_table(ev, positive_only=False)
    assert list(table.columns)[:10] == ["Game", "Player/Team", "Market", "Side", "Line", "Best Book", "Best Odds",
                                        "Sharp No-Vig Fair", "Model Fair", "Edge %"]
    assert "Kelly Stake $" in table.columns
    assert table["Edge %"].is_monotonic_decreasing
    assert state.scanner_table(ev, categories=["moneyline"], positive_only=False)["Market"].eq("h2h").all()
    assert not state.scanner_table(ev, positive_only=False, include_alternates=False)["Market"].str.contains("alternate").any()


def test_line_shop_matrix_flags_best_stale_off(con):
    odds = state.latest_odds(con, include_started=True)
    price, flags, detail = state.line_shop_matrix(odds, "evt_app", "h2h", stale_minutes=30, off_market_pp=3)
    assert list(price.columns) == ["Fair (no-vig)", "pinnacle", "draftkings", "stalebook"]   # sharp first
    home_row = [i for i in price.index if "Packers" in i or "Lions" in i or "Bears" in i or "Vikings" in i]
    assert len(price) == 2 and len(home_row) == 2
    assert flags["stalebook"].str.contains("stale").all()
    assert flags["stalebook"].str.contains("off").all()                  # -190/+165 vs pinnacle -150/+135
    assert not flags["pinnacle"].str.contains("stale|off").any()
    assert (flags.drop(columns="Fair (no-vig)").apply(lambda c: c.str.contains("best")).sum(axis=1) >= 1).all()
    fair_am = price["Fair (no-vig)"].to_numpy(float)
    p = 1 / dv.american_to_decimal(fair_am)
    assert p.sum() == pytest.approx(1.0, abs=1e-6)
    assert detail["lag_min"].max() == pytest.approx(180)


def test_player_lines_and_line_probabilities(con, env):
    odds = state.latest_odds(con, include_started=True)
    lines = state.player_book_lines(odds, env["game_id"], env["wr"], "receptions", state.make_resolver(con))
    assert set(lines["line"]) == {2.5, 4.5}
    assert lines.loc[lines["line"] == 4.5, "is_alternate"].all()
    assert set(lines["bookmaker_key"]) == {"pinnacle", "draftkings", "stalebook"}
    x = np.array([0, 1, 2, 3, 3, 4, 5, 6] * 100, float)
    pr = state.line_probabilities(x, [2.5, 3.0])
    r3 = pr.set_index("line").loc[3.0]
    assert r3["p_push"] == pytest.approx(0.25) and r3["p_over"] + r3["p_under"] + r3["p_push"] == pytest.approx(1, abs=1e-3)


def test_usage_overrides_snap_scaling():
    pl = pd.Series({"gsis_id": "X", "target_share": 0.20, "carry_share": 0.05, "snap_share_ewm5": 0.80})
    assert state.usage_overrides(pl, 0.20, 0.05, 0.80) == {}
    assert state.usage_overrides(pl, 0.25, 0.05, 0.80) == {"X": {"target_share": pytest.approx(0.25)}}
    ov = state.usage_overrides(pl, 0.20, 0.05, 0.40)["X"]                # half the snaps -> half the shares
    assert ov["target_share"] == pytest.approx(0.10) and ov["carry_share"] == pytest.approx(0.025)


def test_override_changes_simulation(con, env):
    players = state.gs.load_players(con, env["game_id"])
    p = players.set_index("gsis_id").loc[env["wr"]].copy()
    p["gsis_id"] = env["wr"]
    base = state.gs.simulate_game(env["game_id"], con=con, n_trials=4000, seed=3)
    ov = state.usage_overrides(p, min(0.5, float(p["target_share"]) * 2), float(p["carry_share"]), float(p["snap_share_ewm5"]))
    bumped = state.gs.simulate_game(env["game_id"], ov, con=con, n_trials=4000, seed=3)
    assert bumped.get(env["wr"], "targets").mean() > 1.3 * base.get(env["wr"], "targets").mean()


def test_backtest_helpers(con):
    runs = state.backtest_runs(con)
    assert len(runs) == 1
    res, bets = state.load_backtest(con, runs["run_id"].iloc[0])
    bankroll = state.run_bankroll(res)
    assert bankroll == dv.DEFAULT_BANKROLL
    curve = state.equity_curve(bets, bankroll)
    tot = curve[curve["series"] == "All markets"]
    assert tot["bankroll"].iloc[-1] == pytest.approx(bankroll + bets.loc[bets["is_bet"], "profit"].sum())
    assert tot["game_date"].is_monotonic_increasing
    cal = state.calibration_table(bets, n_bins=5)
    assert set(cal["source"]) == {"Model", "Market (devigged)"}
    assert cal["observed"].between(0, 1).all()
    assert cal.groupby("source")["n"].sum().nunique() == 1
    clv = state.clv_summary(bets)
    raw = bets[bets["is_bet"] & bets["clv_pct"].notna()]
    assert clv["n_bets"].sum() == len(raw)
    roi = state.roi_by_market(res)
    assert (roi["n_bets"] > 0).all()


def test_estimate_cost_and_quota_from_cache(env):
    assert state.estimate_pull_cost(["main"], ["us"], 16) == 16 * 3
    assert state.estimate_pull_cost(["main", "player_props"], ["us", "eu"], 2) == 2 * (3 + len(oc.PLAYER_PROP_MARKETS)) * 2
    assert state.estimate_pull_cost([], ["us"], 16) == 0
    q = state.last_known_quota()
    assert q["remaining"] == 4321 and q["n_files"] == 1


# ======================================================================================
# app.py: import + headless runs
# ======================================================================================
def test_app_imports_cleanly_without_side_effects(no_network):
    import app

    importlib.reload(app)
    assert callable(app.main) and no_network == []


def _run_app(**session) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    for k, v in session.items():
        at.session_state[k] = v
    return at.run()


def test_app_renders_all_tabs_without_live_calls(env, no_network):
    at = _run_app()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == ["📈 +EV Scanner", "🎲 Player Sim & Alt Lines", "🛒 Line Shopper",
                                          "📊 Backtest & CLV"]
    assert at.sidebar.metric[0].value == "4,321"                         # remaining credits, read from cache
    pull = next(b for b in at.sidebar.button if b.label == "Pull Live Odds")
    assert pull.disabled                                                  # needs explicit approval first
    # interact with many widgets -> many reruns, still no live call
    at.sidebar.slider(key="kelly").set_value(0.5).run()
    at.sidebar.slider(key="min_edge_pct").set_value(0.0).run()
    at.sidebar.selectbox(key="sharp_book").set_value("draftkings").run()
    at.toggle(key="scan_started").set_value(True).run()
    at.toggle(key="exp_past").set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    assert no_network == []
    assert st_live_calls(at) == 0


def st_live_calls(at: AppTest) -> int:
    return at.session_state["live_calls"] if "live_calls" in at.session_state else 0


def test_scanner_shopper_explorer_and_backtest_content(env, no_network):
    at = _run_app(min_edge_pct=0.0)
    at.toggle(key="scan_started").set_value(True).run()
    at.number_input(key="scan_max_edge").set_value(100.0).run()       # synthetic edges are extreme
    at.number_input(key="scan_min_games").set_value(0).run()
    at.toggle(key="shop_started").set_value(True).run()
    at.toggle(key="exp_past").set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    scanner = at.dataframe[0].value
    assert {"Game", "Player/Team", "Best Book", "Sharp No-Vig Fair", "Model Fair", "Edge %", "Kelly Stake $"} <= set(scanner.columns)
    assert len(scanner) > 0 and (scanner["Kelly Stake $"] > 0).all()
    assert any(b.label.startswith("⬇ Export CSV") for b in at.get("download_button"))
    labels = [m.label for m in at.metric]
    assert {"+EV bets", "Off-market quotes", "Stale quotes", "Bets", "ROI", "Avg CLV"} <= set(labels)
    # explorer: choose the WR with book lines -> histogram + book line table
    at.selectbox(key="exp_game").set_value(env["game_id"]).run()
    at.selectbox(key="exp_player").set_value(env["wr"]).run()
    at.selectbox(key=f"exp_stat_{env['wr']}").set_value("receptions").run()
    assert not at.exception, [e.value for e in at.exception]
    assert "P(over 2.5)" in [m.label for m in at.metric]
    before = next(m for m in at.metric if m.label == "Mean").value
    k = f"{env['game_id']}_{env['wr']}"
    at.slider(key=f"ts_{k}").set_value(0.45).run()
    assert not at.exception, [e.value for e in at.exception]
    after = next(m for m in at.metric if m.label == "Mean").value
    assert float(after) > float(before)                                    # live re-run with the override
    assert no_network == []


def test_pull_live_odds_only_via_button(env, no_network):
    at = _run_app()
    at.sidebar.multiselect(key="pull_groups").set_value(["main"]).run()
    at.sidebar.number_input(key="pull_max_events").set_value(4).run()
    assert no_network == []
    at.sidebar.checkbox(key="pull_confirm").check().run()
    pull = next(b for b in at.sidebar.button if b.label == "Pull Live Odds")
    assert not pull.disabled and no_network == []
    pull.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(no_network) == 1
    call = no_network[0]
    assert call["markets"] == list(oc.MAIN_MARKETS) and call["max_events"] == 4
    assert call["max_credits"] == 4 * 3                                     # spend capped at the shown estimate
    assert at.session_state["last_pull"]["fetched"] == 1
    at.sidebar.slider(key="kelly").set_value(0.3).run()                     # later reruns don't re-pull
    assert len(no_network) == 1


def test_live_pull_blocked_without_api_key(env, no_network, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "")
    monkeypatch.setattr(state.oc, "load_dotenv", lambda *a, **k: False, raising=False)
    at = _run_app()
    at.sidebar.checkbox(key="pull_confirm").check().run()
    assert next(b for b in at.sidebar.button if b.label == "Pull Live Odds").disabled
    assert any("ODDS_API_KEY" in w.value for w in at.sidebar.warning)


def test_app_handles_missing_database(tmp_path, monkeypatch, no_network):
    monkeypatch.setenv("NFL_DB_PATH", str(tmp_path / "nope.duckdb"))
    at = _run_app()
    assert not at.exception and any("Database not found" in e.value for e in at.error)


def _evaluated() -> pd.DataFrame:
    return pd.DataFrame({
        "game": "A @ B", "market_category": ["player_receiving", "player_receiving", "spread", "player_receiving"],
        "market_key": ["player_receptions", "player_receptions", "spreads", "player_reception_yds"],
        "selection": ["Rookie", "Vet", "B", "Vet"], "side": ["Under", "Over", "", "Over"],
        "line": [2.5, 5.5, -3, 60.5], "best_book": "dk", "best_american": [-110, 120, -105, 110],
        "fair_american": [100, 105, -100, 102], "model_american": [-800, -110, -150, -105],
        "edge_pct": [38.0, 4.0, 9.0, 16.0], "ev_pct": [60.0, 8.0, 15, 20], "kelly_stake": [500.0, 120, 300, 400],
        "is_bet": True, "is_alternate": False, "fair_source": "pinnacle", "n_books": 3,
        "hist_games": [1.0, 40, np.nan, 40]})


def test_scanner_max_edge_and_min_games_filters():
    ev = _evaluated()
    assert len(state.filter_evaluated(ev)) == 4                                       # filters off
    capped = state.filter_evaluated(ev, max_edge_pct=15.0)
    assert set(capped["selection"]) == {"Vet", "B"} and (capped["edge_pct"] <= 15).all()
    thin = state.filter_evaluated(ev, min_hist_games=3)
    assert "Rookie" not in set(thin["selection"]) and "B" in set(thin["selection"])  # game markets unaffected
    both = state.scanner_table(ev, max_edge_pct=state.DEFAULT_MAX_EDGE_PCT,
                               min_hist_games=state.DEFAULT_MIN_HIST_GAMES)
    assert both["Player/Team"].tolist() == ["B", "Vet"] and "Hist. Games" in both.columns
    assert state.DEFAULT_MAX_EDGE_PCT == 15.0


def test_priced_rows_carry_player_history(con, env):
    priced, _ = state.price_markets(con, n_trials=1000, include_started=True)
    props = priced[priced["market_key"].str.startswith("player_")]
    assert props["hist_games"].notna().all() and (props["hist_games"] >= 1).all()
    assert priced.loc[~priced["market_key"].str.startswith("player_"), "hist_games"].isna().all()


def test_app_default_filters_hide_extreme_edges(env, no_network):
    at = _run_app(min_edge_pct=0.0)
    at.toggle(key="scan_started").set_value(True).run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.number_input(key="scan_max_edge").value == 15.0
    assert at.number_input(key="scan_min_games").value == 5
    assert at.number_input(key="scan_min_books").value == 3
    assert at.sidebar.slider(key="min_edge_pct").value == 0.0  # set by the test; default is 2.5
    table = at.dataframe[0].value
    assert table.empty or (table["Edge %"] <= 15.0).all()
    assert any("Hidden by filters" in c.value for c in at.caption)


# ======================================================================================
# Injury & depth chart overrides, re-run button, defaults
# ======================================================================================
def test_depth_spec_is_canonical_and_reactivates_cleared_players():
    a = state.depth_spec(["b", "a"], ["a", "c"], {"PHI": "qb2", "DAL": ""}, 0.85, 1.0)
    b = state.depth_spec(["a", "b", "a"], ["c", "a"], {"PHI": "qb2"}, 0.85, 1.0)
    assert a == b and state.depth_json(a) == state.depth_json(b)
    assert a["inactive"] == ["a", "b"] and a["reactivate"] == ["c"] and a["starting_qbs"] == {"PHI": "qb2"}
    assert state.manual_depth_ids(a) == {"qb2", "c"}


def test_min_books_filter_and_manual_players_skip_history_filter():
    ev = _evaluated().assign(n_books=[3, 2, 5, 3], manual_depth=[True, False, False, False])
    d = state.filter_evaluated(ev, min_hist_games=5, min_books=3)
    assert set(d["selection"]) == {"Rookie", "B", "Vet"} and len(d) == 3     # Rookie kept: activated by hand
    assert (d["n_books"] >= 3).all()
    assert (state.DEFAULT_MIN_EDGE_PCT, state.DEFAULT_MAX_EDGE_PCT, state.DEFAULT_MIN_HIST_GAMES,
            state.DEFAULT_MIN_BOOKS) == (2.5, 15.0, 5, 3)
    assert state.Settings().min_edge_pct == 2.5 and state.SESSION_DEFAULTS["min_edge_pct"] == 2.5


def test_thin_history_props_blend_toward_market():
    out = pd.DataFrame({"market_key": ["player_receptions", "player_receptions", "player_receptions", "spreads"],
                        "hist_games": [0.0, 2.0, 10.0, np.nan], "model_prob": [0.05, 0.05, 0.05, 0.05],
                        "fair_prob": [0.5, 0.5, 0.5, 0.5], "model_p_push": [0.0, 0.0, 0.0, 0.1]})
    b = state.blend_thin_history(out)
    w0 = state.MARKET_BLEND_MIN_WEIGHT
    assert b["model_prob"].iloc[0] == pytest.approx(w0 * 0.05 + (1 - w0) * 0.5)      # rookie: mostly market
    assert b["model_prob"].iloc[0] > b["model_prob"].iloc[1] > b["model_prob"].iloc[2] == pytest.approx(0.05)
    assert b["model_prob"].iloc[3] == pytest.approx(0.05)                             # game market untouched
    assert b["model_p_win"].iloc[0] == pytest.approx(b["model_prob"].iloc[0])


def test_price_markets_skips_inactive_players(con, env):
    base, _ = state.price_markets(con, n_trials=1000, include_started=True, depth={})
    out, diag = state.price_markets(con, n_trials=1000, include_started=True,
                                    depth=state.depth_spec([env["wr"]], [], {}, 0.85, 1.0))
    assert (base["gsis_id"] == env["wr"]).any() and not (out["gsis_id"] == env["wr"]).any()
    assert diag["inactive_players"] > 0
    assert len(out[~out["market_key"].str.startswith("player_")]) == len(base[~base["market_key"].str.startswith("player_")])


@pytest.fixture
def slate(env, monkeypatch):
    """Make the fixture's past game the 'upcoming slate' of the depth panel."""
    gdate = pd.Timestamp(pd.Timestamp(env["now"]).date())
    real = state.depth_players
    monkeypatch.setattr(state, "depth_players", lambda con, today=None, days=8, unplayed_only=True:
                        real(con, gdate, days, unplayed_only=False))
    state.cached_depth_players.clear()
    yield env
    state.cached_depth_players.clear()


def test_depth_panel_stages_and_rerun_applies(slate, no_network):
    at = _run_app(min_edge_pct=0.0)
    assert not at.exception, [e.value for e in at.exception]
    ms = at.sidebar.multiselect(key="depth_inactive")
    assert any(o.startswith(f"{slate['home']} WR1 (WR, {slate['home']})") for o in ms.options)
    assert at.session_state["depth_applied"]["inactive"] == list(ms.value)   # injury report applied on load
    ms.set_value([slate["wr"]]).run()
    assert any("re-run the simulations" in w.value for w in at.sidebar.warning)
    assert slate["wr"] not in at.session_state["depth_applied"]["inactive"]  # staged, not applied yet
    runs = at.session_state["sim_runs"]
    next(b for b in at.sidebar.button if b.label == "🔄 Re-run Match Simulations").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["depth_applied"]["inactive"] == [slate["wr"]]
    assert at.session_state["sim_runs"] == runs + 1
    assert not any("re-run the simulations" in w.value for w in at.sidebar.warning)
    # player sim: the ruled-out WR stays selectable but explains how to bring him back
    at.toggle(key="exp_past").set_value(True).run()
    at.selectbox(key="exp_game").set_value(slate["game_id"]).run()
    at.selectbox(key="exp_player").set_value(slate["wr"]).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("Inactive / Out" in i.value for i in at.info)
    assert no_network == []


def test_starting_qb_override_widget(slate, no_network):
    at = _run_app()
    team = at.sidebar.selectbox(key="qb_team").value
    pick = at.sidebar.selectbox(key=f"qb_pick_{team}")
    assert len(pick.options) == 2                                         # Auto + the team's QB
    qb = f"{team}_QB"
    pick.set_value(qb).run()
    assert at.session_state["qb_staged"] == {team: qb}
    next(b for b in at.sidebar.button if b.label == "🔄 Re-run Match Simulations").click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["depth_applied"]["starting_qbs"] == {team: qb}
    at.sidebar.button(key="qb_clear").click().run()
    assert at.session_state["qb_staged"] == {}
    assert no_network == []
