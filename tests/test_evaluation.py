"""Tests for src/evaluation/devig.py and src/evaluation/backtest.py."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.evaluation import backtest as bt
from src.evaluation import devig as dv
from src.ingestion import odds_client as oc
from tests.test_features import LEAGUE, UPCOMING_WEEK
from tests.test_models import build_db, enrich

RNG = np.random.default_rng(123)


# ======================================================================================
# Devig: probabilities sum to 1
# ======================================================================================
def _random_market(n: int, margin: float) -> np.ndarray:
    p = 0.5 * RNG.dirichlet(np.ones(n) * 2) + 0.5 / n  # max p <= 0.75 -> p*(1+margin) < 1, as real books must
    return 1.0 / (p * (1 + margin))  # decimal odds with the given overround


@pytest.mark.parametrize("method", ["multiplicative", "shin"])
@pytest.mark.parametrize("n_outcomes", [2, 3, 5, 12, 30])
@pytest.mark.parametrize("margin", [0.0, 0.02, 0.045, 0.10, 0.25])
def test_devigged_probabilities_sum_to_one(method, n_outcomes, margin):
    for _ in range(20):
        odds = _random_market(n_outcomes, margin)
        p = dv.devig(odds, method)
        assert p.shape == (n_outcomes,)
        assert abs(p.sum() - 1.0) < 1e-9
        assert np.all(p > 0) and np.all(p < 1)


@pytest.mark.parametrize("method", ["multiplicative", "shin"])
def test_two_way_american_examples(method):
    p = dv.devig_american([-110, -110], method)
    assert np.allclose(p, [0.5, 0.5])
    p = dv.devig_american([-200, 170], method)
    assert p.sum() == pytest.approx(1.0) and p[0] > 0.6
    assert dv.sum_to_one([dv.devig_american(x, method) for x in ([-150, 130], [+250, -320], [-105, -115])])


def test_multiplicative_formula():
    d = np.array([1.8, 2.1])
    q = 1 / d
    assert np.allclose(dv.devig_multiplicative(d), q / q.sum())


def test_shin_removes_more_margin_from_longshots():
    odds = dv.american_to_decimal([-450, 350])
    mult, shin = dv.devig_multiplicative(odds), dv.devig_shin(odds)
    assert shin[1] < mult[1] and shin[0] > mult[0]       # favourite-longshot bias correction
    assert 0 < dv.shin_z(odds) < 0.2


def test_shin_without_vig_is_normalisation():
    odds = 1 / np.array([0.25, 0.25, 0.5])
    assert dv.shin_z(odds) == 0.0
    assert np.allclose(dv.devig_shin(odds), [0.25, 0.25, 0.5])


@pytest.mark.parametrize("bad", [[2.0], [1.0, 2.0], [0.5, 3.0], [np.nan, 2.0]])
def test_devig_rejects_invalid_markets(bad):
    with pytest.raises(ValueError):
        dv.devig(bad)


def test_odds_conversion_roundtrip():
    am = np.array([-500, -110, 100, 150, 1200])
    assert np.allclose(dv.decimal_to_american(dv.american_to_decimal(am)), am)
    assert dv.overround(dv.american_to_decimal([-110, -110])) == pytest.approx(1.0476, abs=1e-4)


# ======================================================================================
# EV, edge, Kelly
# ======================================================================================
def test_ev_and_edge():
    assert dv.ev_pct(0.55, 2.0) == pytest.approx(10.0)
    assert dv.ev_pct(0.5, 1.909) == pytest.approx(-4.55, abs=0.01)
    assert dv.edge_pct(0.56, 0.52) == pytest.approx(4.0)
    assert dv.ev_pct(0.45, 2.0, p_push=0.10) == pytest.approx(0.0)  # 0.45 win, 0.45 loss, 0.10 refund


@pytest.mark.parametrize("p,dec", [(0.40, 2.0), (0.50, 1.909), (0.20, 4.5), (0.0, 3.0), (0.01, 50.0), (0.5, 2.0)])
def test_kelly_zero_stake_for_non_positive_ev(p, dec):
    assert dv.expected_value(p, dec) <= 1e-12
    assert dv.kelly_fraction(p, dec) == 0.0
    stake = dv.kelly_stake(p, dec, bankroll=10_000, fraction=0.25)
    assert stake == 0.0 and not np.signbit(stake)


def test_kelly_vectorised_negative_ev_entries_are_zero():
    p = np.array([0.30, 0.55, 0.45, 0.70])
    d = np.array([2.0, 2.0, 2.0, 1.5])
    stakes = dv.kelly_stake(p, d, bankroll=1000, fraction=0.25, max_stake_pct=None)
    assert stakes[0] == 0.0 and stakes[2] == 0.0
    assert stakes[1] == pytest.approx(1000 * 0.25 * 0.10)          # f* = (1*0.55 - 0.45)/1 = 0.10
    assert stakes[3] == pytest.approx(1000 * 0.25 * (0.5 * 0.7 - 0.3) / 0.5)


def test_kelly_formula_fraction_bankroll_and_cap():
    f = dv.kelly_fraction(0.60, 2.2)                                 # b = 1.2
    assert f == pytest.approx((1.2 * 0.6 - 0.4) / 1.2)
    assert dv.kelly_stake(0.60, 2.2, bankroll=5000, fraction=0.25, max_stake_pct=None) == pytest.approx(5000 * 0.25 * f)
    assert dv.kelly_stake(0.60, 2.2, bankroll=5000, fraction=1.0, max_stake_pct=None) == pytest.approx(5000 * f)
    assert dv.kelly_stake(0.90, 3.0, bankroll=5000, fraction=0.25, max_stake_pct=0.05) == pytest.approx(250.0)
    assert dv.DEFAULT_KELLY_FRACTION == 0.25
    with pytest.raises(ValueError):
        dv.kelly_stake(0.6, 2.2, bankroll=-1)


def test_kelly_with_push_probability():
    # 0.45 win / 0.45 loss / 0.10 push at even money -> zero edge -> no bet
    assert dv.kelly_fraction(0.45, 2.0, p_push=0.10) == 0.0
    assert dv.kelly_fraction(0.50, 2.0, p_push=0.10) == pytest.approx((0.5 / 0.9) - (0.4 / 0.9))


# ======================================================================================
# raw_odds -> fair probabilities
# ======================================================================================
def _rows(book, market, outcomes, event="e1", home="Detroit Lions", away="Green Bay Packers"):
    return [{"event_id": event, "bookmaker_key": book, "market_key": market, "outcome_name": n,
             "outcome_description": desc, "outcome_point": pt, "price_american": price,
             "price_decimal": float(dv.american_to_decimal(price)), "home_team": home, "away_team": away}
            for n, desc, pt, price in outcomes]


def test_sharp_book_priority_over_consensus():
    df = pd.DataFrame(
        _rows("draftkings", "h2h", [("Detroit Lions", None, None, -170), ("Green Bay Packers", None, None, 145)]) +
        _rows("fanduel", "h2h", [("Detroit Lions", None, None, -160), ("Green Bay Packers", None, None, 135)]) +
        _rows("pinnacle", "h2h", [("Detroit Lions", None, None, -150), ("Green Bay Packers", None, None, 138)]))
    f = dv.fair_probabilities(df, "multiplicative").set_index("outcome_name")
    pin = dv.devig_american([-150, 138], "multiplicative")
    assert (f["fair_source"] == "pinnacle").all()
    assert f.loc["Detroit Lions", "fair_prob"] == pytest.approx(pin[0])
    assert f["fair_prob"].sum() == pytest.approx(1.0)
    assert f.loc["Green Bay Packers", "best_book"] == "draftkings"     # line shopping: best price
    assert f.loc["Detroit Lions", "best_book"] == "pinnacle"
    assert f.loc["Detroit Lions", "n_books"] == 3


def test_consensus_fallback_when_no_sharp_book():
    df = pd.DataFrame(
        _rows("draftkings", "totals", [("Over", None, 44.5, -115), ("Under", None, 44.5, -105)]) +
        _rows("fanduel", "totals", [("Over", None, 44.5, -105), ("Under", None, 44.5, -115)]))
    f = dv.fair_probabilities(df, "shin")
    assert (f["fair_source"] == "consensus").all()
    assert f["fair_prob"].sum() == pytest.approx(1.0)
    assert f["fair_prob"].to_numpy() == pytest.approx([0.5, 0.5], abs=1e-9)


def test_sharp_book_must_quote_whole_group():
    df = pd.DataFrame(
        _rows("pinnacle", "totals", [("Over", None, 44.5, -110)]) +
        _rows("draftkings", "totals", [("Over", None, 44.5, -110), ("Under", None, 44.5, -110)]))
    f = dv.fair_probabilities(df)
    assert (f["fair_source"] == "consensus").all() and f["fair_prob"].sum() == pytest.approx(1.0)


def test_spread_sides_pair_and_alt_lines_separate():
    df = pd.DataFrame(_rows("pinnacle", "spreads", [
        ("Detroit Lions", None, -3.5, -105), ("Green Bay Packers", None, 3.5, -115),
        ("Detroit Lions", None, 3.5, -400), ("Green Bay Packers", None, -3.5, 300)]))
    f = dv.fair_probabilities(df, "multiplicative")
    assert f["group_key"].nunique() == 2
    for _, g in f.groupby("group_key"):
        assert len(g) == 2 and g["fair_prob"].sum() == pytest.approx(1.0)
    main = f[(f.outcome_name == "Detroit Lions") & (f.outcome_point == -3.5)]["fair_prob"].iloc[0]
    assert main == pytest.approx(dv.devig_american([-105, -115], "multiplicative")[0])


def test_player_ou_groups_by_player_and_line():
    df = pd.DataFrame(_rows("pinnacle", "player_receptions", [
        ("Over", "Amon-Ra St. Brown", 6.5, -120), ("Under", "Amon-Ra St. Brown", 6.5, 100),
        ("Over", "Amon-Ra St. Brown", 7.5, 130), ("Under", "Amon-Ra St. Brown", 7.5, -155),
        ("Over", "Sam LaPorta", 4.5, -110), ("Under", "Sam LaPorta", 4.5, -110)]))
    f = dv.fair_probabilities(df)
    assert f["group_key"].nunique() == 3
    assert dv.sum_to_one(g["fair_prob"].to_numpy() for _, g in f.groupby("group_key"))


def test_first_td_multiway_and_anytime_one_way():
    first = pd.DataFrame(_rows("draftkings", "player_1st_td", [
        ("Yes", "A", None, 450), ("Yes", "B", None, 550), ("Yes", "C", None, 700), ("Yes", "D", None, 900),
        ("Yes", "E", None, 1200), ("Yes", "F", None, 2000)]))
    f = dv.fair_probabilities(first)
    assert f["group_key"].nunique() == 1 and f["fair_prob"].sum() == pytest.approx(1.0)
    anytime = pd.DataFrame(_rows("draftkings", "player_anytime_td", [("Yes", "A", None, -120), ("Yes", "B", None, 150)]))
    a = dv.fair_probabilities(anytime).set_index("outcome_description")
    assert (a["devig_method"] == "one_way_assumed_margin").all()
    assert a.loc["A", "fair_prob"] == pytest.approx(float(1 / dv.american_to_decimal(-120)) / (1 + dv.DEFAULT_ONE_WAY_MARGIN))
    both = pd.DataFrame(_rows("pinnacle", "player_anytime_td", [("Yes", "A", None, -120), ("No", "A", None, 100)]))
    assert dv.fair_probabilities(both)["fair_prob"].sum() == pytest.approx(1.0)


def test_evaluate_selections_columns_and_zero_stake_for_negative_ev():
    fair = pd.DataFrame({"fair_prob": [0.50, 0.50], "best_price_decimal": [2.10, 1.80], "model_prob": [0.56, 0.50]})
    out = dv.evaluate_selections(fair, bankroll=1000)
    assert {"ev_pct", "edge_pct", "market_ev_pct", "kelly_full", "stake"} <= set(out.columns)
    assert out.loc[0, "ev_pct"] == pytest.approx(17.6) and out.loc[0, "edge_pct"] == pytest.approx(6.0)
    assert out.loc[0, "stake"] > 0 and out.loc[1, "stake"] == 0.0
    assert dv.clv_pct(2.10, 0.50) == pytest.approx(5.0)


# ======================================================================================
# Backtest (synthetic two-season league)
# ======================================================================================
def two_season_league() -> dict[str, pd.DataFrame]:
    base = enrich(LEAGUE)
    prev = {}
    for name, df in base.items():
        d = df.copy()
        if "game_id" in d:
            d["game_id"] = d["game_id"].str.replace("2025_", "2024_", n=1)
        if "season" in d:
            d["season"] = 2024
        for col in ("game_date",):
            if col in d:
                d[col] = pd.to_datetime(d[col]) - pd.Timedelta(days=364)
        if "gameday" in d:
            d["gameday"] = d["game_date"].dt.strftime("%Y-%m-%d")
        if "week" in d and name != "rosters":
            d = d[d["week"] < UPCOMING_WEEK]
        prev[name] = d
    out = {}
    for name in base:
        if name in ("rosters",):
            out[name] = pd.concat([prev[name], base[name]], ignore_index=True)
        elif name == "players":
            out[name] = base[name]
        else:
            cur = base[name].copy()
            if "game_date" in cur:
                cur["game_date"] = pd.to_datetime(cur["game_date"])
            out[name] = pd.concat([prev[name], cur], ignore_index=True)
    out["player_weekly"]["season"] = out["player_weekly"]["game_id"].str[:4].astype(int)
    s = out["schedules"]
    s["game_date"] = pd.to_datetime(s["game_date"]).dt.date
    for k in ("snap_counts", "player_weekly"):
        if "game_date" in out[k]:
            out[k]["game_date"] = pd.to_datetime(out[k]["game_date"]).dt.date
    fav_home = s["spread_line"] > 0
    s["home_moneyline"] = np.where(fav_home, -100 - 15 * s["spread_line"].abs(), 100 + 12 * s["spread_line"].abs())
    s["away_moneyline"] = np.where(fav_home, 100 + 12 * s["spread_line"].abs(), -100 - 15 * s["spread_line"].abs())
    s.loc[s["spread_line"] == 0, ["home_moneyline", "away_moneyline"]] = -110
    s["home_spread_odds"], s["away_spread_odds"], s["over_odds"], s["under_odds"] = -110, -110, -110, -110
    return out


TEAM_FULL = {"DET": "Detroit Lions", "GB": "Green Bay Packers", "CHI": "Chicago Bears", "MIN": "Minnesota Vikings"}


def add_raw_odds(con, game_id: str, game_date) -> dict:
    home, away = con.execute("SELECT home_team, away_team FROM schedules WHERE game_id = ?", [game_id]).fetchall()[0]
    kick = datetime.combine(pd.Timestamp(game_date).date(), datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=18)
    wr = f"{home} WR1"  # roster full_name of the home WR1 in the synthetic league
    def wrapper(ts, pin_home, dk_home, rec_over):
        books = []
        for key, h2h_home in (("pinnacle", pin_home), ("draftkings", dk_home)):
            books.append({"key": key, "title": key, "markets": [
                {"key": "h2h", "outcomes": [{"name": TEAM_FULL[home], "price": h2h_home},
                                            {"name": TEAM_FULL[away], "price": -h2h_home if h2h_home > 0 else 100 + (-h2h_home - 100) * 0.8}]},
                {"key": "spreads", "outcomes": [{"name": TEAM_FULL[home], "price": -110, "point": -2.5},
                                                {"name": TEAM_FULL[away], "price": -110, "point": 2.5}]},
                {"key": "totals", "outcomes": [{"name": "Over", "price": -108, "point": 41.5},
                                               {"name": "Under", "price": -112, "point": 41.5}]},
                {"key": "player_receptions", "outcomes": [
                    {"name": "Over", "description": wr, "price": rec_over, "point": 2.5},
                    {"name": "Under", "description": wr, "price": -rec_over - 20 if rec_over > 0 else -100 - rec_over // 2, "point": 2.5},
                    {"name": "Over", "description": "Totally Unknown Guy", "price": -110, "point": 1.5},
                    {"name": "Under", "description": "Totally Unknown Guy", "price": -110, "point": 1.5}]},
            ]})
        return {"fetched_at_utc": ts.isoformat(), "request_params": {"oddsFormat": "american"},
                "data": {"id": "evt_bt", "home_team": TEAM_FULL[home], "away_team": TEAM_FULL[away],
                         "commence_time": kick.isoformat(), "bookmakers": books}}
    open_ts, close_ts = kick - timedelta(days=3), kick - timedelta(hours=1)
    rows = oc.parse_event_odds(wrapper(open_ts, -130, -125, -105), "open.json") + \
        oc.parse_event_odds(wrapper(close_ts, -160, -150, -125), "close.json")
    oc.ingest_rows(con, rows)
    return {"home": home, "away": away, "open": open_ts, "close": close_ts}


BT_INFO: dict = {}
CFG = dict(eval_seasons=[2025], n_trials=4000, prop_trials=2000, min_edge_pct=0.0,
           td_params={"min_data_in_leaf": 5, "num_leaves": 4, "bagging_fraction": 1.0, "bagging_freq": 0},
           td_rounds=20)


@pytest.fixture(scope="module")
def bt_con(tmp_path_factory):
    con = build_db(two_season_league(), tmp_path_factory.mktemp("bt") / "bt.duckdb")
    gid = con.execute("SELECT game_id FROM schedules WHERE season = 2025 AND week = 3 ORDER BY game_id LIMIT 1").fetchall()[0][0]
    gdate = con.execute("SELECT game_date FROM schedules WHERE game_id = ?", [gid]).fetchall()[0][0]
    BT_INFO.update(add_raw_odds(con, gid, gdate), game_id=gid)
    yield con
    con.close()


@pytest.fixture(scope="module")
def bt_run(bt_con):
    return bt.run_backtest(bt_con, bt.BacktestConfig(**CFG))


def test_walk_forward_trains_only_on_prior_seasons(bt_run, bt_con):
    summary, bets = bt_run
    first_2025 = pd.Timestamp(bt_con.execute("SELECT MIN(game_date) FROM schedules WHERE season = 2025").fetchall()[0][0])
    assert set(bets["season"]) == {2025}
    assert (pd.to_datetime(bets["trained_through"]) < first_2025).all()
    td_rows = summary[summary["market_category"].str.startswith("model_") & (summary["scope"] == "season")]
    assert len(td_rows) and (pd.to_datetime(td_rows["trained_through"]) < first_2025).all()


def test_predictions_do_not_depend_on_evaluation_season_outcomes(bt_con, bt_run, tmp_path):
    """Scramble season-2025 final scores: settlements change, model probabilities must not."""
    _, bets = bt_run
    lg = two_season_league()
    s = lg["schedules"]
    m = s["season"] == 2025
    s.loc[m, ["home_score", "away_score"]] = s.loc[m, ["away_score", "home_score"]].to_numpy() + 10
    with build_db(lg, tmp_path / "scrambled.duckdb") as c2:
        _, bets2 = bt.run_backtest(c2, bt.BacktestConfig(**{**CFG, "include_td": False}), write=False)
    a = bets[bets.price_source == "schedules_close"].set_index(["game_id", "market_key", "selection"]).sort_index()
    b = bets2.set_index(["game_id", "market_key", "selection"]).sort_index()
    common = a.index.intersection(b.index)
    assert len(common) > 20
    assert np.allclose(a.loc[common, "model_p_win"], b.loc[common, "model_p_win"])
    assert not (a.loc[common, "outcome"] == b.loc[common, "outcome"]).all()


def test_summary_metrics_and_categories(bt_run):
    summary, _ = bt_run
    cats = set(summary["market_category"])
    assert {"moneyline", "spread", "total", "player_receiving"} <= cats
    assert {"model_anytime_td", "model_first_td"} <= cats
    for col in ("brier_model", "brier_market", "logloss_model", "logloss_market", "roi_pct", "flat_roi_pct",
                "avg_clv_pct", "n_bets", "run_id", "trained_through"):
        assert col in summary.columns
    s = summary[summary["n_predictions"] > 0]
    assert s["brier_model"].between(0, 1).all() and (s["logloss_model"] > 0).all()
    assert set(summary["scope"]) == {"season", "all"}


def test_roi_and_settlement_arithmetic(bt_run):
    _, bets = bt_run
    b = bets[bets["is_bet"]]
    assert len(b) > 0
    expected = np.where(b["outcome"] == "win", b["stake"] * (b["bet_decimal"] - 1),
                        np.where(b["outcome"] == "loss", -b["stake"], 0.0))
    assert np.allclose(b["profit"], expected)
    assert (b.loc[b["outcome"] == "push", "profit"] == 0).all()
    assert (bets.loc[~bets["is_bet"], "stake"] == 0).all()
    assert (bets.loc[bets["ev_pct"] <= 0, "stake"] == 0).all()     # never stake negative EV
    assert (b["stake"] <= dv.DEFAULT_BANKROLL * dv.DEFAULT_MAX_STAKE_PCT + 1e-9).all()


def test_clv_from_raw_odds_snapshots(bt_run, bt_con):
    _, bets = bt_run
    raw = bets[bets["price_source"] == "raw_odds"]
    assert len(raw) > 0 and raw["clv_pct"].notna().all()
    assert bets.loc[bets["price_source"] == "schedules_close", "clv_pct"].isna().all()
    home_ml = raw[(raw.market_key == "h2h") & (raw.selection == TEAM_FULL[BT_INFO["home"]])].iloc[0]
    close_fair = dv.devig_american([-160, 100 + (160 - 100) * 0.8], "shin")[0]   # pinnacle at the close
    assert home_ml["close_fair_prob"] == pytest.approx(close_fair, abs=1e-9)
    assert home_ml["bet_decimal"] == pytest.approx(float(dv.american_to_decimal(-125)))  # best open price (DK)
    assert home_ml["clv_pct"] == pytest.approx(100 * (close_fair * home_ml["bet_decimal"] - 1))
    assert home_ml["clv_pct"] > 0   # line moved toward us


def test_player_props_use_entity_resolution(bt_run, bt_con):
    summary, bets = bt_run
    props = bets[bets.market_key == "player_receptions"]
    assert set(props["player_id"]) == {f"{BT_INFO['home']}_WR1"}
    assert summary["n_unmatched_players"].iloc[0] >= 1      # "Totally Unknown Guy" skipped, not guessed
    assert props["model_p_win"].between(0, 1).all()


def test_results_tables_written_and_appended(bt_con, bt_run):
    n1 = bt_con.execute("SELECT COUNT(DISTINCT run_id) FROM backtest_results").fetchall()[0][0]
    assert n1 == 1
    assert bt_con.execute("SELECT COUNT(*) FROM backtest_bets").fetchall()[0][0] == len(bt_run[1])
    bt.run_backtest(bt_con, bt.BacktestConfig(**{**CFG, "include_td": False, "include_props": False}))
    assert bt_con.execute("SELECT COUNT(DISTINCT run_id) FROM backtest_results").fetchall()[0][0] == 2


def test_market_category_mapping():
    assert bt.market_category("alternate_spreads") == "spread"
    assert bt.market_category("player_pass_yds") == "player_passing"
    assert bt.market_category("player_receptions") == "player_receiving"
    assert bt.market_category("player_1st_td") == "player_first_td"
    assert bt.market_category("player_sacks") == "player_other"
