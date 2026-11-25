"""Tests for src/models/game_sim.py and src/models/td_model.py.

Built on the synthetic league from tests/test_features.py, extended here with touchdowns, scores,
market lines and a player_weekly table, then run through the real feature builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import player_usage as pu
from src.features import team_ratings as tr
from src.models import game_sim as gs
from src.models import td_model as tdm
from tests.test_features import LEAGUE, PLAYED_WEEKS, UPCOMING_WEEK, to_db, truncate

N = 10_000
TD_PARAMS = {"min_data_in_leaf": 5, "num_leaves": 4, "bagging_fraction": 1.0, "bagging_freq": 0}


# ======================================================================================
# Synthetic data
# ======================================================================================
def enrich(league: dict[str, pd.DataFrame], seed: int = 3) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    out = {k: v.copy() for k, v in league.items()}
    p = out["pbp"]
    gain_td = p["yards_gained"] >= p["yardline_100"]
    is_pass = p["play_type"] == "pass"
    p["complete_pass"] = np.where(is_pass & gain_td & (p["qtr"] < 4), 1.0, p["complete_pass"])
    p["pass_touchdown"] = (is_pass & gain_td & (p["complete_pass"] == 1)).astype(float)
    p["rush_touchdown"] = (~is_pass & gain_td).astype(float)
    p["touchdown"] = p["pass_touchdown"] + p["rush_touchdown"]
    p["td_player_id"] = np.where(p["pass_touchdown"] == 1, p["receiver_player_id"],
                                 np.where(p["rush_touchdown"] == 1, p["rusher_player_id"], None))
    p["interception"] = (is_pass & (p["complete_pass"] == 0) & (rng.random(len(p)) < 0.04)).astype(float)

    s = out["schedules"]
    tds = p.groupby(["game_id", "posteam"])["touchdown"].sum()
    for col, side in (("home_score", "home_team"), ("away_score", "away_team")):
        s[col] = [7 * tds.get((g, t), 0) + 3 * rng.integers(0, 3) if w <= PLAYED_WEEKS else np.nan
                  for g, t, w in zip(s["game_id"], s[side], s["week"])]
    s["spread_line"] = np.round(np.nan_to_num(s["home_score"] - s["away_score"]) * 0.3 + rng.normal(0, 3, len(s)) * 2) / 2
    s["total_line"] = np.round((np.nan_to_num(s["home_score"] + s["away_score"], nan=44.0) * 0.3 + 31) * 2) / 2

    weeks = s.set_index("game_id")["week"]
    dates = s.set_index("game_id")["game_date"]
    rec = p[(p.pass_attempt == 1) & p.receiver_player_id.notna()].groupby(["game_id", "posteam", "receiver_player_id"]).agg(
        targets=("play_id", "size"), receptions=("complete_pass", "sum"),
        receiving_yards=("yards_gained", lambda x: x[p.loc[x.index, "complete_pass"] == 1].sum()))
    rec.index.names = ["game_id", "team", "gsis_id"]
    rush = p[p.rush_attempt == 1].groupby(["game_id", "posteam", "rusher_player_id"]).agg(
        carries=("play_id", "size"), rushing_yards=("yards_gained", "sum"))
    rush.index.names = ["game_id", "team", "gsis_id"]
    qb = p[p.pass_attempt == 1].groupby(["game_id", "posteam"]).agg(attempts=("play_id", "size"),
                                                                    passing_interceptions=("interception", "sum"))
    pw = rec.join(rush, how="outer").fillna(0).reset_index()
    qrows = qb.reset_index().rename(columns={"posteam": "team"})
    qrows["gsis_id"] = qrows["team"] + "_QB"
    pw = pw.merge(qrows, on=["game_id", "team", "gsis_id"], how="outer").fillna(0)
    pw["season"], pw["week"] = 2025, pw["game_id"].map(weeks)
    pw["game_date"] = pw["game_id"].map(dates)
    pw["position"] = pw["gsis_id"].str.split("_").str[1].str.rstrip("123")
    out["player_weekly"] = pw
    return out


def build_db(league: dict[str, pd.DataFrame], path):
    con = to_db(league, path)
    tr.build_team_features(con)
    pu.build_player_features(con)
    return con


@pytest.fixture(scope="module")
def con(tmp_path_factory):
    c = build_db(enrich(LEAGUE), tmp_path_factory.mktemp("m") / "models.duckdb")
    yield c
    c.close()


@pytest.fixture(scope="module")
def upcoming_game(con):
    return con.execute("SELECT game_id FROM schedules WHERE week = ? ORDER BY game_id LIMIT 1",
                       [UPCOMING_WEEK]).fetchall()[0][0]


@pytest.fixture(scope="module")
def sim(con, upcoming_game):
    return gs.simulate_game(upcoming_game, con=con, seed=42)


def _team_players(res: gs.SimResult, team: str) -> list[str]:
    return [p for p in res.stats if res.players.set_index("gsis_id").loc[p, "team"] == team]


# ======================================================================================
# Game model
# ======================================================================================
def test_game_model_is_market_anchored(con):
    gd = gs.build_game_dataset(con)
    model = gs.GameModel().fit(gd[gd["week"] < UPCOMING_WEEK])
    pred = model.predict(gd)
    dev = pred["pred_points"] - gd["implied_points"]
    assert dev.abs().max() <= gs.MAX_POINTS_DEVIATION + 1e-9
    assert pred["pred_plays"].between(45, 80).all() and pred["pred_pass_rate"].between(0.35, 0.8).all()
    assert model.resid["sd_plays"] > 0 and 0 < model.resid["sd_logit_pass_rate"] < 2
    for k in ("sack_rate", "scramble_rate", "target_rate", "int_rate", "pass_td_share"):
        assert 0 <= model.league[k] <= 1


def test_lgbm_game_model_option(con):
    gd = gs.build_game_dataset(con)
    m = gs.GameModel(kind="lgbm").fit(gd[gd["week"] < UPCOMING_WEEK])
    pred = m.predict(gd)
    assert np.isfinite(pred[["pred_points", "pred_plays", "pred_pass_rate"]].to_numpy()).all()
    assert (pred["pred_points"] - gd["implied_points"]).abs().max() <= gs.MAX_POINTS_DEVIATION + 1e-9
    with pytest.raises(ValueError):
        gs.GameModel(kind="xgb").fit(gd)


def test_model_trained_on_future_is_rejected(con, upcoming_game):
    gd = gs.build_game_dataset(con)
    leaky = gs.GameModel().fit(gd.dropna(subset=["points"]))
    past_game = gd.loc[gd["week"] == 3, "game_id"].iloc[0]
    with pytest.raises(ValueError, match="leakage"):
        gs.game_inputs(con, past_game, model=leaky)


def test_simulate_game_trains_only_on_prior_games(con, upcoming_game):
    gs._MODEL_CACHE.clear()
    gs.game_inputs(con, upcoming_game)
    (model,) = gs._MODEL_CACHE.values()
    gdate = pd.Timestamp(con.execute("SELECT game_date FROM schedules WHERE game_id = ?", [upcoming_game]).fetchall()[0][0])
    assert model.trained_through < gdate


# ======================================================================================
# Simulation: shapes & internal consistency
# ======================================================================================
COUNT_STATS = ["targets", "receptions", "rush_attempts", "rec_tds", "rush_tds", "anytime_td", "first_td"]


def test_result_shapes_and_types(sim):
    assert sim.n_trials == N and len(sim.stats) > 0
    for pid, stats in sim.stats.items():
        for stat, arr in stats.items():
            assert arr.shape == (N,), (pid, stat)
            assert np.isfinite(arr).all()
        for stat in COUNT_STATS:
            a = stats[stat]
            assert (a >= 0).all() and np.allclose(a, np.round(a)), (pid, stat)
        assert (stats["receptions"] <= stats["targets"]).all()
        assert (stats["reception_yds"] == np.round(stats["reception_yds"])).all()
        has = stats["receptions"] > 0
        assert (stats["reception_longest"][has] <= stats["reception_yds"][has] + 1e-9).all()
        assert (stats["reception_longest"][~has] == 0).all() and (stats["reception_yds"][~has] == 0).all()
        assert (stats["rush_yds"][stats["rush_attempts"] == 0] == 0).all()
        assert set(np.unique(stats["first_td"])) <= {0, 1}


def test_team_totals_are_consistent(sim):
    for team, t in sim.teams.items():
        players = _team_players(sim, team)
        qbs = [p for p in players if "pass_yds" in sim.stats[p]]
        assert len(qbs) == 1, "exactly one starting QB per team"
        qb = sim.stats[qbs[0]]
        named_rec = sum(sim.stats[p]["receptions"] for p in players)
        named_tgt = sum(sim.stats[p]["targets"] for p in players)
        named_car = sum(sim.stats[p]["rush_attempts"] for p in players)
        assert (qb["pass_completions"] >= named_rec).all()          # + unnamed 'other' receivers
        assert (t["targets"] >= named_tgt).all() and (t["carries"] >= named_car).all()
        assert (qb["pass_attempts"] >= t["targets"]).all()
        assert (qb["pass_completions"] <= qb["pass_attempts"]).all()
        assert (qb["pass_interceptions"] <= qb["pass_attempts"] - qb["pass_completions"]).all()
        assert (t["dropbacks"] == t["pass_attempts"] + t["sacks"] + t["scrambles"]).all()
        assert (t["plays"] >= t["dropbacks"]).all()
        assert (qb["pass_tds"] >= sum(sim.stats[p]["rec_tds"] for p in players)).all()  # + 'other' receivers
        assert (qb["pass_tds"] + sum(sim.stats[p]["rush_tds"] for p in players) <= t["tds"]).all()
    first = sum(s["first_td"] for s in sim.stats.values())
    assert (first <= 1).all()                                        # at most one first-TD scorer
    any_td = sum(t["tds"] for t in sim.teams.values()) > 0
    assert first[~any_td].sum() == 0


def test_points_track_model_prediction(sim):
    for team, t in sim.teams.items():
        assert t["points"].mean() == pytest.approx(sim.inputs[team].pred_points, abs=0.6)
        assert 5 < t["points"].std() < 16


def test_seed_reproducibility(con, upcoming_game):
    a = gs.simulate_game(upcoming_game, con=con, seed=7, n_trials=2000)
    b = gs.simulate_game(upcoming_game, con=con, seed=7, n_trials=2000)
    c = gs.simulate_game(upcoming_game, con=con, seed=8, n_trials=2000)
    pid = max(a.stats, key=lambda p: a.get(p, "targets").mean())  # a busy receiver
    assert np.array_equal(a.get(pid, "reception_yds"), b.get(pid, "reception_yds"))
    assert not np.array_equal(a.get(pid, "targets"), c.get(pid, "targets"))


# ======================================================================================
# Distribution shapes (pure simulator, hand-built inputs)
# ======================================================================================
def _params(team: str, **kw) -> gs.TeamParams:
    base = dict(team=team, pred_points=24.0, pred_plays=63.0, pred_pass_rate=0.58, sd_plays=6.0,
                sd_logit_pass_rate=0.25, beta_plays=0.05, beta_pass_rate=-0.02, sack_rate=0.065,
                scramble_rate=0.05, target_rate=0.96, int_rate=0.023, pass_td_share=0.62)
    return gs.TeamParams(**{**base, **kw})


def _players() -> pd.DataFrame:
    rows = []
    for team in ("AAA", "BBB"):
        for slot, pos, ts, cs, cr, ypr, ypc in [("QB", "QB", 0.0, 0.08, 0.5, 8, 5.0), ("RB1", "RB", 0.12, 0.65, 0.8, 7.5, 4.4),
                                                 ("RB2", "RB", 0.05, 0.22, 0.8, 7.0, 4.0), ("WR1", "WR", 0.28, 0.0, 0.65, 13.0, 6.0),
                                                 ("WR2", "WR", 0.20, 0.0, 0.62, 12.0, 6.0), ("TE1", "TE", 0.18, 0.0, 0.70, 10.0, 4.0)]:
            rows.append({"gsis_id": f"{team}_{slot}", "player_name": slot, "team": team, "position": pos,
                         "target_share": ts, "carry_share": cs, "catch_rate": cr, "ypr": ypr, "ypc": ypc,
                         "int_rate": 0.023, "rz_targets_ewm5": ts * 3, "rz_carries_ewm5": cs * 4,
                         "active": True, "is_starting_qb": slot == "QB"})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def pure_sim():
    return gs.simulate(_players(), {"AAA": _params("AAA"), "BBB": _params("BBB", pred_points=20.0)}, n_trials=N, seed=1)


def test_negative_binomial_moments():
    rng = np.random.default_rng(0)
    mu, k = 8.0, 6.0
    x = gs._nb(rng, np.full(400_000, mu), k)
    assert x.mean() == pytest.approx(mu, rel=0.01)
    assert x.var() == pytest.approx(mu + mu ** 2 / k, rel=0.03)


@pytest.mark.parametrize("allocation", ["constrained", "independent_nb"])
def test_counts_are_overdispersed(allocation):
    res = gs.simulate(_players(), {"AAA": _params("AAA"), "BBB": _params("BBB")}, n_trials=N, seed=2,
                      allocation=allocation)
    for pid, stat in [("AAA_WR1", "targets"), ("AAA_RB1", "rush_attempts"), ("AAA_WR1", "receptions"),
                      ("AAA_QB", "pass_completions")]:
        a = res.get(pid, stat)
        assert a.var() > a.mean(), (pid, stat, a.mean(), a.var())   # NB-type: Var > mean


def test_yardage_distributions(pure_sim):
    wr = pure_sim.stats["AAA_WR1"]
    has = wr["receptions"] > 0
    ypr_sim = wr["reception_yds"][has].sum() / wr["receptions"][has].sum()
    assert ypr_sim == pytest.approx(13.0, rel=0.04)                            # Log-Normal mean preserved
    assert wr["reception_yds"].mean() > np.median(wr["reception_yds"])        # right-skewed
    assert (wr["reception_yds"] >= 0).all()
    rb = pure_sim.stats["AAA_RB1"]
    ypc_sim = rb["rush_yds"].sum() / rb["rush_attempts"].sum()
    assert ypc_sim == pytest.approx(4.4, abs=0.15)                             # shifted Gamma mean
    assert (rb["rush_yds"] < 0).any()                                          # losses possible
    assert (rb["rush_longest"] <= rb["rush_yds"].max()).all()


def test_volume_follows_inputs(pure_sim):
    a = pure_sim.teams["AAA"]
    assert a["plays"].mean() == pytest.approx(63, abs=1.0)
    assert a["pass_rate"].mean() == pytest.approx(0.58, abs=0.02)
    t = pure_sim.stats["AAA_WR1"]["targets"].mean() / a["targets"].mean()
    assert t == pytest.approx(0.28, abs=0.015)
    c = pure_sim.stats["AAA_RB1"]["rush_attempts"].mean() / a["carries"].mean()
    assert c == pytest.approx(0.65, abs=0.02)


# ======================================================================================
# Pricing
# ======================================================================================
def test_probabilities_strictly_inside_unit_interval(sim):
    for pid, stats in sim.stats.items():
        for stat in ("receptions", "reception_yds", "rush_yds", "anytime_td"):
            for line in (-100.5, 0.5, 2.5, 49.5, 1000.5):
                p = sim.prob_over(pid, stat, line)
                assert 0 < p < 1


def test_over_under_and_alt_lines(pure_sim):
    pid = "AAA_WR1"
    lines = [0.5, 2.5, 4.5, 6.5, 9.5]
    overs = [pure_sim.prob_over(pid, "receptions", l) for l in lines]
    assert all(a > b for a, b in zip(overs, overs[1:]))                       # monotone in the line
    for l in lines:
        assert pure_sim.prob_over(pid, "receptions", l) + pure_sim.prob_under(pid, "receptions", l) == pytest.approx(1.0)
    for k in (1, 3, 5, 8):                                                     # alt "X+" == Over (X-0.5)
        assert pure_sim.prob_at_least(pid, "receptions", k) == pure_sim.prob_over(pid, "receptions", k - 0.5)
    raw = pure_sim.prob_over(pid, "reception_yds", 59.5, smooth=False)
    assert raw == (pure_sim.get(pid, "reception_yds") > 59.5).mean()
    p = pure_sim.price_market("player_reception_yds", pid, 59.5)
    assert p == pytest.approx(raw, abs=1e-3)
    assert pure_sim.price_market("player_anytime_td", pid) == pure_sim.prob_at_least(pid, "anytime_td", 1)
    assert gs.SimResult.fair_american(0.5) == pytest.approx(-100)
    assert gs.SimResult.fair_american(0.25) == pytest.approx(300)


def test_summary_frame(sim):
    s = sim.summary()
    assert {"gsis_id", "stat", "mean", "median", "p10", "p90"} <= set(s.columns)
    assert (s["p10"] <= s["p90"]).all()


# ======================================================================================
# Usage overrides
# ======================================================================================
def _wr_and_team(con, game_id):
    pl = gs.load_players(con, game_id)
    wr = pl[(pl.position == "WR") & pl.active].sort_values("target_share").iloc[0]
    return pl, wr["gsis_id"], wr["team"]


def test_target_share_override_moves_volume(con, upcoming_game, sim):
    pl, wr, team = _wr_and_team(con, upcoming_game)
    base_share = pl.set_index("gsis_id").loc[wr, "target_share"]
    res = gs.simulate_game(upcoming_game, {wr: {"target_share": base_share + 0.15}}, con=con, seed=42)
    assert res.get(wr, "targets").mean() > sim.get(wr, "targets").mean() * 1.3
    mates = [p for p in _team_players(res, team) if p != wr and sim.get(p, "targets").mean() > 0.5]
    assert all(res.get(p, "targets").mean() < sim.get(p, "targets").mean() for p in mates)
    assert res.teams[team]["targets"].mean() == pytest.approx(sim.teams[team]["targets"].mean(), rel=0.03)
    shares = gs.apply_usage_overrides(pl, {wr: {"target_share": base_share + 0.15}}, [team])[0]
    assert shares.loc[shares.team == team, "target_share"].sum() == pytest.approx(
        pl.loc[pl.team == team, "target_share"].sum())


def test_inactive_override_zeroes_player_and_redistributes(con, upcoming_game, sim):
    pl, _, _ = _wr_and_team(con, upcoming_game)
    top = pl[pl.active & (pl.position == "WR")].sort_values("target_share", ascending=False).iloc[0]
    pid, team = top["gsis_id"], top["team"]
    res = gs.simulate_game(upcoming_game, {pid: {"active": False}}, con=con, seed=42)
    assert pid not in res.stats  # removed from the active roster entirely
    mates = [p for p in _team_players(res, team) if sim.get(p, "targets").mean() > 0.5]
    assert all(res.get(p, "targets").mean() > sim.get(p, "targets").mean() for p in mates)
    named = lambda r: sum(r.get(p, "targets").mean() for p in _team_players(r, team))  # noqa: E731
    assert named(res) == pytest.approx(named(sim), rel=0.05)  # share handed to teammates, not lost
    no_redis = gs.simulate_game(upcoming_game, {pid: {"active": False, "redistribute": False}}, con=con, seed=42)
    assert named(no_redis) < named(res)


def test_team_and_efficiency_overrides(con, upcoming_game, sim):
    team = next(iter(sim.teams))
    res = gs.simulate_game(upcoming_game, {team: {"pass_rate": 0.75, "points": 35.0}}, con=con, seed=42)
    assert res.teams[team]["dropbacks"].mean() > sim.teams[team]["dropbacks"].mean() + 3
    assert res.teams[team]["points"].mean() == pytest.approx(35.0, abs=0.8)
    pl, wr, wteam = _wr_and_team(con, upcoming_game)
    big = gs.simulate_game(upcoming_game, {wr: {"ypr": 20.0, "catch_rate": 0.9}}, con=con, seed=42)
    assert big.get(wr, "reception_yds").mean() > sim.get(wr, "reception_yds").mean()
    with pytest.raises(KeyError):
        gs.simulate_game(upcoming_game, {"NOT_A_PLAYER": {"target_share": 0.2}}, con=con)


def test_market_line_override(con, upcoming_game, sim):
    res = gs.simulate_game(upcoming_game, con=con, seed=42, spread=10.0, total=60.0)
    home = con.execute("SELECT home_team FROM schedules WHERE game_id = ?", [upcoming_game]).fetchall()[0][0]
    away = next(t for t in res.teams if t != home)
    assert res.inputs[home].pred_points == pytest.approx(35.0, abs=gs.MAX_POINTS_DEVIATION + 1e-9)
    assert res.inputs[away].pred_points == pytest.approx(25.0, abs=gs.MAX_POINTS_DEVIATION + 1e-9)


# ======================================================================================
# TD model
# ======================================================================================
@pytest.fixture(scope="module")
def td_data(con):
    return tdm.build_td_dataset(con)


@pytest.fixture(scope="module")
def td_models(td_data):
    return {t: tdm.train_td_model(td_data, t, params=TD_PARAMS, num_boost_round=40, cal_frac=0.4)
            for t in tdm.TARGETS}


def test_td_dataset_has_features_and_labels(td_data):
    for f in ["rz_snap_share_ewm5", "i10_carries_ewm5", "i10_targets_ewm5", "implied_team_total", "tds_ewm5"]:
        assert f in td_data.columns
    played = td_data[td_data.played == 1]
    assert played["anytime_td"].sum() > 0 and played["first_td"].sum() > 0
    assert set(played["anytime_td"].unique()) <= {0, 1}
    assert (played["first_td"] <= played["anytime_td"]).all()
    per_game = played.groupby("game_id")["first_td"].sum()
    assert (per_game <= 1).all()
    wk1 = td_data[td_data.week == 1]
    assert wk1["i10_carries_ewm5"].isna().all()                     # no history before week 1
    assert td_data["rz_snap_share_ewm5"].dropna().between(0, 1).all()


def test_td_features_point_in_time(tmp_path, td_data):
    week_n = 5
    lg = enrich(LEAGUE)
    tr_lg = truncate(lg, week_n)
    weeks = lg["schedules"].set_index("game_id")["week"]
    tr_lg["player_weekly"] = lg["player_weekly"][lg["player_weekly"]["game_id"].map(weeks) < week_n]
    with build_db(tr_lg, tmp_path / "trunc.duckdb") as c2:
        trunc = tdm.build_td_dataset(c2)
    cols = [c for c in tdm.FEATURES if c in td_data.columns]
    a = td_data[(td_data.week == week_n) & td_data.eligible].set_index(["game_id", "gsis_id"]).sort_index()[cols]
    b = trunc[(trunc.week == week_n) & trunc.eligible].set_index(["game_id", "gsis_id"]).sort_index()[cols]
    pd.testing.assert_frame_equal(a.astype(float), b.astype(float), check_exact=False, rtol=1e-9)


@pytest.mark.parametrize("target", tdm.TARGETS)
def test_td_probabilities_strictly_between_0_and_1(td_models, td_data, target):
    p = td_models[target].predict_proba(td_data)
    assert np.isfinite(p).all() and (p > 0).all() and (p < 1).all()
    extreme = td_data.head(50).copy()
    for f in td_models[target].features:
        extreme[f] = 1e6
    pe = td_models[target].predict_proba(extreme)
    assert (pe > 0).all() and (pe < 1).all()
    extreme[td_models[target].features] = -1e6
    pe = td_models[target].predict_proba(extreme)
    assert (pe > 0).all() and (pe < 1).all()


@pytest.mark.parametrize("target", tdm.TARGETS)
def test_td_calibration_in_aggregate(td_models, td_data, target):
    m = td_models[target]
    ev = tdm.evaluate(m, td_data)
    assert ev["mean_pred"] == pytest.approx(ev["base_rate"], abs=0.06)
    assert 0 < ev["brier"] < 0.25 and np.isfinite(ev["logloss"])
    assert {"mean_pred", "observed", "n"} <= set(ev["calibration"].columns)


def test_td_monotone_in_implied_total_and_red_zone_usage(td_models, td_data):
    m = td_models["anytime_td"]
    rows = td_data[td_data.played == 1].head(40).copy()
    lo, hi = rows.copy(), rows.copy()
    lo["implied_team_total"], hi["implied_team_total"] = 14.0, 34.0
    assert (m.predict_proba(hi, normalize_games=False) >= m.predict_proba(lo, normalize_games=False) - 1e-12).all()
    lo["i10_carries_ewm5"], hi["i10_carries_ewm5"] = 0.0, 5.0
    assert (m.predict_proba(hi, normalize_games=False) >= m.predict_proba(lo, normalize_games=False) - 1e-12).all()


def test_first_td_probabilities_sum_at_most_one_per_game(td_models, td_data):
    m = td_models["first_td"]
    rows = td_data[td_data.eligible].copy()
    rows["p"] = m.predict_proba(rows)
    assert (rows.groupby("game_id")["p"].sum() <= m.first_td_cap + 1e-9).all()
    assert m.first_td_cap <= 1.0


def test_td_model_save_load_roundtrip(td_models, td_data, tmp_path):
    m = td_models["anytime_td"]
    m.save(tmp_path / "m.pkl")
    m2 = tdm.CalibratedTDModel.load(tmp_path / "m.pkl")
    assert np.allclose(m.predict_proba(td_data), m2.predict_proba(td_data))


def test_td_training_respects_cutoff(td_data):
    cutoff = td_data.loc[td_data.week == 7, "game_date"].min()
    m = tdm.train_td_model(td_data, "anytime_td", before=cutoff, params=TD_PARAMS, num_boost_round=20, cal_frac=0.4)
    assert pd.Timestamp(m.meta["trained_through"]) < pd.Timestamp(cutoff)


def test_predict_game_output(td_models, td_data, upcoming_game):
    out = tdm.predict_game(td_data, upcoming_game, td_models)
    assert len(out) > 0 and {"p_anytime_td", "p_first_td"} <= set(out.columns)
    assert (out["p_first_td"] < out["p_anytime_td"] + 1e-9).mean() > 0.9


# ======================================================================================
# Cold-start blending
# ======================================================================================
def test_blend_cold_start_weights_by_qualifying_games():
    df = pd.DataFrame({"qual_games": [0, 1, 2, 3, 12], "prior_games": [0, 4, 5, 6, 40],
                       "target_share_own": [0.0, 0.03, 0.06, 0.20, 0.25], "carry_share_own": [0.0, 0.0, 0.1, 0.0, 0.0],
                       "target_share_prior": [0.12, 0.12, 0.12, 0.05, 0.05],
                       "carry_share_prior": [0.0, 0.0, 0.4, 0.0, 0.0]})
    out = gs.blend_cold_start(df.copy())
    assert out["usage_prior_weight"].tolist() == pytest.approx([1.0, 2 / 3, 1 / 3, 0.0, 0.0])
    assert out["target_share"].tolist() == pytest.approx([0.12, 0.03 / 3 + 0.12 * 2 / 3, 0.06 * 2 / 3 + 0.04,
                                                          0.20, 0.25])
    assert out["carry_share"].iloc[2] == pytest.approx(0.1 * 2 / 3 + 0.4 / 3)
    assert out["cold_start"].tolist() == [True, True, True, False, False]


def test_blend_cold_start_without_prior_columns_keeps_own_usage():
    df = pd.DataFrame({"qual_games": [None, None], "prior_games": [1, 9], "target_share_own": [0.02, 0.2],
                       "carry_share_own": [0.0, 0.0], "target_share_prior": [None, None],
                       "carry_share_prior": [None, None]})
    out = gs.blend_cold_start(df.copy())
    assert out["target_share"].tolist() == [0.02, 0.2] and not out["cold_start"].any()
    assert out["qual_games"].tolist() == [1, 9]


def test_load_players_blends_only_thin_history(con):
    gid = con.execute("SELECT game_id FROM schedules WHERE week = 9 ORDER BY game_id LIMIT 1").fetchall()[0][0]
    players = gs.load_players(con, gid)
    est = players[players["qual_games"] >= gs.COLD_START_GAMES]
    assert len(est) and (est["usage_prior_weight"] == 0).all()
    own = est[["target_share_ewm3_adj", "target_share_ewm5_adj"]].mean(axis=1).fillna(0)
    assert np.allclose(est["target_share_own"], own)
    rookie = players.iloc[[0]].copy()
    rookie[["qual_games", "target_share_own", "target_share_prior", "carry_share_own", "carry_share_prior"]] = \
        [1, 0.03, 0.12, 0.0, 0.0]
    blended = gs.blend_cold_start(rookie)
    assert blended["target_share"].iloc[0] == pytest.approx(0.03 / 3 + 0.12 * 2 / 3)   # 0.09: 3x raw EWMA
