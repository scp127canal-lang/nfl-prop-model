"""Strict point-in-time tests for src/features/team_ratings.py and src/features/player_usage.py.

Core proof (for every week N):
  1. **Truncation:** build features from the full synthetic league and from a copy in which every
     stat from week N onward has been deleted (pbp, snap counts, participation; injury reports
     after week N, since week-N reports come out before kickoff). Week-N features must match
     exactly.
  2. **Poisoning:** overwrite week >= N stats with absurd values. Week-N features must not move.
  3. **Non-vacuity:** the same perturbations *do* change week N+1 features, so the comparison is
     really able to detect leakage.
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.features import player_usage as pu
from src.features import team_ratings as tr

SEASON = 2025
TEAMS = ["DET", "GB", "CHI", "MIN"]
PLAYED_WEEKS = 8
UPCOMING_WEEK = 9
PAIRINGS = [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]
SLOTS = {"QB": "QB", "RB1": "RB", "RB2": "RB", "WR1": "WR", "WR2": "WR", "WR3": "WR", "TE1": "TE"}
TARGET_W = {"WR1": .30, "WR2": .20, "WR3": .10, "TE1": .20, "RB1": .15, "RB2": .05}
CARRY_W = {"RB1": .70, "RB2": .20, "QB": .10}
SNAP_PCT = {"QB": 1.0, "RB1": .7, "RB2": .3, "WR1": .9, "WR2": .85, "WR3": .5, "TE1": .8}
GARBAGE_EPA = 50.0
# (week, team, slot): DET's WR1 is Out in week 5 and doesn't play; GB's RB1 is Out in week 7.
OUTS = {(5, "DET", "WR1"), (7, "GB", "RB1")}


def gid(team: str, slot: str) -> str:
    return f"{team}_{slot}"


def game_date(week: int) -> date:
    return date(SEASON, 9, 7) + timedelta(days=7 * (week - 1))


# ======================================================================================
# Synthetic league
# ======================================================================================
def make_league(seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    sched, pbp, part, snaps, inj = [], [], [], [], []
    for week in range(1, UPCOMING_WEEK + 1):
        for h, a in PAIRINGS[(week - 1) % 3]:
            home, away = TEAMS[h], TEAMS[a]
            game_id = f"{SEASON}_{week:02d}_{away}_{home}"
            sched.append({"game_id": game_id, "season": SEASON, "week": week, "game_type": "REG",
                          "game_date": game_date(week), "gameday": str(game_date(week)), "home_team": home,
                          "away_team": away, "location": "Home"})
            if week > PLAYED_WEEKS:
                continue
            play_id, gsr = 1, 3600
            for drive in range(12):
                off, de = (home, away) if drive % 2 == 0 else (away, home)
                active = [s for s in SLOTS if (week, off, s) not in OUTS]
                tw = np.array([TARGET_W.get(s, 0) for s in active]); tw = tw / tw.sum()
                cw = np.array([CARRY_W.get(s, 0) for s in active]); cw = cw / cw.sum()
                for _ in range(5):
                    qtr = min(4, 1 + (3600 - gsr) // 900)
                    is_pass = rng.random() < 0.6
                    yl = int(rng.integers(1, 99))
                    row = {"game_id": game_id, "play_id": float(play_id), "season": SEASON, "week": week,
                           "season_type": "REG", "posteam": off, "defteam": de, "qtr": float(qtr), "wp": 0.5,
                           "game_seconds_remaining": float(gsr), "fixed_drive": float(drive + 1),
                           "two_point_attempt": 0.0, "epa": float(rng.normal(0.05, 1.0)),
                           "yardline_100": float(yl), "down": float(rng.integers(1, 5)),
                           "ydstogo": float(rng.integers(1, 15)), "qb_scramble": 0.0, "sack": 0.0,
                           "receiver_player_id": None, "rusher_player_id": None, "air_yards": None,
                           "complete_pass": 0.0, "yards_gained": float(rng.integers(-2, 20))}
                    row["success"] = float(row["epa"] > 0)
                    if is_pass:
                        rec = active[rng.choice(len(active), p=tw)]
                        row.update(play_type="pass", qb_dropback=1.0, rush=0.0, pass_attempt=1.0, rush_attempt=0.0,
                                   receiver_player_id=gid(off, rec), air_yards=float(rng.integers(-3, 30)),
                                   complete_pass=float(rng.random() < 0.65))
                        on_field = [gid(off, s) for s in active if s != "WR3" or rng.random() < 0.5]
                        part.append({"game_id": game_id, "play_id": play_id, "season": SEASON,
                                     "offense_players": ";".join(on_field)})
                    else:
                        rusher = active[rng.choice(len(active), p=cw)]
                        row.update(play_type="run", qb_dropback=0.0, rush=1.0, pass_attempt=0.0, rush_attempt=1.0,
                                   rusher_player_id=gid(off, rusher))
                    pbp.append(row)
                    play_id += 1
                    gsr -= int(rng.integers(20, 40))
            # Garbage-time plays: 4th quarter, wp outside [0.05, 0.95], absurd EPA.
            for k, wp in enumerate([0.99, 0.01, 0.97]):
                pbp.append({**pbp[-1], "play_id": float(play_id + k), "qtr": 4.0, "wp": wp, "epa": GARBAGE_EPA,
                            "success": 1.0, "posteam": home, "defteam": away, "game_seconds_remaining": 60.0 - k,
                            "fixed_drive": 99.0, "play_type": "pass", "qb_dropback": 1.0, "rush": 0.0,
                            "pass_attempt": 1.0, "rush_attempt": 0.0, "receiver_player_id": gid(home, "TE1"),
                            "complete_pass": 1.0, "air_yards": 10.0})
            for team in (home, away):
                for slot, pct in SNAP_PCT.items():
                    if (week, team, slot) in OUTS:
                        continue
                    snaps.append({"game_id": game_id, "season": SEASON, "week": week, "team": team,
                                  "gsis_id": gid(team, slot), "offense_snaps": 60 * pct, "offense_pct": pct})
    for week, team, slot in OUTS:
        inj.append({"season": SEASON, "week": week, "team": team, "gsis_id": gid(team, slot), "report_status": "Out"})
    inj.append({"season": SEASON, "week": 3, "team": "CHI", "gsis_id": gid("CHI", "WR2"), "report_status": "Questionable"})
    people = [{"gsis_id": gid(t, s), "full_name": f"{t} {s}", "position": p} for t in TEAMS for s, p in SLOTS.items()]
    rosters = pd.DataFrame([{**p, "season": SEASON, "team": p["gsis_id"].split("_")[0]} for p in people])
    players = pd.DataFrame(people).rename(columns={"full_name": "display_name"})
    return {"schedules": pd.DataFrame(sched), "pbp": pd.DataFrame(pbp), "participation": pd.DataFrame(part),
            "snap_counts": pd.DataFrame(snaps), "injuries": pd.DataFrame(inj), "rosters": rosters, "players": players}


LEAGUE = make_league()
STAT_TABLES = ("pbp", "participation", "snap_counts")


def truncate(league: dict[str, pd.DataFrame], week_n: int) -> dict[str, pd.DataFrame]:
    """Drop all stats from week >= N; keep injury reports through week N (pre-game info)."""
    out = {k: v.copy() for k, v in league.items()}
    weeks = league["schedules"].set_index("game_id")["week"]
    for t in STAT_TABLES:
        out[t] = out[t][out[t]["game_id"].map(weeks) < week_n]
    out["injuries"] = out["injuries"][out["injuries"]["week"] <= week_n]
    return out


def poison(league: dict[str, pd.DataFrame], week_n: int) -> dict[str, pd.DataFrame]:
    """Overwrite stats from week >= N with absurd values (and add future injury reports)."""
    out = {k: v.copy() for k, v in league.items()}
    weeks = league["schedules"].set_index("game_id")["week"]
    p = out["pbp"]
    fut = p["game_id"].map(weeks) >= week_n
    p.loc[fut, "epa"] = 999.0
    p.loc[fut, "success"] = 1.0
    p.loc[fut, "yards_gained"] = 99.0
    p.loc[fut, "air_yards"] = 80.0
    p.loc[fut & (p["play_type"] == "pass"), "receiver_player_id"] = p.loc[fut & (p["play_type"] == "pass"), "posteam"] + "_WR3"
    p.loc[fut & (p["play_type"] == "run"), "rusher_player_id"] = p.loc[fut & (p["play_type"] == "run"), "posteam"] + "_RB2"
    p.loc[fut, "game_seconds_remaining"] = p.loc[fut, "game_seconds_remaining"] * 0.5
    s = out["snap_counts"]
    s.loc[s["game_id"].map(weeks) >= week_n, "offense_pct"] = 0.01
    extra = [{"season": SEASON, "week": w, "team": t, "gsis_id": gid(t, "WR2"), "report_status": "Out"}
             for w in range(week_n + 1, UPCOMING_WEEK + 1) for t in TEAMS]
    out["injuries"] = pd.concat([out["injuries"], pd.DataFrame(extra)], ignore_index=True)
    return out


def to_db(league: dict[str, pd.DataFrame], path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(path))
    for name, df in league.items():
        con.register("_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM _df")
        con.unregister("_df")
    return con


def team_feats(league, path) -> pd.DataFrame:
    with to_db(league, path) as con:
        return tr.compute_team_features(con)


def player_feats(league, path) -> pd.DataFrame:
    with to_db(league, path) as con:
        return pu.compute_player_features(con)


def assert_same(a: pd.DataFrame, b: pd.DataFrame, keys: list[str], label: str) -> None:
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)
    assert len(a) == len(b) and len(a) > 0, f"{label}: row counts {len(a)} vs {len(b)}"
    assert (a[keys].astype(str).to_numpy() == b[keys].astype(str).to_numpy()).all(), f"{label}: key mismatch"
    for col in a.columns:
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(x):
            ok = np.isclose(x.astype(float), y.astype(float), rtol=1e-9, atol=1e-12, equal_nan=True)
        else:
            norm = lambda s: np.array([None if pd.isna(v) else str(v) for v in s], dtype=object)  # noqa: E731
            ok = norm(x) == norm(y)
        assert np.all(ok), f"{label}: column {col!r} differs:\n{a.loc[~ok, keys + [col]]}\nvs\n{b.loc[~ok, keys + [col]]}"


@pytest.fixture(scope="module")
def full_team(tmp_path_factory):
    return team_feats(LEAGUE, tmp_path_factory.mktemp("t") / "full.duckdb")


@pytest.fixture(scope="module")
def full_player(tmp_path_factory):
    return player_feats(LEAGUE, tmp_path_factory.mktemp("p") / "full.duckdb")


# ======================================================================================
# Team features: point-in-time
# ======================================================================================
@pytest.mark.parametrize("week_n", range(2, UPCOMING_WEEK + 1))
def test_team_features_truncation_invariant(full_team, tmp_path, week_n):
    trunc = team_feats(truncate(LEAGUE, week_n), tmp_path / "trunc.duckdb")
    assert_same(full_team[full_team.week == week_n], trunc[trunc.week == week_n], ["game_id", "team"],
                f"team week {week_n}")


@pytest.mark.parametrize("week_n", range(2, PLAYED_WEEKS + 1))
def test_team_features_poison_invariant_and_non_vacuous(full_team, tmp_path, week_n):
    pois = team_feats(poison(LEAGUE, week_n), tmp_path / "poison.duckdb")
    for w in range(1, week_n + 1):
        assert_same(full_team[full_team.week == w], pois[pois.week == w], ["game_id", "team"], f"team week {w}")
    nxt_full = full_team[full_team.week == week_n + 1].sort_values(["game_id", "team"])
    nxt_pois = pois[pois.week == week_n + 1].sort_values(["game_id", "team"])
    assert not np.allclose(nxt_full["off_epa_per_dropback_5"], nxt_pois["off_epa_per_dropback_5"])


def test_team_rolling_matches_manual_computation(full_team):
    p = LEAGUE["pbp"]
    weeks = LEAGUE["schedules"].set_index("game_id")["week"]
    p = p.assign(week=p["game_id"].map(weeks))
    clean = p[~((p.qtr == 4) & ((p.wp < 0.05) | (p.wp > 0.95)))]
    for team, week_n, w in [("DET", 7, 5), ("GB", 8, 5), ("CHI", 6, 10), ("MIN", UPCOMING_WEEK, 5)]:
        prior_games = sorted(clean[(clean.posteam == team) & (clean.week < week_n)]["week"].unique())[-w:]
        sub = clean[(clean.posteam == team) & clean.week.isin(prior_games)]
        db = sub[sub.qb_dropback == 1]
        ru = sub[(sub.rush == 1) & (sub.qb_dropback == 0)]
        row = full_team[(full_team.team == team) & (full_team.week == week_n)].iloc[0]
        assert row[f"off_epa_per_dropback_{w}"] == pytest.approx(db.epa.sum() / len(db))
        assert row[f"off_epa_per_rush_{w}"] == pytest.approx(ru.epa.sum() / len(ru))
        assert row[f"off_success_rate_{w}"] == pytest.approx(sub.success.mean())
        assert row[f"games_in_window_{w}"] == len(prior_games)
        d = clean[(clean.defteam == team) & clean.week.isin(
            sorted(clean[(clean.defteam == team) & (clean.week < week_n)]["week"].unique())[-w:])]
        assert row[f"def_epa_per_dropback_{w}"] == pytest.approx(d[d.qb_dropback == 1].epa.mean())


def test_garbage_time_excluded(full_team):
    played = full_team[full_team.games_in_window_5 > 0]
    assert played["off_epa_per_dropback_5"].abs().max() < 5  # GARBAGE_EPA=50 plays would dominate
    assert played["off_success_rate_5"].max() < 0.9


def test_week1_has_no_history_and_rest_home_flags(full_team):
    wk1 = full_team[full_team.week == 1]
    assert wk1["off_epa_per_dropback_5"].isna().all() and wk1["rest_days"].isna().all()
    assert (full_team[full_team.week > 1]["rest_days"] == 7).all()
    g = full_team[full_team.game_id == f"{SEASON}_01_GB_DET"].set_index("team")
    assert g.loc["DET", "is_home"] == 1 and g.loc["GB", "is_home"] == 0
    assert g.loc["DET", "opponent"] == "GB"


def test_pace_seconds_per_play(full_team):
    pace = full_team["pace_sec_per_play_5"].dropna()
    assert pace.between(20, 40).all()  # synthetic snaps are 20-39s apart within a drive


def test_upcoming_week_uses_latest_played_games(full_team):
    up = full_team[full_team.week == UPCOMING_WEEK]
    assert len(up) == 4 and up["games_in_window_5"].eq(5).all()
    assert (pd.to_datetime(up["last_played_date"]) == pd.Timestamp(game_date(PLAYED_WEEKS))).all()


def test_opponent_features_join(full_team):
    r = full_team[(full_team.team == "DET") & (full_team.week == 6)].iloc[0]
    o = full_team[(full_team.team == r.opponent) & (full_team.week == 6)].iloc[0]
    assert r["opp_def_epa_per_dropback_5"] == pytest.approx(o["def_epa_per_dropback_5"])


def test_build_team_features_writes_table(tmp_path):
    with to_db(LEAGUE, tmp_path / "w.duckdb") as con:
        n = tr.build_team_features(con)
        assert n == con.execute("SELECT COUNT(*) FROM team_features").fetchall()[0][0] == len(LEAGUE["schedules"]) * 2


# ======================================================================================
# Player features: point-in-time
# ======================================================================================
PKEYS = ["game_id", "team", "gsis_id"]


@pytest.mark.parametrize("week_n", range(2, UPCOMING_WEEK + 1))
def test_player_features_truncation_invariant(full_player, tmp_path, week_n):
    trunc = player_feats(truncate(LEAGUE, week_n), tmp_path / "trunc.duckdb")
    a = full_player[(full_player.week == week_n) & full_player.eligible]
    b = trunc[(trunc.week == week_n) & trunc.eligible]
    assert_same(a, b, PKEYS, f"player week {week_n}")


@pytest.mark.parametrize("week_n", [3, 5, 6, 8])
def test_player_features_poison_invariant_and_non_vacuous(full_player, tmp_path, week_n):
    pois = player_feats(poison(LEAGUE, week_n), tmp_path / "poison.duckdb")
    for w in range(1, week_n + 1):
        a = full_player[(full_player.week == w) & full_player.eligible]
        b = pois[(pois.week == w) & pois.eligible]
        if len(a):
            assert_same(a, b, PKEYS, f"player week {w}")
    nf = full_player[(full_player.week == week_n + 1) & full_player.eligible].set_index(PKEYS).sort_index()
    npz = pois[(pois.week == week_n + 1) & pois.eligible].set_index(PKEYS).sort_index()
    common = nf.index.intersection(npz.index)
    assert not np.allclose(nf.loc[common, "target_share_ewm3"].fillna(0), npz.loc[common, "target_share_ewm3"].fillna(0))


def _raw_target_share() -> pd.DataFrame:
    """Independent recomputation: per-game target share for every game a player appeared in
    (snap counts), 0 when he wasn't targeted."""
    p = LEAGUE["pbp"]
    weeks = LEAGUE["schedules"].set_index("game_id")["week"]
    t = p[(p.pass_attempt == 1) & p.receiver_player_id.notna()]
    team_t = t.groupby(["game_id", "posteam"]).size().rename("team_targets")
    tg = t.groupby(["game_id", "receiver_player_id"]).size().rename("targets")
    apps = LEAGUE["snap_counts"][["game_id", "team", "gsis_id"]].rename(columns={"gsis_id": "receiver_player_id"})
    pl = apps.join(tg, on=["game_id", "receiver_player_id"]).fillna({"targets": 0})
    pl = pl.join(team_t, on=["game_id", "team"]).assign(week=lambda d: d.game_id.map(weeks))
    pl["share"] = pl["targets"] / pl["team_targets"]
    return pl.sort_values("week")


@pytest.mark.parametrize("player,week_n", [("DET_WR1", 4), ("DET_WR1", 7), ("GB_TE1", 6), ("CHI_RB1", UPCOMING_WEEK)])
def test_ewma_equals_pandas_on_prior_games_only(full_player, player, week_n):
    raw = _raw_target_share()
    hist = raw[(raw.receiver_player_id == player) & (raw.week < week_n)]["share"]
    expected3 = hist.ewm(span=3, min_periods=1).mean().iloc[-1]
    expected5 = hist.ewm(span=5, min_periods=1).mean().iloc[-1]
    row = full_player[(full_player.gsis_id == player) & (full_player.week == week_n)].iloc[0]
    assert row["target_share_ewm3"] == pytest.approx(expected3)
    assert row["target_share_ewm5"] == pytest.approx(expected5)
    assert row["prior_games"] == len(hist)


def test_route_participation_and_snap_share(full_player):
    wk = full_player[(full_player.week == 6) & full_player.eligible].set_index("gsis_id")
    assert wk.loc["DET_WR1", "route_participation_ewm5"] == pytest.approx(1.0)
    assert 0.3 < wk.loc["DET_WR3", "route_participation_ewm5"] < 0.7   # on the field for ~50% of dropbacks
    assert wk.loc["DET_QB", "route_participation_ewm5"] == 0.0
    assert wk.loc["DET_WR3", "snap_share_ewm5"] == pytest.approx(0.5)
    assert wk.loc["DET_WR1", "routes_from_participation_ewm5"] == pytest.approx(1.0)
    assert wk.loc["DET_WR1", "yprr_ewm5"] > 0
    assert np.isfinite(wk.loc["DET_RB1", "ryoe_per_carry_ewm5"])
    assert wk.loc["DET_RB1", "carry_share_ewm5"] > wk.loc["DET_RB2", "carry_share_ewm5"]
    assert wk.loc["DET_RB1", "rz_touches_ewm5"] >= wk.loc["DET_RB1", "rz_carries_ewm5"]


def test_snap_share_proxy_when_participation_missing(tmp_path):
    league = {k: v for k, v in LEAGUE.items() if k != "participation"}
    df = player_feats(league, tmp_path / "nopart.duckdb")
    wk = df[(df.week == 6) & df.eligible].set_index("gsis_id")
    assert wk.loc["DET_WR3", "route_participation_ewm5"] == pytest.approx(wk.loc["DET_WR3", "snap_share_ewm5"])
    assert wk.loc["DET_WR3", "routes_from_participation_ewm5"] == 0.0


def test_expected_rush_yards_uses_only_prior_dates(tmp_path):
    def xy(league, path):
        with to_db(league, path) as con:
            d = con.execute(pu.expected_rush_yards_sql()).df()
        return d.merge(league["pbp"][["game_id", "play_id", "week"]], on=["game_id", "play_id"])
    full = xy(LEAGUE, tmp_path / "a.duckdb")
    assert np.allclose(full[full.week == 1]["x_yards"], pu.XYARDS_DEFAULT)  # no prior carries
    for n in (3, 6):
        pois = xy(poison(LEAGUE, n), tmp_path / f"b{n}.duckdb")
        a = full[full.week <= n].sort_values(["game_id", "play_id"])["x_yards"].to_numpy()
        b = pois[pois.week <= n].sort_values(["game_id", "play_id"])["x_yards"].to_numpy()
        assert np.allclose(a, b)
        assert not np.allclose(full[full.week == n + 1]["x_yards"].to_numpy(),
                               pois.sort_values(["game_id", "play_id"])
                               .merge(full[["game_id", "play_id"]], on=["game_id", "play_id"])
                               .query("week == @n + 1")["x_yards"].to_numpy())


# ======================================================================================
# Injury spillover
# ======================================================================================
def test_injury_spillover_redistributes_vacated_share(full_player):
    g = full_player[(full_player.week == 5) & (full_player.team == "DET") & full_player.eligible].set_index("gsis_id")
    wr1 = g.loc["DET_WR1"]
    assert wr1["is_out"] and wr1["is_top2"] and wr1["injury_status"] == "Out"
    assert wr1["target_share_ewm5_adj"] == 0.0
    assert g["vacated_target_share_ewm5"].iloc[0] == pytest.approx(wr1["target_share_ewm5"])
    assert g["team_top2_out"].iloc[0] == 1
    # Totals preserved; every healthy pass-catcher gains; same group (WR) gains relatively more.
    assert g["target_share_ewm5_adj"].sum() == pytest.approx(g["target_share_ewm5"].fillna(0).sum())
    healthy = g.drop(index="DET_WR1")
    gain = healthy["target_share_ewm5_adj"] - healthy["target_share_ewm5"].fillna(0)
    assert (gain[healthy["target_share_ewm5"].fillna(0) > 0] > 0).all()
    rel = gain / healthy["target_share_ewm5"]
    assert rel["DET_WR2"] > rel["DET_TE1"]
    # WR1 isn't a rusher -> no carry spillover
    assert g["vacated_carry_share_ewm5"].iloc[0] == pytest.approx(wr1["carry_share_ewm5"])


def test_rb_out_redistributes_carries(full_player):
    g = full_player[(full_player.week == 7) & (full_player.team == "GB") & full_player.eligible].set_index("gsis_id")
    assert g.loc["GB_RB1", "is_out"]
    assert g.loc["GB_RB1", "carry_share_ewm3_adj"] == 0.0
    assert g.loc["GB_RB2", "carry_share_ewm3_adj"] == pytest.approx(
        g.loc["GB_RB2", "carry_share_ewm3"] + g.loc["GB_RB1", "carry_share_ewm3"])  # only other RB
    assert g.loc["GB_QB", "carry_share_ewm3_adj"] == pytest.approx(g.loc["GB_QB", "carry_share_ewm3"])


def test_no_spillover_without_out_or_for_questionable(full_player):
    g = full_player[(full_player.week == 3) & (full_player.team == "CHI") & full_player.eligible]
    assert g.set_index("gsis_id").loc["CHI_WR2", "injury_status"] == "Questionable"
    assert (g["vacated_target_share_ewm5"] == 0).all()
    assert np.allclose(g["target_share_ewm5_adj"].fillna(0), g["target_share_ewm5"].fillna(0))


def test_future_injury_does_not_leak_into_earlier_weeks(full_player):
    # GB RB1 is Out in week 7 only: weeks 6 and 8 must show no vacated share for GB.
    for w in (6, 8):
        g = full_player[(full_player.week == w) & (full_player.team == "GB")]
        assert (g["vacated_carry_share_ewm5"] == 0).all() and not g["is_out"].any()


# ======================================================================================
# Panel structure
# ======================================================================================
def test_upcoming_week_rows_exist_with_history_only(full_player):
    up = full_player[full_player.week == UPCOMING_WEEK]
    assert set(up.team) == set(TEAMS) and up["eligible"].all()
    assert len(up) == len(TEAMS) * len(SLOTS)
    assert up["target_share_ewm5"].notna().sum() > 0
    assert (up["days_since_last_game"] == 7).all()


def test_out_player_still_has_row_and_returns_next_week(full_player):
    r5 = full_player[(full_player.gsis_id == "DET_WR1") & (full_player.week == 5)].iloc[0]
    r6 = full_player[(full_player.gsis_id == "DET_WR1") & (full_player.week == 6)].iloc[0]
    assert r5["eligible"] and r6["eligible"]
    assert r6["days_since_last_game"] == 14 and r6["prior_games"] == r5["prior_games"]


def test_build_player_features_writes_table(tmp_path):
    with to_db(LEAGUE, tmp_path / "w.duckdb") as con:
        n = pu.build_player_features(con)
        assert n == con.execute("SELECT COUNT(*) FROM player_features").fetchall()[0][0] > 0
        assert con.execute("SELECT typeof(game_date) FROM player_features LIMIT 1").fetchall()[0][0] == "DATE"


# ======================================================================================
# Cold-start usage prior
# ======================================================================================
def test_qual_games_count_only_prior_games_above_snap_threshold(full_player):
    d = full_player[full_player.eligible & (full_player.gsis_id == "DET_WR1")].sort_values("week")
    played_before = d["week"].map(lambda w: sum(1 for x in range(1, w) if (x, "DET", "WR1") not in OUTS))
    assert (d["qual_games"].to_numpy() == played_before.to_numpy()).all()
    assert (full_player["qual_games"] <= full_player["prior_games"]).all()
    wk2 = full_player[(full_player.week == 2) & full_player.eligible]
    assert wk2["cold_start"].all()
    assert not full_player[(full_player.week == UPCOMING_WEEK) & full_player.eligible]["cold_start"].any()


def _stats(rows):
    cols = ["gsis_id", "position", "game_date", "snap_share", "target_share", "carry_share"]
    return pd.DataFrame(rows, columns=cols).assign(game_date=lambda d: pd.to_datetime(d["game_date"]))


def test_positional_usage_rates_are_cumulative_and_shrunk():
    s = _stats([("a", "WR", "2025-09-07", 0.9, 0.30, 0.0), ("b", "WR", "2025-09-07", 0.95, 0.20, 0.0),
                ("c", "WR", "2025-09-07", 0.10, 0.50, 0.0),                     # below 20% snaps: excluded
                ("a", "WR", "2025-09-14", 0.9, 0.10, 0.0)])
    r = pu.positional_usage_rates(s, k=10).set_index("rate_date")
    d_t = pu.USAGE_PRIOR_DEFAULTS["WR"][2][0]
    assert r.loc["2025-09-07", "pos_target_per_snap"] == pytest.approx((0.50 + 10 * d_t) / (1.85 + 10))
    assert r.loc["2025-09-14", "pos_target_per_snap"] == pytest.approx((0.60 + 10 * d_t) / (2.75 + 10))


def test_usage_prior_uses_only_earlier_dates():
    stats = _stats([("a", "WR", "2025-09-07", 0.9, 0.30, 0.0), ("a", "WR", "2025-09-14", 0.9, 0.90, 0.0)])
    panel = pd.DataFrame({"gsis_id": ["r", "r", "r"], "position": "WR",
                          "game_date": pd.to_datetime(["2025-09-07", "2025-09-14", "2025-09-21"]),
                          "last_snap_share": [0.9, 0.9, 0.9], "snap_share_ewm3": [0.9, 0.9, 0.9],
                          "qual_games": [0, 1, 1]})
    p = pu.attach_usage_prior(panel, stats).sort_values("game_date")
    d_t = pu.USAGE_PRIOR_DEFAULTS["WR"][2][0]
    k = pu.USAGE_PRIOR_K
    assert p["pos_target_per_snap"].iloc[0] == pytest.approx(d_t)                           # nothing earlier
    assert p["pos_target_per_snap"].iloc[1] == pytest.approx((0.30 + k * d_t) / (0.9 + k))  # not same-day 0.90
    assert p["pos_target_per_snap"].iloc[2] == pytest.approx((1.20 + k * d_t) / (1.8 + k))
    assert p["target_share_prior"].iloc[0] == pytest.approx(d_t * 0.9)
    assert p["cold_start"].all()


def test_expected_snap_weights_latest_game_and_sets_tier():
    stats = _stats([("a", "TE", "2025-09-07", 0.6, 0.1, 0.0)])
    panel = pd.DataFrame({"gsis_id": ["r"], "position": "TE", "game_date": pd.to_datetime(["2025-09-21"]),
                          "last_snap_share": [0.61], "snap_share_ewm3": [0.45], "qual_games": [1]})
    p = pu.attach_usage_prior(panel, stats).iloc[0]
    assert p["snap_share_expected"] == pytest.approx(0.6 * 0.61 + 0.4 * 0.45)
    assert p["usage_tier"] == 1                                            # starter tier (0.5-0.8)


# ======================================================================================
# Full roster / depth chart for upcoming games
# ======================================================================================
def with_roster(league: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Current-season roster snapshot with statuses + a depth chart:

    * DET_QB2 / DET_WR4: active backups with no snaps at all (must still get rows);
    * GB_WR2 on reserve/IR, CHI_RB2 cut, MIN_WR3 traded to GB."""
    out = {k: v.copy() for k, v in league.items()}
    r = out["rosters"].assign(status="ACT", week=UPCOMING_WEEK)
    extra = pd.DataFrame([{"gsis_id": "DET_QB2", "full_name": "DET QB2", "position": "QB", "season": SEASON,
                           "team": "DET", "status": "ACT", "week": UPCOMING_WEEK},
                          {"gsis_id": "DET_WR4", "full_name": "DET WR4", "position": "WR", "season": SEASON,
                           "team": "DET", "status": "ACT", "week": UPCOMING_WEEK}])
    r = pd.concat([r, extra], ignore_index=True)
    r.loc[r.gsis_id == "GB_WR2", "status"] = "RES"
    r.loc[r.gsis_id == "CHI_RB2", "status"] = "CUT"
    r.loc[r.gsis_id == "MIN_WR3", "team"] = "GB"
    out["rosters"] = r
    depth = [("DET", "DET_QB", "QB", 1), ("DET", "DET_QB2", "QB", 2), ("DET", "DET_WR1", "WR", 1),
             ("DET", "DET_WR2", "WR", 2), ("DET", "DET_WR3", "WR", 3), ("DET", "DET_WR4", "WR", 4)]
    out["depth_charts"] = pd.DataFrame([{"season": SEASON, "week": None, "snapshot_date": game_date(UPCOMING_WEEK) - timedelta(days=2),
                                         "team": t, "gsis_id": g, "player_name": g, "position": p, "depth_rank": k}
                                        for t, g, p, k in depth])
    return out


@pytest.fixture(scope="module")
def roster_player(tmp_path_factory):
    return player_feats(with_roster(LEAGUE), tmp_path_factory.mktemp("r") / "roster.duckdb")


def _row(df, gsis_id, week=UPCOMING_WEEK, team=None):
    d = df[(df.gsis_id == gsis_id) & (df.week == week)]
    return d[d.team == team].iloc[0] if team else d.iloc[0]


def test_backups_without_snaps_get_roster_rows(roster_player):
    qb2, wr4 = _row(roster_player, "DET_QB2"), _row(roster_player, "DET_WR4")
    for r in (qb2, wr4):
        assert r["eligible"] and r["roster_only"] and r["roster_status"] == "ACT" and r["prior_games"] == 0
    assert qb2["player_name"] == "DET QB2" and qb2["depth_rank"] == 2
    assert qb2["snap_share_expected"] == 0.0                              # backup QB: no usage prior
    assert wr4["snap_share_expected"] == pytest.approx(pu.DEPTH_SNAP_PRIOR["WR"][3])
    assert 0 < wr4["target_share_prior"] < 0.1                            # conservative, not zero
    assert not roster_player[roster_player.week < UPCOMING_WEEK]["roster_only"].any()


def test_roster_status_marks_ir_and_drops_cut_or_traded(roster_player):
    wr2 = _row(roster_player, "GB_WR2")
    assert wr2["is_out"] and wr2["injury_status"] == "IR"
    assert wr2["target_share_ewm5_adj"] == 0.0 and wr2["is_top2"]         # spillover to healthy teammates
    gb_wr1 = _row(roster_player, "GB_WR1")
    assert gb_wr1["target_share_ewm5_adj"] > gb_wr1["target_share_ewm5"]
    up = roster_player[roster_player.week == UPCOMING_WEEK]
    assert "CHI_RB2" not in set(up.gsis_id)                                # cut: not on the team any more
    moved = up[up.gsis_id == "MIN_WR3"]
    assert list(moved.team) == ["GB"] and moved["roster_only"].all()        # traded: new team's backup row


def test_roster_context_never_touches_played_games(full_player, roster_player):
    keys = ["game_id", "gsis_id"]
    a = full_player[full_player.week < UPCOMING_WEEK].drop(columns=["roster_only", "roster_status", "depth_rank"])
    b = roster_player[roster_player.week < UPCOMING_WEEK].drop(columns=["roster_only", "roster_status", "depth_rank"])
    assert_same(a, b, keys, "played weeks")
    played = roster_player[roster_player.week < UPCOMING_WEEK]
    assert played["roster_status"].isna().all() and played["depth_rank"].isna().all()


def test_depth_snap_prior_tiers():
    pos = pd.Series(["QB", "QB", "RB", "WR", "WR", "TE"])
    got = pu.depth_snap_prior(pos, pd.Series([1, 2, 9, 3, None, 2])).tolist()
    assert got == [0.97, 0.0, 0.12, 0.55, pu.NO_DEPTH_SNAP_PRIOR["WR"], 0.30]
