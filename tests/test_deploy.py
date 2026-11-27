"""Streamlit Community Cloud deployment: secrets fallback, password gate, DB selection, deploy export."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import duckdb
import pandas as pd
import pytest
import streamlit
from streamlit.testing.v1 import AppTest

from src.app import state
from src.ingestion import odds_client as oc
from src.models import game_sim as gs
from src.utils.secrets import get_secret
from tests.test_app import APP_PATH, TIMEOUT, app_env, env, no_network  # noqa: F401 - fixtures
from tests.test_evaluation import two_season_league
from tests.test_models import build_db

ROOT = Path(__file__).resolve().parents[1]


def _load_export_module():
    spec = importlib.util.spec_from_file_location("export_deploy_db", ROOT / "scripts" / "export_deploy_db.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------- secrets
def test_get_secret_prefers_env_then_st_secrets(monkeypatch):
    monkeypatch.setattr(streamlit, "secrets", {"MY_KEY": "from-secrets"})
    monkeypatch.setenv("MY_KEY", "from-env")
    assert get_secret("MY_KEY") == "from-env"
    monkeypatch.setenv("MY_KEY", "")
    assert get_secret("MY_KEY") == "from-secrets"
    monkeypatch.setattr(streamlit, "secrets", {})
    assert get_secret("MY_KEY", "dflt") == "dflt"


def test_get_secret_tolerates_broken_secrets(monkeypatch):
    class Boom:
        def get(self, *a, **k):
            raise FileNotFoundError("no secrets.toml")
    monkeypatch.setattr(streamlit, "secrets", Boom())
    monkeypatch.delenv("MY_KEY", raising=False)
    assert get_secret("MY_KEY") == ""


def test_load_api_key_falls_back_to_st_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("ODDS_API_KEY", "")
    monkeypatch.setattr(streamlit, "secrets", {"ODDS_API_KEY": "cloud-key"})
    assert oc.load_api_key(tmp_path / "missing.env") == "cloud-key"
    monkeypatch.setattr(streamlit, "secrets", {})
    with pytest.raises(RuntimeError, match="ODDS_API_KEY"):
        oc.load_api_key(tmp_path / "missing.env")


# ---------------------------------------------------------------------------------- DB selection
def test_db_path_prefers_full_then_deploy(monkeypatch, tmp_path):
    full, deploy = tmp_path / "nfl_model.duckdb", tmp_path / "deploy_model.duckdb"
    monkeypatch.delenv("NFL_DB_PATH", raising=False)
    monkeypatch.setattr(oc, "DEFAULT_DB_PATH", full)
    monkeypatch.setattr(state, "DEPLOY_DB_PATH", deploy)
    assert state.db_path() == full                      # neither exists: report the canonical path
    deploy.touch()
    assert state.db_path() == deploy                    # cloud: only the deploy DB is in the repo
    full.touch()
    assert state.db_path() == full                      # local: full DB wins
    monkeypatch.setenv("NFL_DB_PATH", str(tmp_path / "x.duckdb"))
    assert state.db_path() == tmp_path / "x.duckdb"     # explicit override wins


# ---------------------------------------------------------------------------------- password gate
def test_check_password(monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    assert state.check_password("s3cret") and not state.check_password("nope")
    monkeypatch.setenv("APP_PASSWORD", "")
    assert state.check_password("anything")


def _buttons(at):
    return [b.label for b in at.sidebar.button]


def test_password_gate_blocks_dashboard_until_correct(env, no_network, monkeypatch):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT).run()
    assert not at.exception
    assert "Pull Live Odds" not in _buttons(at) and not at.tabs
    at.text_input(key="login_password").input("wrong")     # form: value is sent with the submit
    at.button[0].click().run()
    assert any("Incorrect password" in e.value for e in at.error)
    assert "Pull Live Odds" not in _buttons(at)
    at.text_input(key="login_password").input("s3cret")
    at.button[0].click().run()
    assert not at.exception and "Pull Live Odds" in _buttons(at)
    assert at.session_state["authenticated"] and no_network == []


def test_password_gate_reads_st_secrets(env, no_network):
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.secrets["APP_PASSWORD"] = "cloudpw"
    at.run()
    assert "Pull Live Odds" not in _buttons(at)
    from streamlit.proto.TextInput_pb2 import TextInput as TextInputProto
    assert at.text_input(key="login_password").proto.type == TextInputProto.PASSWORD


def test_no_gate_without_password(env, no_network):
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT).run()
    assert "Pull Live Odds" in _buttons(at)


def test_pull_callback_refuses_when_unauthenticated(monkeypatch, no_network):
    monkeypatch.setenv("APP_PASSWORD", "s3cret")
    monkeypatch.setattr(state.st, "session_state", {})
    state.pull_live_odds_callback(groups=("main",), regions=("us",), max_events=1)
    assert no_network == [] and state.st.session_state["pull_error"] == "not authenticated"


# ---------------------------------------------------------------------------------- export
def test_export_deploy_db(tmp_path):
    src = tmp_path / "full.duckdb"
    build_db(two_season_league(), src).close()
    mod = _load_export_module()
    dst = tmp_path / "deploy.duckdb"
    counts = mod.export(src, dst, max_mb=40)
    assert dst.exists() and "pbp" not in counts
    assert {"team_features", "player_features", "schedules", "player_weekly", gs.TEAM_GAME_ACTUALS_TABLE} <= set(counts)
    with duckdb.connect(str(src), read_only=True) as a, duckdb.connect(str(dst), read_only=True) as b:
        full, slim = gs.build_game_dataset(a), gs.build_game_dataset(b)
        slim_cols = {r[0] for r in b.execute("DESCRIBE player_weekly").fetchall()}
        assert slim_cols <= set(mod.PLAYER_WEEKLY_COLUMNS) and {"gsis_id", "game_date", "targets"} <= slim_cols
    pd.testing.assert_frame_equal(full.reset_index(drop=True), slim.reset_index(drop=True), check_dtype=False)


def test_export_enforces_size_limit(tmp_path):
    src = tmp_path / "full.duckdb"
    build_db(two_season_league(), src).close()
    mod = _load_export_module()
    with pytest.raises(RuntimeError, match="MB"):
        mod.export(src, tmp_path / "deploy.duckdb", max_mb=0.001)
    assert not (tmp_path / "deploy.duckdb").exists()


def test_export_keeps_latest_depth_chart_and_roster_rows(tmp_path):
    src = tmp_path / "full.duckdb"
    con = build_db(two_season_league(), src)
    con.execute("""CREATE TABLE depth_charts AS SELECT * FROM (VALUES
        (2025, NULL, DATE '2025-11-20', 'DET', 'DET_QB', 'DET QB', 'QB', 1),
        (2025, NULL, DATE '2025-11-27', 'DET', 'DET_QB2', 'DET QB2', 'QB', 1),
        (2025, NULL, DATE '2025-11-27', 'DET', 'DET_QB', 'DET QB', 'QB', 2),
        (2025, NULL, DATE '2025-11-25', 'GB', 'GB_QB', 'GB QB', 'QB', 1))
        t(season, week, snapshot_date, team, gsis_id, player_name, position, depth_rank)""")
    con.close()
    dst = tmp_path / "deploy.duckdb"
    counts = _load_export_module().export(src, dst, max_mb=40)
    assert counts["depth_charts"] == 3
    with duckdb.connect(str(dst), read_only=True) as b:
        rows = b.execute("SELECT team, gsis_id, depth_rank FROM depth_charts ORDER BY team, depth_rank").fetchall()
        assert rows == [("DET", "DET_QB2", 1), ("DET", "DET_QB", 2), ("GB", "GB_QB", 1)]
        assert "roster_only" in {r[0] for r in b.execute("DESCRIBE player_features").fetchall()}
