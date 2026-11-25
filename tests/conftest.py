import pytest


@pytest.fixture(autouse=True)
def _no_app_password(monkeypatch):
    """Keep a developer's real APP_PASSWORD (.env) from gating the AppTest runs; tests that
    exercise the gate set it explicitly. An existing (empty) env var stops load_dotenv overriding it."""
    monkeypatch.setenv("APP_PASSWORD", "")
