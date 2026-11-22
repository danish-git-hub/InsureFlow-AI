"""Unit tests for central configuration."""

from app.config import PROJECT_ROOT, Settings, _resolve_sqlite_url


def test_demo_mode_without_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    assert Settings().demo_mode is True


def test_placeholder_key_is_demo_mode(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "your_api_key_here")
    assert Settings().demo_mode is True


def test_live_mode_with_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-test-key")
    s = Settings()
    assert s.demo_mode is False
    assert "Gemini" in s.mode_label


def test_relative_sqlite_path_is_made_absolute():
    url = _resolve_sqlite_url("sqlite:///data/test.db")
    assert url == "sqlite:///" + str((PROJECT_ROOT / "data" / "test.db").resolve())


def test_non_sqlite_url_unchanged():
    pg = "postgresql+psycopg://user:pw@localhost/insureflow"
    assert _resolve_sqlite_url(pg) == pg
