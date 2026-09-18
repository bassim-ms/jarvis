"""
Unit tests for SQLiteTokenStore (app/calendar/token_store.py):
persistence, restart behavior, and clear(). Uses real SQLite (":memory:"
or a temp file), never mocked - same convention as
test_reminders_store.py.
"""

from app.calendar.base import OAuthToken
from app.calendar.token_store import SQLiteTokenStore


def _token(access: str = "a", refresh: str = "r") -> OAuthToken:
    return OAuthToken(
        access_token=access,
        refresh_token=refresh,
        expires_at="2026-01-01T00:00:00+00:00",
        scope="calendar.events",
    )


def test_get_returns_none_when_never_saved():
    store = SQLiteTokenStore(":memory:")

    assert store.get() is None


def test_save_then_get_round_trips():
    store = SQLiteTokenStore(":memory:")

    store.save(_token())

    assert store.get() == _token()


def test_save_overwrites_the_previous_token():
    store = SQLiteTokenStore(":memory:")
    store.save(_token(access="first"))

    store.save(_token(access="second"))

    assert store.get().access_token == "second"


def test_clear_removes_the_token():
    store = SQLiteTokenStore(":memory:")
    store.save(_token())

    store.clear()

    assert store.get() is None


def test_clear_when_nothing_saved_does_not_raise():
    store = SQLiteTokenStore(":memory:")

    store.clear()  # must not raise


def test_token_persists_across_store_instances(tmp_path):
    db_path = str(tmp_path / "token.db")
    first_store = SQLiteTokenStore(db_path)
    first_store.save(_token(refresh="survive-restart"))

    second_store = SQLiteTokenStore(db_path)

    fetched = second_store.get()
    assert fetched is not None
    assert fetched.refresh_token == "survive-restart"


def test_a_restarted_store_reflects_a_cleared_token(tmp_path):
    db_path = str(tmp_path / "token.db")
    first_store = SQLiteTokenStore(db_path)
    first_store.save(_token())
    first_store.clear()

    second_store = SQLiteTokenStore(db_path)

    assert second_store.get() is None
