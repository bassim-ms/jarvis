"""
Tests for GoogleCalendarProvider (app/calendar/google_provider.py): event
JSON<->CalendarEvent mapping, all four CRUD operations, automatic token
refresh before a call, and error handling. All httpx calls are
monkeypatched - no real network call is ever made.
"""

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from app.calendar import google_provider as provider_module
from app.calendar import oauth as oauth_module
from app.calendar.base import OAuthToken, TokenStore
from app.calendar.google_provider import GoogleCalendarError, GoogleCalendarProvider
from app.calendar.oauth import GoogleOAuthClient


class FakeTokenStore(TokenStore):
    def __init__(self, token: OAuthToken | None = None):
        self.token = token

    def get(self):
        return self.token

    def save(self, token):
        self.token = token

    def clear(self):
        self.token = None


def _fresh_token(access: str = "access-1") -> OAuthToken:
    return OAuthToken(
        access_token=access,
        refresh_token="refresh-1",
        expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        scope="calendar.events",
    )


def _expired_token(access: str = "stale") -> OAuthToken:
    return OAuthToken(
        access_token=access,
        refresh_token="refresh-1",
        expires_at=(datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat(),
        scope="calendar.events",
    )


def make_provider(token: OAuthToken | None):
    oauth_client = GoogleOAuthClient(
        client_id="cid", client_secret="secret", redirect_uri="http://localhost:8765/"
    )
    token_store = FakeTokenStore(token=token)
    return GoogleCalendarProvider(oauth_client=oauth_client, token_store=token_store), token_store


# --- not authorized ------------------------------------------------------


def test_raises_clean_error_when_never_authorized():
    provider, _ = make_provider(token=None)

    with pytest.raises(GoogleCalendarError, match="not authorized"):
        provider.list_events()


# --- automatic token refresh ----------------------------------------------


def test_expired_token_is_refreshed_before_a_call(monkeypatch):
    provider, token_store = make_provider(token=_expired_token())

    def fake_post(url, data, timeout):
        assert data["grant_type"] == "refresh_token"
        return httpx.Response(
            200,
            json={"access_token": "fresh-access", "expires_in": 3600},
            request=httpx.Request("POST", url),
        )

    def fake_request(method, url, headers, timeout, **kwargs):
        assert headers["Authorization"] == "Bearer fresh-access"
        return httpx.Response(200, json={"items": []}, request=httpx.Request(method, url))

    monkeypatch.setattr(oauth_module.httpx, "post", fake_post)
    monkeypatch.setattr(provider_module.httpx, "request", fake_request)

    provider.list_events()

    assert token_store.token.access_token == "fresh-access"


def test_fresh_token_is_not_refreshed(monkeypatch):
    provider, _ = make_provider(token=_fresh_token())

    def fake_post(url, data, timeout):
        raise AssertionError("should not need to refresh a fresh token")

    def fake_request(method, url, headers, timeout, **kwargs):
        assert headers["Authorization"] == "Bearer access-1"
        return httpx.Response(200, json={"items": []}, request=httpx.Request(method, url))

    monkeypatch.setattr(oauth_module.httpx, "post", fake_post)
    monkeypatch.setattr(provider_module.httpx, "request", fake_request)

    provider.list_events()


def test_refresh_failure_raises_a_clean_error(monkeypatch):
    provider, _ = make_provider(token=_expired_token())

    def fake_post(url, data, timeout):
        raise httpx.ConnectError("nope")

    monkeypatch.setattr(oauth_module.httpx, "post", fake_post)

    with pytest.raises(GoogleCalendarError, match="Failed to refresh"):
        provider.list_events()


# --- list_events -----------------------------------------------------


def test_list_events_maps_google_response_to_calendar_events(monkeypatch):
    provider, _ = make_provider(token=_fresh_token())

    def fake_request(method, url, headers, timeout, **kwargs):
        assert method == "GET"
        assert url.endswith("/calendars/primary/events")
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "evt1",
                        "summary": "Standup",
                        "start": {"dateTime": "2026-01-01T09:00:00+00:00"},
                        "end": {"dateTime": "2026-01-01T09:30:00+00:00"},
                        "description": "daily",
                        "location": "Zoom",
                    }
                ]
            },
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(provider_module.httpx, "request", fake_request)

    events = provider.list_events()

    assert len(events) == 1
    assert events[0].id == "evt1"
    assert events[0].title == "Standup"
    assert events[0].start == "2026-01-01T09:00:00+00:00"
    assert events[0].description == "daily"
    assert events[0].location == "Zoom"


def test_list_events_handles_all_day_events(monkeypatch):
    provider, _ = make_provider(token=_fresh_token())

    def fake_request(method, url, headers, timeout, **kwargs):
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": "e",
                        "summary": "Holiday",
                        "start": {"date": "2026-01-01"},
                        "end": {"date": "2026-01-02"},
                    }
                ]
            },
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(provider_module.httpx, "request", fake_request)

    events = provider.list_events()

    assert events[0].start == "2026-01-01"
    assert events[0].end == "2026-01-02"


def test_list_events_passes_time_window_and_max_results(monkeypatch):
    provider, _ = make_provider(token=_fresh_token())
    captured = {}

    def fake_request(method, url, headers, timeout, **kwargs):
        captured["params"] = kwargs.get("params")
        return httpx.Response(200, json={"items": []}, request=httpx.Request(method, url))

    monkeypatch.setattr(provider_module.httpx, "request", fake_request)

    provider.list_events(time_min="2026-01-01T00:00:00+00:00", time_max="2026-01-08T00:00:00+00:00", max_results=5)

    assert captured["params"]["timeMin"] == "2026-01-01T00:00:00+00:00"
    assert captured["params"]["timeMax"] == "2026-01-08T00:00:00+00:00"
    assert captured["params"]["maxResults"] == 5


def test_list_events_raises_on_unexpected_response(monkeypatch):
    provider, _ = make_provider(token=_fresh_token())

    def fake_request(method, url, headers, timeout, **kwargs):
        return httpx.Response(200, json={"unexpected": True}, request=httpx.Request(method, url))

    monkeypatch.setattr(provider_module.httpx, "request", fake_request)

    with pytest.raises(GoogleCalendarError, match="Unexpected Google Calendar response"):
        provider.list_events()


def test_list_events_raises_on_a_malformed_event_item(monkeypatch):
    provider, _ = make_provider(token=_fresh_token())

    def fake_request(method, url, headers, timeout, **kwargs):
        return httpx.Response(
            200, json={"items": [{"id": "e1"}]}, request=httpx.Request(method, url)
        )

    monkeypatch.setattr(provider_module.httpx, "request", fake_request)

    with pytest.raises(GoogleCalendarError, match="Unexpected Google Calendar event"):
        provider.list_events()


def test_list_events_raises_on_http_error(monkeypatch):
    provider, _ = make_provider(token=_fresh_token())

    def fake_request(method, url, headers, timeout, **kwargs):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(provider_module.httpx, "request", fake_request)

    with pytest.raises(GoogleCalendarError, match="Failed to reach Google Calendar"):
        provider.list_events()


# --- create_event -----------------------------------------------------


def test_create_event_sends_correct_body_and_maps_response(monkeypatch):
    provider, _ = make_provider(token=_fresh_token())
    sent = {}

    def fake_request(method, url, headers, timeout, **kwargs):
        sent["method"] = method
        sent["url"] = url
        sent["json"] = kwargs.get("json")
        return httpx.Response(
            200,
            json={
                "id": "new1",
                "summary": "Meeting",
                "start": {"dateTime": "s"},
                "end": {"dateTime": "e"},
            },
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(provider_module.httpx, "request", fake_request)

    event = provider.create_event(
        title="Meeting", start="s", end="e", description="d", location="l"
    )

    assert sent["method"] == "POST"
    assert sent["url"].endswith("/calendars/primary/events")
    assert sent["json"] == {
        "summary": "Meeting",
        "start": {"dateTime": "s"},
        "end": {"dateTime": "e"},
        "description": "d",
        "location": "l",
    }
    assert event.id == "new1"


def test_create_event_raises_on_http_error(monkeypatch):
    provider, _ = make_provider(token=_fresh_token())

    def fake_request(method, url, headers, timeout, **kwargs):
        raise httpx.ConnectError("nope")

    monkeypatch.setattr(provider_module.httpx, "request", fake_request)

    with pytest.raises(GoogleCalendarError):
        provider.create_event(title="x", start="s", end="e")


# --- update_event -----------------------------------------------------


def test_update_event_sends_only_provided_fields(monkeypatch):
    provider, _ = make_provider(token=_fresh_token())
    sent = {}

    def fake_request(method, url, headers, timeout, **kwargs):
        sent["method"] = method
        sent["url"] = url
        sent["json"] = kwargs.get("json")
        return httpx.Response(
            200,
            json={
                "id": "evt1",
                "summary": "New title",
                "start": {"dateTime": "s"},
                "end": {"dateTime": "e"},
            },
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(provider_module.httpx, "request", fake_request)

    event = provider.update_event("evt1", title="New title")

    assert sent["method"] == "PATCH"
    assert sent["url"].endswith("/events/evt1")
    assert sent["json"] == {"summary": "New title"}
    assert event.title == "New title"


def test_update_event_can_set_multiple_fields(monkeypatch):
    provider, _ = make_provider(token=_fresh_token())
    sent = {}

    def fake_request(method, url, headers, timeout, **kwargs):
        sent["json"] = kwargs.get("json")
        return httpx.Response(
            200,
            json={"id": "evt1", "summary": "t", "start": {"dateTime": "s2"}, "end": {"dateTime": "e2"}},
            request=httpx.Request(method, url),
        )

    monkeypatch.setattr(provider_module.httpx, "request", fake_request)

    provider.update_event("evt1", start="s2", end="e2", location="Room 1")

    assert sent["json"] == {
        "start": {"dateTime": "s2"},
        "end": {"dateTime": "e2"},
        "location": "Room 1",
    }


# --- delete_event -----------------------------------------------------


def test_delete_event_sends_delete_request(monkeypatch):
    provider, _ = make_provider(token=_fresh_token())
    sent = {}

    def fake_request(method, url, headers, timeout, **kwargs):
        sent["method"] = method
        sent["url"] = url
        return httpx.Response(204, request=httpx.Request(method, url))

    monkeypatch.setattr(provider_module.httpx, "request", fake_request)

    provider.delete_event("evt1")

    assert sent["method"] == "DELETE"
    assert sent["url"].endswith("/events/evt1")


def test_delete_event_raises_on_http_error(monkeypatch):
    provider, _ = make_provider(token=_fresh_token())

    def fake_request(method, url, headers, timeout, **kwargs):
        raise httpx.ConnectError("nope")

    monkeypatch.setattr(provider_module.httpx, "request", fake_request)

    with pytest.raises(GoogleCalendarError):
        provider.delete_event("evt1")
