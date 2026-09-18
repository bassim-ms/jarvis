"""
Tests for GoogleOAuthClient/OAuthCallbackServer/CalendarAuthFlow
(app/calendar/oauth.py). httpx calls to Google's real endpoints are
monkeypatched - no real network call is ever made. OAuthCallbackServer is
tested against a real local (loopback-only) socket - not mocked - since
it's the actual redirect-catching mechanism the flow depends on; no real
browser and no real Google server are ever involved.
"""

import threading
import time
import urllib.parse
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from app.calendar import oauth as oauth_module
from app.calendar.base import OAuthToken, TokenStore
from app.calendar.oauth import (
    CalendarAuthFlow,
    GoogleOAuthClient,
    OAuthCallbackServer,
    OAuthError,
    needs_refresh,
)


class FakeTokenStore(TokenStore):
    def __init__(self):
        self.saved: OAuthToken | None = None

    def get(self):
        return self.saved

    def save(self, token):
        self.saved = token

    def clear(self):
        self.saved = None


def make_client(client_id="cid", client_secret="secret") -> GoogleOAuthClient:
    return GoogleOAuthClient(
        client_id=client_id, client_secret=client_secret, redirect_uri="http://localhost:8765/"
    )


# --- build_authorization_url ---------------------------------------------


def test_build_authorization_url_includes_required_params():
    url = make_client().build_authorization_url()

    assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    assert "client_id=cid" in url
    assert "redirect_uri=http%3A%2F%2Flocalhost%3A8765%2F" in url
    assert "response_type=code" in url
    assert "access_type=offline" in url
    assert "prompt=consent" in url


def test_build_authorization_url_requires_client_id():
    client = make_client(client_id="")

    with pytest.raises(OAuthError, match="JARVIS_GOOGLE_CLIENT_ID"):
        client.build_authorization_url()


# --- exchange_code -----------------------------------------------------


def test_exchange_code_returns_a_token(monkeypatch):
    def fake_post(url, data, timeout):
        assert data["grant_type"] == "authorization_code"
        assert data["code"] == "auth-code-123"
        return httpx.Response(
            200,
            json={
                "access_token": "access-1",
                "refresh_token": "refresh-1",
                "expires_in": 3600,
                "scope": "calendar.events",
                "token_type": "Bearer",
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(oauth_module.httpx, "post", fake_post)

    token = make_client().exchange_code("auth-code-123")

    assert token.access_token == "access-1"
    assert token.refresh_token == "refresh-1"
    assert token.scope == "calendar.events"


def test_exchange_code_raises_if_no_refresh_token_returned(monkeypatch):
    def fake_post(url, data, timeout):
        return httpx.Response(
            200,
            json={"access_token": "access-1", "expires_in": 3600},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(oauth_module.httpx, "post", fake_post)

    with pytest.raises(OAuthError, match="refresh_token"):
        make_client().exchange_code("auth-code-123")


def test_exchange_code_requires_client_credentials():
    client = make_client(client_id="", client_secret="")

    with pytest.raises(OAuthError):
        client.exchange_code("code")


def test_exchange_code_raises_on_http_error(monkeypatch):
    def fake_post(url, data, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(oauth_module.httpx, "post", fake_post)

    with pytest.raises(OAuthError, match="Failed to reach Google"):
        make_client().exchange_code("code")


def test_exchange_code_raises_on_unexpected_response(monkeypatch):
    def fake_post(url, data, timeout):
        return httpx.Response(200, json={"unexpected": True}, request=httpx.Request("POST", url))

    monkeypatch.setattr(oauth_module.httpx, "post", fake_post)

    with pytest.raises(OAuthError, match="Unexpected token response"):
        make_client().exchange_code("code")


# --- refresh -----------------------------------------------------------


def test_refresh_returns_a_new_access_token(monkeypatch):
    def fake_post(url, data, timeout):
        assert data["grant_type"] == "refresh_token"
        assert data["refresh_token"] == "old-refresh"
        return httpx.Response(
            200,
            json={"access_token": "access-2", "expires_in": 3600, "scope": "calendar.events"},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(oauth_module.httpx, "post", fake_post)

    token = make_client().refresh("old-refresh")

    assert token.access_token == "access-2"
    # Google didn't return a new refresh_token on this call - keep the old one.
    assert token.refresh_token == "old-refresh"


def test_refresh_keeps_new_refresh_token_if_google_rotates_it(monkeypatch):
    def fake_post(url, data, timeout):
        return httpx.Response(
            200,
            json={"access_token": "access-2", "refresh_token": "rotated", "expires_in": 3600},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(oauth_module.httpx, "post", fake_post)

    token = make_client().refresh("old-refresh")

    assert token.refresh_token == "rotated"


def test_refresh_raises_on_http_error(monkeypatch):
    def fake_post(url, data, timeout):
        raise httpx.ConnectError("nope")

    monkeypatch.setattr(oauth_module.httpx, "post", fake_post)

    with pytest.raises(OAuthError):
        make_client().refresh("old-refresh")


# --- needs_refresh -------------------------------------------------------


def _token_expiring_in(seconds: float) -> OAuthToken:
    return OAuthToken(
        access_token="a",
        refresh_token="r",
        expires_at=(datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat(),
        scope="s",
    )


def test_needs_refresh_true_for_an_expired_token():
    assert needs_refresh(_token_expiring_in(-10)) is True


def test_needs_refresh_true_within_the_skew_window():
    assert needs_refresh(_token_expiring_in(30), skew_seconds=60) is True


def test_needs_refresh_false_for_a_fresh_token():
    assert needs_refresh(_token_expiring_in(3600)) is False


# --- OAuthCallbackServer (real local socket, no network mocking) ---------


def test_callback_server_captures_the_code():
    server = OAuthCallbackServer(port=0)
    try:
        captured = {}

        def hit_it():
            httpx.get(f"http://localhost:{server.port}/?code=abc123", timeout=5)

        thread = threading.Thread(target=hit_it)
        thread.start()
        captured["code"] = server.wait_for_code(timeout=5)
        thread.join()

        assert captured["code"] == "abc123"
    finally:
        server.close()


def test_callback_server_raises_on_error_param():
    server = OAuthCallbackServer(port=0)
    try:
        thread = threading.Thread(
            target=lambda: httpx.get(
                f"http://localhost:{server.port}/?error=access_denied", timeout=5
            )
        )
        thread.start()
        with pytest.raises(OAuthError, match="access_denied"):
            server.wait_for_code(timeout=5)
        thread.join()
    finally:
        server.close()


def test_callback_server_times_out_with_no_request():
    server = OAuthCallbackServer(port=0)
    try:
        with pytest.raises(OAuthError, match="Timed out"):
            server.wait_for_code(timeout=0.2)
    finally:
        server.close()


# --- CalendarAuthFlow (end-to-end against the real local server) ---------


def test_calendar_auth_flow_runs_end_to_end(monkeypatch):
    def fake_post(url, data, timeout):
        assert data["grant_type"] == "authorization_code"
        assert data["code"] == "test-code-999"
        return httpx.Response(
            200,
            json={
                "access_token": "access-1",
                "refresh_token": "refresh-1",
                "expires_in": 3600,
                "scope": "calendar.events",
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(oauth_module.httpx, "post", fake_post)

    oauth_client = make_client()
    token_store = FakeTokenStore()

    def fake_browser_open(auth_url: str) -> None:
        # Simulate the browser completing Google's real consent screen
        # and landing back on our local redirect URI with a code - pull
        # the (now-correctly-bound-port) redirect_uri straight out of the
        # auth URL rather than guessing it.
        query = urllib.parse.parse_qs(urllib.parse.urlparse(auth_url).query)
        redirect_uri = query["redirect_uri"][0]
        threading.Thread(
            target=lambda: httpx.get(f"{redirect_uri}?code=test-code-999", timeout=5)
        ).start()

    flow = CalendarAuthFlow(
        oauth_client=oauth_client,
        token_store=token_store,
        redirect_port=0,
        browser_open=fake_browser_open,
    )

    token = flow.run(timeout=5)

    assert token.access_token == "access-1"
    assert token_store.saved is token


def test_calendar_auth_flow_raises_when_user_declines(monkeypatch):
    oauth_client = make_client()
    token_store = FakeTokenStore()

    def fake_browser_open(auth_url: str) -> None:
        query = urllib.parse.parse_qs(urllib.parse.urlparse(auth_url).query)
        redirect_uri = query["redirect_uri"][0]
        threading.Thread(
            target=lambda: httpx.get(f"{redirect_uri}?error=access_denied", timeout=5)
        ).start()

    flow = CalendarAuthFlow(
        oauth_client=oauth_client,
        token_store=token_store,
        redirect_port=0,
        browser_open=fake_browser_open,
    )

    with pytest.raises(OAuthError, match="access_denied"):
        flow.run(timeout=5)

    assert token_store.saved is None
