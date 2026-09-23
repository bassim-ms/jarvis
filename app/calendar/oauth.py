"""
Google OAuth 2.0 "Desktop app" flow (M9): a loopback (localhost) redirect,
no hosted server, no paid infra - authorize once interactively, then
refresh silently forever. Talks to Google's OAuth endpoints directly over
httpx (already a project dependency) - no Google SDK.

Three pieces, each independently testable:
- `GoogleOAuthClient` builds the consent URL and exchanges/refreshes
  tokens over plain HTTP (mockable exactly like GeminiProvider/
  SearXNGProvider already are in their own tests).
- `OAuthCallbackServer` is a minimal local HTTP server, bound only to the
  loopback interface, that captures the single redirect a real browser
  would make after the user grants consent - real, but 100% local, so
  tests exercise it with a real (non-network) HTTP request rather than
  mocking it away.
- `CalendarAuthFlow` orchestrates one full interactive authorization:
  open the consent URL (via an injectable `browser_open`, defaulting to
  the real `webbrowser.open`), wait for the redirect, exchange the code,
  and persist the token through `TokenStore`.

The refresh token this flow produces is a sensitive credential - never
logged, never printed, never included in an exception message anywhere
in this module (see also app/calendar/token_store.py).
"""

import http.server
import urllib.parse
import webbrowser
from datetime import datetime, timedelta, timezone
from typing import Callable

import httpx

from app.calendar.base import OAuthToken, TokenStore

AUTHORIZATION_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"

#: Read/write access to events only (never calendar settings/sharing) -
#: the minimal scope this milestone's Calendar tools need.
CALENDAR_EVENTS_SCOPE = "https://www.googleapis.com/auth/calendar.events"


class OAuthError(RuntimeError):
    """Raised when the OAuth authorization or token-refresh flow fails -
    the user declined consent, the local redirect timed out, or Google's
    token endpoint returned an error/unusable response. Never includes a
    token value in its message."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


class GoogleOAuthClient:
    """Builds the authorization URL and talks to Google's OAuth token
    endpoint. The "Desktop app" client type this targets doesn't treat
    `client_secret` as a server-side secret (it ships with the app, which
    runs entirely on the user's own machine) - it's still read only from
    config/env, never hard-coded, matching every other credential in this
    codebase."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        scope: str = CALENDAR_EVENTS_SCOPE,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.scope = scope

    def build_authorization_url(self) -> str:
        if not self.client_id:
            raise OAuthError("JARVIS_GOOGLE_CLIENT_ID is not set.")
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": self.scope,
            # offline + consent: guarantees a refresh_token comes back,
            # even if the user already authorized this app before.
            "access_type": "offline",
            "prompt": "consent",
        }
        return f"{AUTHORIZATION_URL}?{urllib.parse.urlencode(params)}"

    def exchange_code(self, code: str) -> OAuthToken:
        """Exchange a one-time authorization code for a token set. Raises
        OAuthError if Google doesn't return a refresh_token - with
        access_type=offline + prompt=consent it always should, and a
        token set with no way to refresh would fail silently much later
        instead."""
        token = self._request_token(
            {
                "code": code,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "redirect_uri": self.redirect_uri,
                "grant_type": "authorization_code",
            }
        )
        if not token.refresh_token:
            raise OAuthError(
                "Google did not return a refresh_token - re-run authorization."
            )
        return token

    def refresh(self, refresh_token: str) -> OAuthToken:
        """Exchange a refresh_token for a new access token. Google's
        token endpoint doesn't always return a new refresh_token on a
        refresh call - the caller's existing one is kept in that case."""
        token = self._request_token(
            {
                "refresh_token": refresh_token,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "grant_type": "refresh_token",
            }
        )
        if not token.refresh_token:
            token.refresh_token = refresh_token
        return token

    def _request_token(self, data: dict) -> OAuthToken:
        if not self.client_id or not self.client_secret:
            raise OAuthError(
                "JARVIS_GOOGLE_CLIENT_ID/JARVIS_GOOGLE_CLIENT_SECRET are not set."
            )
        try:
            response = httpx.post(TOKEN_URL, data=data, timeout=30)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise OAuthError(f"Failed to reach Google's token endpoint: {exc}") from exc

        payload = response.json()
        try:
            access_token = payload["access_token"]
            expires_in = payload["expires_in"]
        except KeyError as exc:
            raise OAuthError(f"Unexpected token response: missing {exc}") from exc

        expires_at = (_now() + timedelta(seconds=expires_in)).isoformat()
        return OAuthToken(
            access_token=access_token,
            refresh_token=payload.get("refresh_token", ""),
            expires_at=expires_at,
            scope=payload.get("scope", self.scope),
            token_type=payload.get("token_type", "Bearer"),
        )


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Captures exactly one GET's `code`/`error` query parameter onto the
    owning server, then responds with a static "you can close this tab"
    page. Never logs the request to stderr (the stdlib default) - a
    `code` value in a server access log is exactly the kind of incidental
    credential leak this module avoids everywhere else."""

    def do_GET(self) -> None:
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        self.server.oauth_code = params.get("code", [None])[0]
        self.server.oauth_error = params.get("error", [None])[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(
            b"<html><body>Rhasspy calendar authorization complete. "
            b"You can close this tab.</body></html>"
        )

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        pass


class OAuthCallbackServer:
    """A minimal local HTTP server that waits for exactly one OAuth
    redirect and captures its `code` (or `error`) query parameter, then
    stops. Binds only to the loopback interface (`localhost`) - never
    reachable from outside this machine. `port=0` lets the OS pick a free
    port (used by tests); the running application uses a fixed configured
    port so it matches the redirect URI registered in Google Cloud
    Console."""

    def __init__(self, port: int):
        self._server = http.server.HTTPServer(("localhost", port), _CallbackHandler)
        self._server.oauth_code = None
        self._server.oauth_error = None

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    def wait_for_code(self, timeout: float = 120) -> str:
        """Block until the redirect arrives (or `timeout` elapses) and
        return its `code`. Raises OAuthError if Google reported an error
        (e.g. the user declined consent) or nothing arrived in time."""
        self._server.timeout = timeout
        self._server.handle_request()
        if self._server.oauth_error:
            raise OAuthError(f"Google denied authorization: {self._server.oauth_error}")
        if not self._server.oauth_code:
            raise OAuthError("Timed out waiting for the OAuth redirect.")
        return self._server.oauth_code

    def close(self) -> None:
        self._server.server_close()


class CalendarAuthFlow:
    """Orchestrates one full interactive OAuth authorization: open the
    consent URL in a browser, wait for the local redirect, exchange the
    code, and persist the resulting token via `token_store`.

    The redirect URI's port is always taken from the actually-bound
    `OAuthCallbackServer` (not just the requested one), so this works
    correctly whether `redirect_port` is a fixed configured port (the
    running application) or `0`/OS-assigned (tests) - `oauth_client`'s
    `redirect_uri` is updated to match right before building the
    authorization URL.

    `browser_open` defaults to the real `webbrowser.open` but is
    injectable, so tests exercise the whole flow against a real local
    socket without a real browser or a real Google server.
    """

    def __init__(
        self,
        oauth_client: GoogleOAuthClient,
        token_store: TokenStore,
        redirect_port: int,
        browser_open: Callable[[str], object] = webbrowser.open,
    ):
        self.oauth_client = oauth_client
        self.token_store = token_store
        self.redirect_port = redirect_port
        self.browser_open = browser_open

    def run(self, timeout: float = 120) -> OAuthToken:
        server = OAuthCallbackServer(self.redirect_port)
        try:
            self.oauth_client.redirect_uri = f"http://localhost:{server.port}/"
            url = self.oauth_client.build_authorization_url()
            self.browser_open(url)
            code = server.wait_for_code(timeout=timeout)
        finally:
            server.close()

        token = self.oauth_client.exchange_code(code)
        self.token_store.save(token)
        return token


def needs_refresh(token: OAuthToken, skew_seconds: float = 60) -> bool:
    """Whether `token`'s access_token is expired or close enough to
    expiring (`skew_seconds`) that it should be refreshed before use -
    shared by GoogleCalendarProvider (app/calendar/google_provider.py) so
    a request never starts with a token that expires mid-flight."""
    expires_at = datetime.fromisoformat(token.expires_at)
    return _now() + timedelta(seconds=skew_seconds) >= expires_at
