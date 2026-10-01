"""
HTTP client for Jarvis's chat endpoints (app/main.py) - the *only* way
app/voice ever reaches Jarvis Core. Deliberately never imports
`app.core.core`: voice is just another client of the already-running
Jarvis server, exactly like curl or a future UI would be.

Two modes, chosen by whether the client was given device credentials:

  * WITH credentials (how the PC voice client runs - see app/voice/factory.py): it is the registered,
    trusted PC. Messages go to the AUTHENTICATED `/device/chat` (Device Gateway, app/devices/), with the
    device id and credential sent only as HTTP HEADERS - never in the message body, the conversation, a
    URL, a log line or an error. The body is exactly what it always was: `{message, conversation_id}`. The
    gateway, not anything the user says, decides which device this is. `check_in()` calls the authenticated
    `/device/check-in`.
  * WITHOUT credentials: the legacy, anonymous `/chat` exactly as before (request shape unchanged). It is
    never a trusted device. The default factory never builds this mode: a PC without a credential fails
    with a clear local error rather than quietly talking as an anonymous client.

Authentication failures (missing/unknown/pending/revoked/wrong credential are indistinguishable by design)
raise `DeviceAuthError`; there is no fallback to `/chat`, and the client never re-registers itself.

Talks over plain httpx (already a project dependency) - the same
"interface + typed error" convention every other HTTP-backed provider in
this codebase already uses (GeminiProvider, SearXNGProvider,
GoogleCalendarProvider).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

DEVICE_ID_HEADER = "X-Jarvis-Device-Id"
DEVICE_TOKEN_HEADER = "X-Jarvis-Device-Token"
#: Before a send, re-check-in if the last one is older than this (well inside the registry's stale window),
#: so the PC reads ONLINE while it is in use - without any background thread.
CHECK_IN_REFRESH_SECONDS = 120.0


class ChatClientError(RuntimeError):
    """Raised when `/chat` can't be reached, times out, or returns
    something unusable - never a raw httpx exception past this
    boundary."""


class DeviceCredentialError(ChatClientError):
    """This PC has no usable device credential (missing or unreadable file). A local problem: nothing was sent."""


class DeviceAuthError(ChatClientError):
    """The server rejected this PC's device authentication. Nothing was sent anonymously instead."""


@dataclass(frozen=True)
class DeviceCredentials:
    """The PC's device id and secret. The secret is hidden from repr/str so it cannot reach a log or traceback."""

    device_id: str
    credential: str = field(repr=False)

    def __str__(self) -> str:
        return f"DeviceCredentials(device_id={self.device_id!r}, credential=<hidden>)"

    def headers(self) -> dict[str, str]:
        return {DEVICE_ID_HEADER: self.device_id, DEVICE_TOKEN_HEADER: self.credential}


def load_device_credentials(path: str) -> DeviceCredentials:
    """Read the credential file the admin command wrote (`python -m app.devices.admin register-pc`).
    Every problem is a DeviceCredentialError that names the file but never shows its contents."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        raise DeviceCredentialError(
            f"No device credential found at '{path}'. Register this PC first: python -m app.devices.admin register-pc"
        ) from None
    except (OSError, ValueError):
        raise DeviceCredentialError(f"The device credential file '{path}' could not be read.") from None
    device_id = data.get("device_id") if isinstance(data, dict) else None
    credential = data.get("credential") if isinstance(data, dict) else None
    if not isinstance(device_id, str) or not isinstance(credential, str) or not device_id or not credential:
        raise DeviceCredentialError(f"The device credential file '{path}' is not valid (expected device_id and credential).")
    return DeviceCredentials(device_id, credential)


class ChatClient:
    """Thin wrapper around the chat endpoints (see the module docstring for the two modes)."""

    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        credentials: DeviceCredentials | None = None,
        check_in_refresh_seconds: float = CHECK_IN_REFRESH_SECONDS,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.credentials = credentials
        self.check_in_refresh_seconds = check_in_refresh_seconds
        self._last_check_in: float | None = None  # time.monotonic() of the last successful check-in

    @property
    def authenticated(self) -> bool:
        return self.credentials is not None

    def send(self, message: str, conversation_id: str | None) -> tuple[str, str]:
        """Send `message` (optionally continuing `conversation_id`) and
        return `(reply, conversation_id)` - the conversation_id to pass
        into the *next* call, so a caller (app/voice/session.py) just
        threads this return value through without any logic of its own.
        Raises ChatClientError on any failure - connection refused (the
        server isn't running), a timeout, a non-2xx response, or an
        unexpected response shape; DeviceAuthError if an authenticated
        client is rejected."""
        body = {"message": message, "conversation_id": conversation_id}
        if self.credentials is None:
            data = self._post("/chat", body)
            where = "/chat"
        else:
            self._refresh_check_in()
            data = self._post("/device/chat", body, headers=self.credentials.headers())
            where = "/device/chat"
        try:
            return data["reply"], data["conversation_id"]
        except (KeyError, TypeError) as exc:
            raise ChatClientError(f"Unexpected {where} response: {data}") from exc

    def check_in(self) -> dict:
        """An authenticated "I am alive" to `/device/check-in`: the registry marks this PC ONLINE and records when.
        Only for an authenticated client; raises DeviceAuthError if rejected (never silently ignored)."""
        if self.credentials is None:
            raise DeviceCredentialError("This client has no device credential, so it cannot check in.")
        data = self._post("/device/check-in", None, headers=self.credentials.headers())
        self._last_check_in = time.monotonic()
        return data

    def _refresh_check_in(self) -> None:
        if self._last_check_in is None or time.monotonic() - self._last_check_in >= self.check_in_refresh_seconds:
            self.check_in()

    def _post(self, path: str, body: dict | None, headers: dict[str, str] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        kwargs: dict[str, Any] = {"timeout": self.timeout}
        if body is not None:
            kwargs["json"] = body
        if headers is not None:
            kwargs["headers"] = headers  # only ever passed for an authenticated client
        try:
            response = httpx.post(url, **kwargs)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 401 and headers is not None:
                raise DeviceAuthError(
                    "Device authentication failed: the server did not accept this PC's credential "
                    "(it may have been revoked or replaced). Nothing was sent. Ask the admin to register this PC again."
                ) from None
            raise ChatClientError(f"Failed to reach Jarvis at {url}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ChatClientError(f"Failed to reach Jarvis at {url}: {exc}") from exc
        try:
            return response.json()
        except ValueError as exc:
            raise ChatClientError(f"Unexpected {path} response") from exc
