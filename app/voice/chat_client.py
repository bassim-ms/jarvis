"""
HTTP client for Rhasspy's existing `/chat` endpoint (app/main.py) - the
*only* way app/voice ever reaches Jarvis Core. Deliberately never imports
`app.core.core`: voice is just another client of the already-running
Rhasspy server, exactly like curl or a future UI would be, so this
milestone needed (and made) zero changes to Core, `/chat`, or its
request/response contract (`ChatRequest`/`ChatResponse` in app/main.py).

Talks over plain httpx (already a project dependency) - the same
"interface + typed error" convention every other HTTP-backed provider in
this codebase already uses (GeminiProvider, SearXNGProvider,
GoogleCalendarProvider).
"""

from typing import Any

import httpx


class ChatClientError(RuntimeError):
    """Raised when `/chat` can't be reached, times out, or returns
    something unusable - never a raw httpx exception past this
    boundary."""


class ChatClient:
    """Thin wrapper around `POST {base_url}/chat`."""

    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def send(self, message: str, conversation_id: str | None) -> tuple[str, str]:
        """Send `message` (optionally continuing `conversation_id`) and
        return `(reply, conversation_id)` - the conversation_id to pass
        into the *next* call, so a caller (app/voice/session.py) just
        threads this return value through without any logic of its own.
        Raises ChatClientError on any failure - connection refused (the
        server isn't running), a timeout, a non-2xx response, or an
        unexpected response shape."""
        try:
            response = httpx.post(
                f"{self.base_url}/chat",
                json={"message": message, "conversation_id": conversation_id},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ChatClientError(
                f"Failed to reach Rhasspy at {self.base_url}/chat: {exc}"
            ) from exc

        data: Any = response.json()
        try:
            return data["reply"], data["conversation_id"]
        except (KeyError, TypeError) as exc:
            raise ChatClientError(f"Unexpected /chat response: {data}") from exc
