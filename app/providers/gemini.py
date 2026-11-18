"""
Minimal Gemini API client, behind the AIProvider interface.

Talks to Google's Gemini REST API directly over HTTP using httpx (already a
project dependency), so no extra SDK is needed for a single call.
"""

import httpx

from app.config import settings
from app.providers import usage
from app.providers.base import AIProvider, ProviderError

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "{model}:generateContent"
)


class GeminiError(ProviderError):
    """Raised when Gemini can't be reached or returns something unusable."""


class GeminiProvider(AIProvider):
    """AIProvider implementation backed by Google's Gemini API."""

    def generate_reply(self, message: str) -> str:
        if not settings.gemini_api_key:
            raise GeminiError("JARVIS_GEMINI_API_KEY is not set.")

        url = GEMINI_URL.format(model=settings.gemini_model)

        try:
            response = httpx.post(
                url,
                params={"key": settings.gemini_api_key},
                json={"contents": [{"parts": [{"text": message}]}]},
                timeout=settings.ai_provider_timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # exc's own message embeds exc.request.url, which includes the
            # "key" query param above - use status/body instead so the API
            # key never leaks into a GeminiError (and from there into
            # /chat's 502 response body).
            raise GeminiError(
                f"Gemini returned {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise GeminiError(f"Failed to reach Gemini: {exc}") from exc

        data = response.json()
        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise GeminiError(f"Unexpected Gemini response: {data}") from exc
        usage.note_gemini(data)  # token counts only, for the provider statistics (app/providers/stats.py)
        return text
