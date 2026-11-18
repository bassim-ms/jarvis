"""
Shared HTTP/JSON handling for vendors exposing an OpenAI-compatible
`/chat/completions` endpoint - Groq and OpenRouter both document this
exact request shape (`{"model", "messages": [{"role", "content"}]}`) and
response shape (`choices[0].message.content`), so this base class
implements that shape once instead of twice. Kept as a shared base
(rather than inlined into groq.py/openrouter.py) because it's also the
natural extension point if a future OpenAI-compatible provider is added
later (e.g. a self-hosted local model reachable over the local network
frequently exposes this same shape).

This is a private module (leading underscore): it is not itself a
provider and is never selected by name in app/providers/factory.py. Each
vendor still gets its own module (groq.py, openrouter.py) with its own
error class, base URL, and settings fields - "keep the base URL/model
configuration provider-specific rather than hardcoding assumptions into
Core" (M11 requirement) means Core/this base class never hardcode a
specific vendor's URL or model; only the small, distinct subclasses below
do, one per vendor.

Gemini and Cloudflare Workers AI are NOT built on this: both have their
own, meaningfully different request/response JSON (see
app/providers/gemini.py and app/providers/cloudflare.py), so sharing this
base class for them would mean branching inside it - worse than just
implementing AIProvider directly, which is what they do.
"""

import httpx

from app.config import settings
from app.providers.base import AIProvider, ProviderError


class OpenAICompatibleProvider(AIProvider):
    """Base for a vendor's OpenAI-compatible chat/completions endpoint.

    A subclass sets three class attributes (`display_name`, `env_key_name`,
    `error_cls`) and implements the three single-line hooks below to pull
    its own base URL/API key/model out of `app.config.settings` - nothing
    else. `error_cls` must be a ProviderError subclass; using the
    subclass's own error type (not a generic ProviderError) keeps the
    per-vendor error typing the same as GeminiError's, for callers that
    care which provider failed.
    """

    display_name: str
    env_key_name: str
    error_cls: type[ProviderError]

    def _base_url(self) -> str:
        raise NotImplementedError

    def _api_key(self) -> str:
        raise NotImplementedError

    def _model(self) -> str:
        raise NotImplementedError

    def generate_reply(self, message: str) -> str:
        api_key = self._api_key()
        if not api_key:
            raise self.error_cls(f"{self.env_key_name} is not set.")

        try:
            response = httpx.post(
                self._base_url(),
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": self._model(),
                    "messages": [{"role": "user", "content": message}],
                },
                timeout=settings.ai_provider_timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            # Build the message from status/body, never from str(exc) or
            # exc.request - the API key lives in the Authorization header
            # here (not the URL), but this mirrors GeminiProvider's same
            # discipline so no future header/param ever leaks by accident.
            raise self.error_cls(
                f"{self.display_name} returned "
                f"{exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise self.error_cls(
                f"Failed to reach {self.display_name}: {exc}"
            ) from exc

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise self.error_cls(
                f"Unexpected {self.display_name} response: {data}"
            ) from exc
