"""
FreeLLMAPI provider, behind the AIProvider interface.

FreeLLMAPI (https://github.com/tashfeenahmed/freellmapi) is a self-hosted,
OpenAI-compatible AI gateway the user runs locally (Docker Compose). It
does its OWN provider routing/fallback across whatever free-tier LLM
backends it's configured with - Jarvis treats it as exactly one more
AIProvider, the same way it treats Groq/OpenRouter, and never reproduces
FreeLLMAPI's internal routing: every request always sends model="auto" so
FreeLLMAPI makes that choice itself. A successful FreeLLMAPI response may
include an `X-Routed-Via` header identifying which backend actually
answered - this is diagnostic-only and deliberately not surfaced anywhere
here, since the AIProvider interface returns a plain string and this
project's own M14 review explicitly ruled out redesigning that interface
just to carry one optional debug header.

Reuses OpenAICompatibleProvider (app/providers/_openai_compatible.py) for
the shared request/response shape, exactly like Groq/OpenRouter - this
module only supplies FreeLLMAPI's own (locally-configurable) URL, API key,
and error type.

Provider name (see app/providers/factory.py): "freellmapi"
API base: JARVIS_FREELLMAPI_BASE_URL (default: http://localhost:3001/v1) -
unlike Groq/OpenRouter/Cloudflare's fixed vendor URLs, this is genuinely
configurable, since FreeLLMAPI is self-hosted and its address is the
user's own choice.
Required env: JARVIS_FREELLMAPI_API_KEY - FreeLLMAPI's own "unified" key
from its Dashboard -> Keys page. This is NOT the same thing as any
individual provider key FreeLLMAPI may hold internally (e.g. its own
Gemini key) - Jarvis never sees or needs those.
Model: always the literal string "auto" - FreeLLMAPI's own routing picks
the actual backend model. There is deliberately no JARVIS_FREELLMAPI_MODEL
setting; reproducing that choice in Jarvis would duplicate the exact
routing logic FreeLLMAPI already exists to own.
"""

from app.config import settings
from app.providers._openai_compatible import OpenAICompatibleProvider
from app.providers.base import ProviderError

FREELLMAPI_CHAT_COMPLETIONS_PATH = "/chat/completions"


class FreeLLMAPIError(ProviderError):
    """Raised when FreeLLMAPI can't be reached or returns something
    unusable."""


def freellmapi_url() -> str:
    """The full chat-completions URL for the currently configured
    FreeLLMAPI base URL - computed fresh each call (never cached), since
    JARVIS_FREELLMAPI_BASE_URL is user-configurable, unlike Groq/
    OpenRouter's fixed vendor URLs. Used by both FreeLLMAPIProvider itself
    and factory.py's describe_providers() (non-secret metadata only - see
    ProviderInfo)."""
    return settings.freellmapi_base_url.rstrip("/") + FREELLMAPI_CHAT_COMPLETIONS_PATH


class FreeLLMAPIProvider(OpenAICompatibleProvider):
    """AIProvider implementation backed by a self-hosted FreeLLMAPI
    gateway. Always requests model="auto" - see module docstring."""

    display_name = "FreeLLMAPI"
    env_key_name = "JARVIS_FREELLMAPI_API_KEY"
    error_cls = FreeLLMAPIError

    def _base_url(self) -> str:
        return freellmapi_url()

    def _api_key(self) -> str:
        return settings.freellmapi_api_key

    def _model(self) -> str:
        return "auto"
