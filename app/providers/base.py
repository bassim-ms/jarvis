"""
AI provider abstraction.

Jarvis Core depends only on this interface, never on a specific AI vendor's
SDK or API shape. New providers get added by implementing AIProvider.

Since M11 (multi-provider foundation), several AIProvider implementations
exist side by side (Gemini, Groq, OpenRouter, Cloudflare Workers AI,
Mistral, xAI - see app/providers/factory.py for how they're selected by
name). The interface itself is unchanged: every provider still takes a
single prompt string and returns a single reply string, and still raises
ProviderError (or one of its per-provider subclasses, e.g. GeminiError) on
failure - that return type and error type ARE "the normalized response/
error" every provider adapts its own vendor-specific HTTP/JSON into.
M11 only adds adapters and a registry to construct them; JarvisCore still
builds its own default provider directly (GeminiProvider()) and has no
notion of provider selection, fallback, or rotation - that's a later
milestone.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class ProviderError(RuntimeError):
    """Raised when an AI provider can't be reached or returns something unusable."""


class AIProvider(ABC):
    """A single-turn text AI provider."""

    @abstractmethod
    def generate_reply(self, message: str) -> str:
        """Send a single user message to the provider and return its text reply."""


@dataclass
class ProviderInfo:
    """Static, non-secret metadata about one registered AIProvider - used
    only for documentation/introspection (see
    app/providers/factory.py:describe_providers), never by Core or any
    request path. `required_env`/`model_env` name environment variables,
    never their values, so this is always safe to log or display.
    `configured` is computed fresh from current settings each call, not
    cached, so it reflects whatever's in `.env` right now."""

    name: str
    display_name: str
    base_url: str
    required_env: list[str] = field(default_factory=list)
    model_env: str = ""
    configured: bool = False
