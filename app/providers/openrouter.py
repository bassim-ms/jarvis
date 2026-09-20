"""
OpenRouter provider, behind the AIProvider interface.

OpenRouter exposes an OpenAI-compatible `/chat/completions` endpoint, so
the actual HTTP/JSON handling lives in OpenAICompatibleProvider
(app/providers/_openai_compatible.py) - this module only supplies
OpenRouter's own base URL, API key, model, and error type.

Provider name (see app/providers/factory.py): "openrouter"
API base URL: https://openrouter.ai/api/v1/chat/completions
Required env: JARVIS_OPENROUTER_API_KEY
Model config: JARVIS_OPENROUTER_MODEL (see app/config.py for the default -
OpenRouter routes a single model name across many underlying vendors, so
this is the one string that picks which one).
"""

from app.config import settings
from app.providers._openai_compatible import OpenAICompatibleProvider
from app.providers.base import ProviderError

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterError(ProviderError):
    """Raised when OpenRouter can't be reached or returns something unusable."""


class OpenRouterProvider(OpenAICompatibleProvider):
    """AIProvider implementation backed by OpenRouter."""

    display_name = "OpenRouter"
    env_key_name = "JARVIS_OPENROUTER_API_KEY"
    error_cls = OpenRouterError

    def _base_url(self) -> str:
        return OPENROUTER_URL

    def _api_key(self) -> str:
        return settings.openrouter_api_key

    def _model(self) -> str:
        return settings.openrouter_model
