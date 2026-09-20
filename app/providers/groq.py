"""
Groq provider, behind the AIProvider interface.

Groq exposes an OpenAI-compatible `/chat/completions` endpoint, so the
actual HTTP/JSON handling lives in OpenAICompatibleProvider
(app/providers/_openai_compatible.py) - this module only supplies Groq's
own base URL, API key, model, and error type.

Provider name (see app/providers/factory.py): "groq"
API base URL: https://api.groq.com/openai/v1/chat/completions
Required env: JARVIS_GROQ_API_KEY
Model config: JARVIS_GROQ_MODEL (see app/config.py for the default)
"""

from app.config import settings
from app.providers._openai_compatible import OpenAICompatibleProvider
from app.providers.base import ProviderError

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqError(ProviderError):
    """Raised when Groq can't be reached or returns something unusable."""


class GroqProvider(OpenAICompatibleProvider):
    """AIProvider implementation backed by Groq."""

    display_name = "Groq"
    env_key_name = "JARVIS_GROQ_API_KEY"
    error_cls = GroqError

    def _base_url(self) -> str:
        return GROQ_URL

    def _api_key(self) -> str:
        return settings.groq_api_key

    def _model(self) -> str:
        return settings.groq_model
