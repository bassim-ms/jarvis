"""
Mistral provider, behind the AIProvider interface.

Mistral exposes an OpenAI-compatible `/chat/completions` endpoint, so the
actual HTTP/JSON handling lives in OpenAICompatibleProvider
(app/providers/_openai_compatible.py) - this module only supplies
Mistral's own base URL, API key, model, and error type.

Provider name (see app/providers/factory.py): "mistral"
API base URL: https://api.mistral.ai/v1/chat/completions
Required env: JARVIS_MISTRAL_API_KEY
Model config: JARVIS_MISTRAL_MODEL (see app/config.py for the default)
"""

from app.config import settings
from app.providers._openai_compatible import OpenAICompatibleProvider
from app.providers.base import ProviderError

MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"


class MistralError(ProviderError):
    """Raised when Mistral can't be reached or returns something unusable."""


class MistralProvider(OpenAICompatibleProvider):
    """AIProvider implementation backed by Mistral."""

    display_name = "Mistral"
    env_key_name = "JARVIS_MISTRAL_API_KEY"
    error_cls = MistralError

    def _base_url(self) -> str:
        return MISTRAL_URL

    def _api_key(self) -> str:
        return settings.mistral_api_key

    def _model(self) -> str:
        return settings.mistral_model
