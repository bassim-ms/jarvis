"""
xAI (Grok) provider, behind the AIProvider interface.

xAI exposes an OpenAI-compatible `/chat/completions` endpoint, so the
actual HTTP/JSON handling lives in OpenAICompatibleProvider
(app/providers/_openai_compatible.py) - this module only supplies xAI's
own base URL, API key, model, and error type.

Provider name (see app/providers/factory.py): "xai"
API base URL: https://api.x.ai/v1/chat/completions
Required env: JARVIS_XAI_API_KEY
Model config: JARVIS_XAI_MODEL (see app/config.py for the default)
"""

from app.config import settings
from app.providers._openai_compatible import OpenAICompatibleProvider
from app.providers.base import ProviderError

XAI_URL = "https://api.x.ai/v1/chat/completions"


class XAIError(ProviderError):
    """Raised when xAI can't be reached or returns something unusable."""


class XAIProvider(OpenAICompatibleProvider):
    """AIProvider implementation backed by xAI (Grok)."""

    display_name = "xAI"
    env_key_name = "JARVIS_XAI_API_KEY"
    error_cls = XAIError

    def _base_url(self) -> str:
        return XAI_URL

    def _api_key(self) -> str:
        return settings.xai_api_key

    def _model(self) -> str:
        return settings.xai_model
