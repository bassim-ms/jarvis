"""
Wiring tests for the OpenRouter provider: proves it uses OpenRouter's own
base URL/model/API key/error type from settings. The shared JSON-handling
logic itself is covered once in tests/test_openai_compatible_provider.py,
since OpenRouter shares that logic verbatim via OpenAICompatibleProvider.
All HTTP calls are monkeypatched; no real API call is made.
"""

import httpx
import pytest

from app.providers import _openai_compatible as oac
from app.providers import openrouter
from app.providers.openrouter import OPENROUTER_URL, OpenRouterError, OpenRouterProvider


def test_generate_reply_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(openrouter.settings, "openrouter_api_key", "")

    with pytest.raises(OpenRouterError, match="JARVIS_OPENROUTER_API_KEY"):
        OpenRouterProvider().generate_reply("hello")


def test_generate_reply_uses_configured_url_model_and_key(monkeypatch):
    monkeypatch.setattr(openrouter.settings, "openrouter_api_key", "fake-key")
    monkeypatch.setattr(openrouter.settings, "openrouter_model", "some/model")

    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hi from openrouter"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(oac.httpx, "post", fake_post)

    assert OpenRouterProvider().generate_reply("hello") == "hi from openrouter"
    assert captured["url"] == OPENROUTER_URL
    assert captured["headers"]["Authorization"] == "Bearer fake-key"
    assert captured["json"]["model"] == "some/model"
