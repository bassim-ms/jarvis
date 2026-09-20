"""
Wiring tests for the Mistral provider: proves it uses Mistral's own base
URL/model/API key/error type from settings. The shared JSON-handling
logic itself is covered once in tests/test_openai_compatible_provider.py,
since Mistral shares that logic verbatim via OpenAICompatibleProvider.
All HTTP calls are monkeypatched; no real API call is made.
"""

import httpx
import pytest

from app.providers import _openai_compatible as oac
from app.providers import mistral
from app.providers.mistral import MISTRAL_URL, MistralError, MistralProvider


def test_generate_reply_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(mistral.settings, "mistral_api_key", "")

    with pytest.raises(MistralError, match="JARVIS_MISTRAL_API_KEY"):
        MistralProvider().generate_reply("hello")


def test_generate_reply_uses_configured_url_model_and_key(monkeypatch):
    monkeypatch.setattr(mistral.settings, "mistral_api_key", "fake-key")
    monkeypatch.setattr(mistral.settings, "mistral_model", "mistral-test")

    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hi from mistral"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(oac.httpx, "post", fake_post)

    assert MistralProvider().generate_reply("hello") == "hi from mistral"
    assert captured["url"] == MISTRAL_URL
    assert captured["headers"]["Authorization"] == "Bearer fake-key"
    assert captured["json"]["model"] == "mistral-test"
