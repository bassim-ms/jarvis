"""
Wiring tests for the Groq provider: proves it uses Groq's own base
URL/model/API key/error type from settings. The shared JSON-handling
logic itself (success/malformed-response/HTTP-error/secret-redaction) is
covered once in tests/test_openai_compatible_provider.py, since Groq
shares that logic verbatim via OpenAICompatibleProvider. All HTTP calls
are monkeypatched; no real API call is made.
"""

import httpx
import pytest

from app.providers import _openai_compatible as oac
from app.providers import groq
from app.providers.groq import GROQ_URL, GroqError, GroqProvider


def test_generate_reply_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(groq.settings, "groq_api_key", "")

    with pytest.raises(GroqError, match="JARVIS_GROQ_API_KEY"):
        GroqProvider().generate_reply("hello")


def test_generate_reply_uses_configured_url_model_and_key(monkeypatch):
    monkeypatch.setattr(groq.settings, "groq_api_key", "fake-key")
    monkeypatch.setattr(groq.settings, "groq_model", "llama-test-model")

    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hi from groq"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(oac.httpx, "post", fake_post)

    assert GroqProvider().generate_reply("hello") == "hi from groq"
    assert captured["url"] == GROQ_URL
    assert captured["headers"]["Authorization"] == "Bearer fake-key"
    assert captured["json"]["model"] == "llama-test-model"
