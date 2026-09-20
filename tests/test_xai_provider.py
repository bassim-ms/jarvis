"""
Wiring tests for the xAI provider: proves it uses xAI's own base
URL/model/API key/error type from settings. The shared JSON-handling
logic itself is covered once in tests/test_openai_compatible_provider.py,
since xAI shares that logic verbatim via OpenAICompatibleProvider. All
HTTP calls are monkeypatched; no real API call is made.
"""

import httpx
import pytest

from app.providers import _openai_compatible as oac
from app.providers import xai
from app.providers.xai import XAI_URL, XAIError, XAIProvider


def test_generate_reply_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(xai.settings, "xai_api_key", "")

    with pytest.raises(XAIError, match="JARVIS_XAI_API_KEY"):
        XAIProvider().generate_reply("hello")


def test_generate_reply_uses_configured_url_model_and_key(monkeypatch):
    monkeypatch.setattr(xai.settings, "xai_api_key", "fake-key")
    monkeypatch.setattr(xai.settings, "xai_model", "grok-test")

    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hi from xai"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(oac.httpx, "post", fake_post)

    assert XAIProvider().generate_reply("hello") == "hi from xai"
    assert captured["url"] == XAI_URL
    assert captured["headers"]["Authorization"] == "Bearer fake-key"
    assert captured["json"]["model"] == "grok-test"
