"""
Tests for the Gemini provider. All HTTP calls are monkeypatched, so no real
API call is made.
"""

import httpx
import pytest

from app.providers import gemini
from app.providers.gemini import GeminiError, GeminiProvider


def test_generate_reply_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(gemini.settings, "gemini_api_key", "")

    with pytest.raises(GeminiError, match="JARVIS_GEMINI_API_KEY"):
        GeminiProvider().generate_reply("hello")


def test_generate_reply_returns_text_from_response(monkeypatch):
    monkeypatch.setattr(gemini.settings, "gemini_api_key", "fake-key")

    def fake_post(url, params, json, timeout):
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "hi there"}]}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(gemini.httpx, "post", fake_post)

    assert GeminiProvider().generate_reply("hello") == "hi there"


def test_generate_reply_raises_on_unexpected_response(monkeypatch):
    monkeypatch.setattr(gemini.settings, "gemini_api_key", "fake-key")

    def fake_post(url, params, json, timeout):
        return httpx.Response(
            200, json={"unexpected": True}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(gemini.httpx, "post", fake_post)

    with pytest.raises(GeminiError, match="Unexpected Gemini response"):
        GeminiProvider().generate_reply("hello")


def test_generate_reply_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(gemini.settings, "gemini_api_key", "fake-key")

    def fake_post(url, params, json, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(gemini.httpx, "post", fake_post)

    with pytest.raises(GeminiError, match="Failed to reach Gemini"):
        GeminiProvider().generate_reply("hello")
