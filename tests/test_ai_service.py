"""
Tests for the Gemini client. All HTTP calls are monkeypatched, so no real
API call is made.
"""

import httpx
import pytest

from app import ai_service
from app.ai_service import GeminiError, ask_gemini


def test_ask_gemini_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(ai_service.settings, "gemini_api_key", "")

    with pytest.raises(GeminiError, match="JARVIS_GEMINI_API_KEY"):
        ask_gemini("hello")


def test_ask_gemini_returns_text_from_response(monkeypatch):
    monkeypatch.setattr(ai_service.settings, "gemini_api_key", "fake-key")

    def fake_post(url, params, json, timeout):
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"parts": [{"text": "hi there"}]}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(ai_service.httpx, "post", fake_post)

    assert ask_gemini("hello") == "hi there"


def test_ask_gemini_raises_on_unexpected_response(monkeypatch):
    monkeypatch.setattr(ai_service.settings, "gemini_api_key", "fake-key")

    def fake_post(url, params, json, timeout):
        return httpx.Response(
            200, json={"unexpected": True}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(ai_service.httpx, "post", fake_post)

    with pytest.raises(GeminiError, match="Unexpected Gemini response"):
        ask_gemini("hello")


def test_ask_gemini_raises_on_http_error(monkeypatch):
    monkeypatch.setattr(ai_service.settings, "gemini_api_key", "fake-key")

    def fake_post(url, params, json, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(ai_service.httpx, "post", fake_post)

    with pytest.raises(GeminiError, match="Failed to reach Gemini"):
        ask_gemini("hello")
