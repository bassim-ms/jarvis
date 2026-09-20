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


def test_generate_reply_error_on_bad_status_never_leaks_api_key(monkeypatch):
    """Regression test: a non-2xx Gemini response must not leak the API key.

    httpx.HTTPStatusError's own message embeds the request URL, which
    includes the "key" query param GeminiProvider sends - GeminiError must
    build its message from the response status/body instead, or the key
    ends up in /chat's 502 response body (see app/main.py's ProviderError
    handling)."""
    monkeypatch.setattr(gemini.settings, "gemini_api_key", "super-secret-key")

    def fake_post(url, params, json, timeout):
        request = httpx.Request("POST", url, params=params)
        return httpx.Response(
            401,
            json={"error": {"message": "invalid credentials"}},
            request=request,
        )

    monkeypatch.setattr(gemini.httpx, "post", fake_post)

    with pytest.raises(GeminiError) as excinfo:
        GeminiProvider().generate_reply("hello")

    assert "super-secret-key" not in str(excinfo.value)
    assert "401" in str(excinfo.value)
