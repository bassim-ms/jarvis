"""
Tests for the shared OpenAI-compatible base class
(app/providers/_openai_compatible.py) that Groq/OpenRouter build on.
Exercised through a local dummy subclass rather than a real vendor, since
this base class's JSON handling is identical across both - each vendor's
own test file only needs to check its own URL/model/key wiring (see
tests/test_groq_provider.py etc.), not this logic again. All HTTP calls
are monkeypatched; no real network call is made.
"""

import httpx
import pytest

from app.providers import _openai_compatible as oac
from app.providers._openai_compatible import OpenAICompatibleProvider
from app.providers.base import ProviderError


class DummyError(ProviderError):
    """Dummy vendor's error type, for asserting the raised class is
    error_cls (not a bare ProviderError)."""


class DummyProvider(OpenAICompatibleProvider):
    display_name = "Dummy"
    env_key_name = "JARVIS_DUMMY_API_KEY"
    error_cls = DummyError

    def __init__(self, api_key: str = "fake-key"):
        self._api_key_value = api_key

    def _base_url(self) -> str:
        return "https://dummy.invalid/v1/chat/completions"

    def _api_key(self) -> str:
        return self._api_key_value

    def _model(self) -> str:
        return "dummy-model"


def test_generate_reply_raises_without_api_key():
    with pytest.raises(DummyError, match="JARVIS_DUMMY_API_KEY"):
        DummyProvider(api_key="").generate_reply("hello")


def test_generate_reply_returns_content_from_response(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "hi there"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(oac.httpx, "post", fake_post)

    assert DummyProvider().generate_reply("hello") == "hi there"
    assert captured["url"] == "https://dummy.invalid/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer fake-key"
    assert captured["json"] == {
        "model": "dummy-model",
        "messages": [{"role": "user", "content": "hello"}],
    }


def test_generate_reply_raises_on_unexpected_response(monkeypatch):
    def fake_post(url, headers, json, timeout):
        return httpx.Response(
            200, json={"unexpected": True}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(oac.httpx, "post", fake_post)

    with pytest.raises(DummyError, match="Unexpected Dummy response"):
        DummyProvider().generate_reply("hello")


def test_generate_reply_raises_on_connection_error(monkeypatch):
    def fake_post(url, headers, json, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(oac.httpx, "post", fake_post)

    with pytest.raises(DummyError, match="Failed to reach Dummy"):
        DummyProvider().generate_reply("hello")


def test_generate_reply_error_on_bad_status_never_leaks_api_key(monkeypatch):
    """A non-2xx response must not leak the API key - it lives in the
    Authorization header (never in this base class's error message), and
    the error is built from status/body, never from str(exc)/exc.request."""

    def fake_post(url, headers, json, timeout):
        request = httpx.Request("POST", url, headers=headers)
        return httpx.Response(
            401,
            json={"error": {"message": "invalid credentials"}},
            request=request,
        )

    monkeypatch.setattr(oac.httpx, "post", fake_post)

    with pytest.raises(DummyError) as excinfo:
        DummyProvider(api_key="super-secret-key").generate_reply("hello")

    assert "super-secret-key" not in str(excinfo.value)
    assert "401" in str(excinfo.value)
