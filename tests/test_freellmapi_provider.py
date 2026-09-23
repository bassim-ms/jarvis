"""
Wiring tests for the FreeLLMAPI provider: proves it uses FreeLLMAPI's own
(user-configurable) base URL, fixed model="auto", API key, and error type
from settings. The shared JSON-handling logic itself (success/malformed-
response/HTTP-error/secret-redaction, including 401/429/5xx/timeout/
connection-failure handling) is covered once in
tests/test_openai_compatible_provider.py, since FreeLLMAPI shares that
logic verbatim via OpenAICompatibleProvider, exactly like Groq/OpenRouter
(see tests/test_groq_provider.py) - this file only adds FreeLLMAPI-specific
wiring and its integration with M12's error classifier. All HTTP calls are
monkeypatched; no real FreeLLMAPI call is made.
"""

import httpx
import pytest

from app.providers import _openai_compatible as oac
from app.providers import freellmapi
from app.providers.freellmapi import FreeLLMAPIError, FreeLLMAPIProvider, freellmapi_url
from app.providers.manager import ErrorCategory, classify_error

# --- Configuration -------------------------------------------------------------


def test_generate_reply_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(freellmapi.settings, "freellmapi_api_key", "")

    with pytest.raises(FreeLLMAPIError, match="JARVIS_FREELLMAPI_API_KEY"):
        FreeLLMAPIProvider().generate_reply("hello")


def test_missing_key_does_not_raise_at_construction(monkeypatch):
    # Matches every other provider's contract (see
    # app/providers/factory.py:create_provider's own docstring) - building
    # a FreeLLMAPIProvider with no key configured must never crash Miles
    # at startup; only an actual generate_reply() call fails.
    monkeypatch.setattr(freellmapi.settings, "freellmapi_api_key", "")

    FreeLLMAPIProvider()  # must not raise


def test_default_base_url_is_localhost_3001():
    assert freellmapi_url() == "http://localhost:3001/v1/chat/completions"


def test_custom_base_url_is_used(monkeypatch):
    monkeypatch.setattr(
        freellmapi.settings, "freellmapi_base_url", "http://example.invalid:9000/v1"
    )

    assert freellmapi_url() == "http://example.invalid:9000/v1/chat/completions"


def test_custom_base_url_with_trailing_slash_is_normalized(monkeypatch):
    monkeypatch.setattr(
        freellmapi.settings, "freellmapi_base_url", "http://example.invalid:9000/v1/"
    )

    assert freellmapi_url() == "http://example.invalid:9000/v1/chat/completions"


# --- Request -------------------------------------------------------------------


def test_generate_reply_sends_the_expected_request(monkeypatch):
    monkeypatch.setattr(freellmapi.settings, "freellmapi_api_key", "fake-unified-key")
    monkeypatch.setattr(
        freellmapi.settings, "freellmapi_base_url", "http://localhost:3001/v1"
    )

    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Tokyo."}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(oac.httpx, "post", fake_post)

    reply = FreeLLMAPIProvider().generate_reply(
        "What is the capital of Japan? Answer in one sentence."
    )

    assert reply == "Tokyo."
    assert captured["url"] == "http://localhost:3001/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer fake-unified-key"
    assert captured["json"] == {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of Japan? Answer in one sentence.",
            }
        ],
    }


def test_generate_reply_always_sends_model_auto(monkeypatch):
    # There is deliberately no JARVIS_FREELLMAPI_MODEL setting (see
    # app/providers/freellmapi.py) - model is always the literal "auto",
    # regardless of anything else in settings.
    monkeypatch.setattr(freellmapi.settings, "freellmapi_api_key", "fake-key")
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "ok"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(oac.httpx, "post", fake_post)

    FreeLLMAPIProvider().generate_reply("hi")

    assert captured["json"]["model"] == "auto"


# --- Response ------------------------------------------------------------------


def test_generate_reply_parses_a_normal_openai_compatible_response(monkeypatch):
    monkeypatch.setattr(freellmapi.settings, "freellmapi_api_key", "fake-key")

    def fake_post(url, headers, json, timeout):
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "The capital of Japan is Tokyo."}}]
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(oac.httpx, "post", fake_post)

    assert FreeLLMAPIProvider().generate_reply("q") == "The capital of Japan is Tokyo."


def test_generate_reply_raises_on_malformed_response(monkeypatch):
    monkeypatch.setattr(freellmapi.settings, "freellmapi_api_key", "fake-key")

    def fake_post(url, headers, json, timeout):
        return httpx.Response(
            200, json={"unexpected": True}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(oac.httpx, "post", fake_post)

    with pytest.raises(FreeLLMAPIError, match="Unexpected FreeLLMAPI response"):
        FreeLLMAPIProvider().generate_reply("q")


def test_x_routed_via_header_is_present_but_ignored(monkeypatch):
    # FreeLLMAPI may include X-Routed-Via on a successful response - Miles
    # must succeed identically whether or not it's present (see module
    # docstring: deliberately not surfaced anywhere - the AIProvider
    # interface returns a plain string, and M14's review explicitly ruled
    # out redesigning it just to carry one optional debug header).
    monkeypatch.setattr(freellmapi.settings, "freellmapi_api_key", "fake-key")

    def fake_post(url, headers, json, timeout):
        return httpx.Response(
            200,
            headers={"X-Routed-Via": "groq/llama-3.1-70b"},
            json={"choices": [{"message": {"content": "ok"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(oac.httpx, "post", fake_post)

    assert FreeLLMAPIProvider().generate_reply("q") == "ok"


# --- Errors: integration with M12's shared classifier ---------------------
#
# The HTTP-error-handling CODE itself (401/429/5xx/timeout/connection
# failure -> a FreeLLMAPIError with a safe message) is the same shared
# OpenAICompatibleProvider logic already exercised generically in
# tests/test_openai_compatible_provider.py - duplicating that here would be
# exactly the "duplicate logic" the M13/M14 error-handling design avoids.
# These tests instead confirm the one genuinely new thing: that a
# FreeLLMAPIError's message (built the same way as every other provider's)
# is correctly classified by app/providers/manager.py:classify_error, so
# M12's retry/cooldown/fallback behaves the same for FreeLLMAPI as for
# every other provider.


def test_auth_error_message_is_classified_as_auth():
    assert (
        classify_error(FreeLLMAPIError("FreeLLMAPI returned 401: invalid unified key"))
        is ErrorCategory.AUTH
    )


def test_rate_limit_error_message_is_classified_as_rate_limit():
    assert (
        classify_error(FreeLLMAPIError("FreeLLMAPI returned 429: too many requests"))
        is ErrorCategory.RATE_LIMIT
    )


def test_server_error_message_is_classified_as_transient():
    assert (
        classify_error(FreeLLMAPIError("FreeLLMAPI returned 503: upstream unavailable"))
        is ErrorCategory.TRANSIENT
    )


def test_connection_error_message_is_classified_as_network():
    assert (
        classify_error(FreeLLMAPIError("Failed to reach FreeLLMAPI: connection refused"))
        is ErrorCategory.NETWORK
    )


def test_missing_key_error_message_is_classified_as_auth():
    assert (
        classify_error(FreeLLMAPIError("JARVIS_FREELLMAPI_API_KEY is not set."))
        is ErrorCategory.AUTH
    )


def test_generate_reply_error_never_leaks_the_unified_key(monkeypatch):
    monkeypatch.setattr(
        freellmapi.settings, "freellmapi_api_key", "super-secret-unified-key"
    )

    def fake_post(url, headers, json, timeout):
        request = httpx.Request("POST", url, headers=headers)
        return httpx.Response(401, json={"error": "invalid key"}, request=request)

    monkeypatch.setattr(oac.httpx, "post", fake_post)

    with pytest.raises(FreeLLMAPIError) as excinfo:
        FreeLLMAPIProvider().generate_reply("q")

    assert "super-secret-unified-key" not in str(excinfo.value)
