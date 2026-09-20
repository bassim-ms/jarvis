"""
Tests for the Cloudflare Workers AI provider. All HTTP calls are
monkeypatched, so no real API call is made. Cloudflare has its own
request/response shape (unlike Groq/OpenRouter/Mistral/xAI, which share
OpenAICompatibleProvider), so it gets its own full test suite rather than
just a wiring smoke test.
"""

import httpx
import pytest

from app.providers import cloudflare
from app.providers.cloudflare import CLOUDFLARE_URL, CloudflareError, CloudflareProvider


def test_generate_reply_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(cloudflare.settings, "cloudflare_api_key", "")
    monkeypatch.setattr(cloudflare.settings, "cloudflare_account_id", "acct123")

    with pytest.raises(CloudflareError, match="JARVIS_CLOUDFLARE_API_KEY"):
        CloudflareProvider().generate_reply("hello")


def test_generate_reply_raises_without_account_id(monkeypatch):
    monkeypatch.setattr(cloudflare.settings, "cloudflare_api_key", "fake-key")
    monkeypatch.setattr(cloudflare.settings, "cloudflare_account_id", "")

    with pytest.raises(CloudflareError, match="JARVIS_CLOUDFLARE_ACCOUNT_ID"):
        CloudflareProvider().generate_reply("hello")


def test_generate_reply_returns_response_text(monkeypatch):
    monkeypatch.setattr(cloudflare.settings, "cloudflare_api_key", "fake-key")
    monkeypatch.setattr(cloudflare.settings, "cloudflare_account_id", "acct123")
    monkeypatch.setattr(cloudflare.settings, "cloudflare_model", "@cf/test/model")

    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return httpx.Response(
            200,
            json={"success": True, "result": {"response": "hi from cloudflare"}},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(cloudflare.httpx, "post", fake_post)

    assert CloudflareProvider().generate_reply("hello") == "hi from cloudflare"
    assert captured["url"] == CLOUDFLARE_URL.format(
        account_id="acct123", model="@cf/test/model"
    )
    assert captured["headers"]["Authorization"] == "Bearer fake-key"
    assert captured["json"] == {"messages": [{"role": "user", "content": "hello"}]}


def test_generate_reply_raises_on_success_false(monkeypatch):
    """Cloudflare can return HTTP 200 with success: false and an errors
    list instead of a non-2xx status - this must still be treated as a
    failure, not silently returned as an empty/missing reply."""
    monkeypatch.setattr(cloudflare.settings, "cloudflare_api_key", "fake-key")
    monkeypatch.setattr(cloudflare.settings, "cloudflare_account_id", "acct123")

    def fake_post(url, headers, json, timeout):
        return httpx.Response(
            200,
            json={"success": False, "errors": [{"code": 5007, "message": "bad model"}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(cloudflare.httpx, "post", fake_post)

    with pytest.raises(CloudflareError, match="Cloudflare reported failure"):
        CloudflareProvider().generate_reply("hello")


def test_generate_reply_raises_on_unexpected_response(monkeypatch):
    monkeypatch.setattr(cloudflare.settings, "cloudflare_api_key", "fake-key")
    monkeypatch.setattr(cloudflare.settings, "cloudflare_account_id", "acct123")

    def fake_post(url, headers, json, timeout):
        return httpx.Response(
            200, json={"success": True}, request=httpx.Request("POST", url)
        )

    monkeypatch.setattr(cloudflare.httpx, "post", fake_post)

    with pytest.raises(CloudflareError, match="Unexpected Cloudflare response"):
        CloudflareProvider().generate_reply("hello")


def test_generate_reply_raises_on_connection_error(monkeypatch):
    monkeypatch.setattr(cloudflare.settings, "cloudflare_api_key", "fake-key")
    monkeypatch.setattr(cloudflare.settings, "cloudflare_account_id", "acct123")

    def fake_post(url, headers, json, timeout):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(cloudflare.httpx, "post", fake_post)

    with pytest.raises(CloudflareError, match="Failed to reach Cloudflare"):
        CloudflareProvider().generate_reply("hello")


def test_generate_reply_error_on_bad_status_never_leaks_api_key(monkeypatch):
    monkeypatch.setattr(cloudflare.settings, "cloudflare_api_key", "super-secret-key")
    monkeypatch.setattr(cloudflare.settings, "cloudflare_account_id", "acct123")

    def fake_post(url, headers, json, timeout):
        request = httpx.Request("POST", url, headers=headers)
        return httpx.Response(
            401,
            json={"errors": [{"code": 10000, "message": "authentication error"}]},
            request=request,
        )

    monkeypatch.setattr(cloudflare.httpx, "post", fake_post)

    with pytest.raises(CloudflareError) as excinfo:
        CloudflareProvider().generate_reply("hello")

    assert "super-secret-key" not in str(excinfo.value)
    assert "401" in str(excinfo.value)
