"""GET /providers/status: the provider chain's live state, in the order every AI call tries it - and never a secret.

Also the key status behind it (app/providers/manager.py KeyCheck / key_status): a key is never reported as working because
it is non-empty - only a real successful answer makes it "authenticated", a real 401/403 makes it "rejected", and nothing
ever sends a test request to find out.
"""

import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app import main
from app.providers import _openai_compatible, freellmapi
from app.providers.base import AIProvider, ProviderError
from app.providers.freellmapi import FreeLLMAPIProvider
from app.providers.manager import (
    KEY_STATUS_TEXT,
    AIProviderEntry,
    AIProviderManager,
    AllProvidersFailedError,
    KeyCheck,
    call_structured,
    key_status,
)

FIXED_ORDER = ["groq", "openrouter", "gemini", "cloudflare", "freellmapi"]
KEY_SETTINGS = ("groq_api_key", "openrouter_api_key", "gemini_api_key", "cloudflare_api_key", "cloudflare_account_id",
                "freellmapi_api_key")


class _Unused(AIProvider):
    def generate_reply(self, message: str) -> str:
        raise AssertionError("the status endpoint must never call a provider")


@pytest.fixture
def client(monkeypatch):
    manager = AIProviderManager([AIProviderEntry(name, _Unused()) for name in FIXED_ORDER], retry_count=1,
                                cooldown_seconds=60.0, max_cooldown_seconds=86400.0)
    monkeypatch.setattr(main.core, "provider", manager)
    return TestClient(main.app, base_url="http://127.0.0.1"), manager


def test_the_status_lists_the_chain_in_the_fixed_order(client):
    http, _ = client

    body = http.get("/providers/status").json()

    assert body["order"] == FIXED_ORDER
    assert [p["position"] for p in body["providers"]] == [1, 2, 3, 4, 5]
    assert body["retry_count"] == 1 and body["max_cooldown_seconds"] == 86400.0
    assert {p["name"]: p["model"] for p in body["providers"]}["freellmapi"] == "auto"


def test_a_resting_and_a_failing_provider_are_shown(client, monkeypatch):
    http, manager = client
    monkeypatch.setattr(main.settings, "gemini_api_key", "test-value")
    gemini = manager.get_health()[2]
    gemini.cooldown_until = time.monotonic() + 3600
    gemini.healthy = False
    gemini.consecutive_failures = 3
    gemini.key_check = KeyCheck.UNAVAILABLE

    status = {p["name"]: p for p in http.get("/providers/status").json()["providers"]}

    assert 3590 <= status["gemini"]["resting_seconds"] <= 3600
    assert status["gemini"]["key_status"] == "unavailable" and status["gemini"]["consecutive_failures"] == 3
    assert status["groq"]["resting_seconds"] == 0 and status["groq"]["consecutive_failures"] == 0
    # The old field said "true" for a provider that had never been asked anything; it is gone.
    assert "healthy_last_time" not in status["groq"] and "configured" not in status["groq"]


def test_the_status_never_contains_a_key_or_an_account_id(client, monkeypatch):
    http, _ = client
    secrets = {
        "groq_api_key": "gsk_" + "S" * 40, "openrouter_api_key": "sk-or-v1-" + "s" * 40, "gemini_api_key": "AIza" + "q" * 35,
        "cloudflare_api_key": "cf" + "k" * 38, "cloudflare_account_id": "0123456789abcdef0123456789abcdef",
        "freellmapi_api_key": "freellmapi-" + "f" * 40,
    }
    for field, value in secrets.items():
        monkeypatch.setattr(main.settings, field, value)

    text = http.get("/providers/status").text

    assert all(value not in text for value in secrets.values())
    assert "http" not in text  # no URLs at all
    providers = http.get("/providers/status").json()["providers"]
    # Every key present - and NOT ONE reported as working, since none has been used.
    assert all(p["key_present"] and p["key_status"] == "present_not_verified" for p in providers)
    assert all(p["key_status_text"] == "key present, not verified" for p in providers)


def test_the_status_is_loopback_only_like_every_route(client):
    http, _ = client
    remote = TestClient(main.app, base_url="http://jarvis.example.com")

    assert remote.get("/providers/status").status_code == 403
    assert http.get("/providers/status").status_code == 200


# --- key status ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("key_present, key_check, expected", [
    (False, KeyCheck.NOT_CHECKED, "no_key"),
    (False, KeyCheck.AUTHENTICATED, "no_key"),  # a key removed since: no key now, whatever happened before
    (True, KeyCheck.NOT_CHECKED, "present_not_verified"),
    (True, KeyCheck.AUTHENTICATED, "authenticated"),
    (True, KeyCheck.REJECTED, "rejected"),
    (True, KeyCheck.UNAVAILABLE, "unavailable"),
])
def test_key_status(key_present, key_check, expected):
    assert key_status(key_present, key_check) == expected
    assert expected in KEY_STATUS_TEXT


def test_no_status_text_calls_a_key_valid_without_a_real_answer():
    assert "verified" in KEY_STATUS_TEXT["present_not_verified"] and "not" in KEY_STATUS_TEXT["present_not_verified"]
    assert all(word not in text for word in ("valid", "ok", "set") for name, text in KEY_STATUS_TEXT.items()
               if name != "authenticated")


class Scripted(AIProvider):
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)

    def generate_reply(self, message: str) -> str:
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def manager_of(*providers, **options):
    options.setdefault("sleep", lambda seconds: None)
    return AIProviderManager([AIProviderEntry(name, provider) for name, provider in zip(FIXED_ORDER, providers)], **options)


@pytest.mark.parametrize("error, expected", [
    (ProviderError("Groq returned 401: invalid api key"), KeyCheck.REJECTED),
    (ProviderError("Groq returned 403: forbidden"), KeyCheck.REJECTED),
    (ProviderError("JARVIS_GROQ_API_KEY is not set."), KeyCheck.REJECTED),
    (ProviderError("Failed to reach Groq: connection refused"), KeyCheck.UNAVAILABLE),
    (ProviderError("Groq request timed out"), KeyCheck.UNAVAILABLE),
    (ProviderError("Groq returned 429: Please try again in 450ms."), KeyCheck.UNAVAILABLE),
    (ProviderError("Groq returned 503: overloaded"), KeyCheck.UNAVAILABLE),
    (ProviderError("Groq returned 404: model not found"), KeyCheck.UNAVAILABLE),
    (RuntimeError("an adapter bug"), KeyCheck.UNAVAILABLE),
])
def test_a_real_failure_sets_what_it_says_about_the_key(error, expected):
    manager = manager_of(Scripted(error, error), Scripted("fine"), retry_count=1)

    assert manager.generate_reply("hi") == "fine"

    groq, openrouter = manager.get_health()[:2]
    assert groq.key_check is expected
    assert openrouter.key_check is KeyCheck.AUTHENTICATED  # it answered: it accepted its key


def test_nothing_is_checked_until_a_provider_is_really_used():
    manager = manager_of(Scripted("fine"), Scripted())

    assert all(health.key_check is KeyCheck.NOT_CHECKED for health in manager.get_health())
    manager.generate_reply("hi")
    assert [health.key_check for health in manager.get_health()] == [KeyCheck.AUTHENTICATED, KeyCheck.NOT_CHECKED]


def test_a_reply_that_fails_validation_still_proves_the_key_was_accepted():
    manager = manager_of(Scripted("not json"), Scripted('{"ok": true}'))

    def parse(reply):
        if not reply.startswith("{"):
            raise ValueError("not JSON")
        return reply

    assert call_structured(manager, "hi", parse) == '{"ok": true}'
    assert manager.get_health()[0].key_check is KeyCheck.AUTHENTICATED


def test_the_latest_real_answer_wins():
    manager = manager_of(Scripted(ProviderError("Groq returned 401: bad key"), "fine", ProviderError("Failed to reach Groq: x")),
                         Scripted("backup", "backup"))
    groq = manager.get_health()[0]

    manager.generate_reply("1")
    assert groq.key_check is KeyCheck.REJECTED
    manager.generate_reply("2")  # the provider now accepts the key (e.g. a temporary refusal on its side)
    assert groq.key_check is KeyCheck.AUTHENTICATED
    manager.generate_reply("3")
    assert groq.key_check is KeyCheck.UNAVAILABLE


def test_a_skipped_provider_keeps_its_status():
    clock = {"now": 0.0}
    manager = manager_of(Scripted(), Scripted("fine", "fine"), clock=lambda: clock["now"])
    groq = manager.get_health()[0]
    groq.cooldown_until = 100.0
    groq.key_check = KeyCheck.AUTHENTICATED

    manager.generate_reply("hi")
    manager.set_enabled("groq", False)
    manager.generate_reply("hi")

    assert groq.key_check is KeyCheck.AUTHENTICATED  # resting or disabled: not asked, so nothing new was learned


def test_freellmapi_rejecting_its_key_shows_as_rejected_end_to_end(client, monkeypatch):
    """The real case: FreeLLMAPI's desktop app answers 401 to the key in .env. Before Jarvis uses it: "present, not
    verified" (never "configured"/valid). After the first real use: "rejected" - via the real adapter and error classifier."""
    http, _ = client
    for name in KEY_SETTINGS:
        monkeypatch.setattr(main.settings, name, "")
    monkeypatch.setattr(freellmapi.settings, "freellmapi_api_key", "f" * 40)
    monkeypatch.setattr(freellmapi.settings, "freellmapi_base_url", "http://127.0.0.1:31415/v1")

    def refuse(url, headers, json, timeout):
        return httpx.Response(401, json={"error": "Invalid API key"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(_openai_compatible.httpx, "post", refuse)
    manager = AIProviderManager([AIProviderEntry("freellmapi", FreeLLMAPIProvider())], sleep=lambda seconds: None)
    monkeypatch.setattr(main.core, "provider", manager)

    before = http.get("/providers/status").json()["providers"][0]
    with pytest.raises(AllProvidersFailedError):
        manager.generate_reply("hi")
    after = http.get("/providers/status").json()["providers"][0]

    assert before["key_present"] is True and before["key_status"] == "present_not_verified"
    assert after["key_status"] == "rejected"
    assert after["key_status_text"] == "authentication rejected (the provider refused the key)"
    assert "f" * 40 not in http.get("/providers/status").text


def test_reading_the_status_never_calls_a_provider(client, monkeypatch):
    http, manager = client  # every provider is _Unused: calling one raises

    for _ in range(3):
        assert http.get("/providers/status").status_code == 200
    assert all(health.key_check is KeyCheck.NOT_CHECKED for health in manager.get_health())
