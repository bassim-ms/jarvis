"""
Tests for the AI provider registry/factory (app/providers/factory.py):
selection by stable name, unknown-name handling, and the
configured/available metadata used for documentation. No HTTP calls are
made anywhere in this file.
"""

import pytest

from app.providers import factory
from app.providers.cloudflare import CloudflareProvider
from app.providers.freellmapi import FreeLLMAPIProvider
from app.providers.gemini import GeminiProvider
from app.providers.groq import GroqProvider
from app.providers.manager import AIProviderManager
from app.providers.openrouter import OpenRouterProvider

EXPECTED_PROVIDERS = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "openrouter": OpenRouterProvider,
    "cloudflare": CloudflareProvider,
    "freellmapi": FreeLLMAPIProvider,
}


@pytest.mark.parametrize("name, expected_type", EXPECTED_PROVIDERS.items())
def test_create_provider_returns_correct_type(name, expected_type):
    assert isinstance(factory.create_provider(name), expected_type)


def test_create_provider_raises_on_unknown_name():
    with pytest.raises(factory.UnknownProviderError, match="unknown-provider"):
        factory.create_provider("unknown-provider")


def test_create_provider_never_requires_an_api_key(monkeypatch):
    """Constructing any provider must not touch settings at all - only
    generate_reply() checks configuration (see each provider's own
    missing-key test)."""
    for name in EXPECTED_PROVIDERS:
        factory.create_provider(name)  # must not raise


def test_describe_providers_lists_all_registered_names():
    infos = factory.describe_providers()
    assert {info.name for info in infos} == set(EXPECTED_PROVIDERS)


def test_describe_providers_never_includes_key_values(monkeypatch):
    monkeypatch.setattr(factory.settings, "gemini_api_key", "super-secret-key")

    for info in factory.describe_providers():
        assert "super-secret-key" not in info.base_url
        assert all("super-secret-key" not in env for env in info.required_env)


def test_describe_providers_reflects_current_configuration(monkeypatch):
    monkeypatch.setattr(factory.settings, "gemini_api_key", "")
    monkeypatch.setattr(factory.settings, "groq_api_key", "fake-key")
    monkeypatch.setattr(factory.settings, "cloudflare_api_key", "fake-key")
    monkeypatch.setattr(factory.settings, "cloudflare_account_id", "")

    infos = {info.name: info for info in factory.describe_providers()}

    assert infos["gemini"].configured is False
    assert infos["groq"].configured is True
    # Cloudflare needs BOTH the key and the account id to count as configured.
    assert infos["cloudflare"].configured is False


def test_describe_providers_reports_cloudflare_requires_two_env_vars():
    infos = {info.name: info for info in factory.describe_providers()}
    assert set(infos["cloudflare"].required_env) == {
        "JARVIS_CLOUDFLARE_API_KEY",
        "JARVIS_CLOUDFLARE_ACCOUNT_ID",
    }


# --- FreeLLMAPI --------------------------------------------------------------


def test_describe_providers_reflects_freellmapi_configuration(monkeypatch):
    monkeypatch.setattr(factory.settings, "freellmapi_api_key", "")
    infos = {info.name: info for info in factory.describe_providers()}
    assert infos["freellmapi"].configured is False

    monkeypatch.setattr(factory.settings, "freellmapi_api_key", "fake-key")
    infos = {info.name: info for info in factory.describe_providers()}
    assert infos["freellmapi"].configured is True


def test_describe_providers_freellmapi_has_no_model_env():
    # FreeLLMAPI always uses model="auto" - there's no configurable model
    # env var (see app/providers/freellmapi.py), so ProviderInfo's default
    # "" is left as-is rather than naming a fake one.
    infos = {info.name: info for info in factory.describe_providers()}
    assert infos["freellmapi"].model_env == ""


def test_describe_providers_freellmapi_base_url_reflects_configured_setting(monkeypatch):
    monkeypatch.setattr(
        factory.settings, "freellmapi_base_url", "http://example.invalid:9000/v1"
    )
    infos = {info.name: info for info in factory.describe_providers()}
    assert infos["freellmapi"].base_url == "http://example.invalid:9000/v1/chat/completions"


def test_default_ai_provider_order_is_still_just_the_original_four():
    """Tests the code DEFAULT directly (via pydantic's own field
    introspection), not the live `settings` singleton - a real .env can
    (and often will) override JARVIS_AI_PROVIDER_ORDER, which must never
    make this test environment-dependent. FreeLLMAPI is deliberately NOT
    part of this default string - its inclusion is entirely driven by
    _effective_provider_order() checking JARVIS_FREELLMAPI_API_KEY, not by
    a static default (see the tests below)."""
    from app.config import Settings

    default_order = Settings.model_fields["ai_provider_order"].default
    assert default_order.split(",") == ["gemini", "groq", "openrouter", "cloudflare"]


# --- _effective_provider_order() / build_ai_provider_entries() ----------------
#
# The actual "only one manual step" contract: JARVIS_FREELLMAPI_API_KEY alone
# controls whether FreeLLMAPI is attempted, regardless of what
# JARVIS_AI_PROVIDER_ORDER says - including an explicit pre-existing .env
# override that predates FreeLLMAPI and never mentions it.


def test_effective_order_appends_freellmapi_when_key_configured(monkeypatch):
    # The exact string a pre-existing .env is likely to have explicitly -
    # predating FreeLLMAPI, never mentioning it.
    monkeypatch.setattr(
        factory.settings, "ai_provider_order", "gemini,groq,openrouter,cloudflare"
    )
    monkeypatch.setattr(factory.settings, "freellmapi_api_key", "fake-unified-key")

    entries = factory.build_ai_provider_entries()

    assert [entry.name for entry in entries] == [
        "gemini",
        "groq",
        "openrouter",
        "cloudflare",
        "freellmapi",
    ]
    assert isinstance(entries[-1].provider, FreeLLMAPIProvider)


def test_effective_order_unchanged_when_key_not_configured(monkeypatch):
    monkeypatch.setattr(
        factory.settings, "ai_provider_order", "gemini,groq,openrouter,cloudflare"
    )
    monkeypatch.setattr(factory.settings, "freellmapi_api_key", "")

    entries = factory.build_ai_provider_entries()

    assert [entry.name for entry in entries] == [
        "gemini",
        "groq",
        "openrouter",
        "cloudflare",
    ]


def test_effective_order_moves_an_explicitly_misplaced_freellmapi_to_the_end(monkeypatch):
    # Even if a user (or a previous call) put "freellmapi" somewhere else
    # in the string, it must never be attempted before any other
    # configured provider.
    monkeypatch.setattr(
        factory.settings, "ai_provider_order", "gemini,freellmapi,groq,cloudflare"
    )
    monkeypatch.setattr(factory.settings, "freellmapi_api_key", "fake-key")

    entries = factory.build_ai_provider_entries()

    assert [entry.name for entry in entries] == ["gemini", "groq", "cloudflare", "freellmapi"]


def test_effective_order_keeps_explicit_freellmapi_last_even_without_a_key(monkeypatch):
    # An explicit mention is never silently removed just because the key
    # is blank - it still fails gracefully at call time, same as every
    # other misconfigured provider (see tests/test_freellmapi_provider.py).
    monkeypatch.setattr(
        factory.settings, "ai_provider_order", "gemini,freellmapi,groq"
    )
    monkeypatch.setattr(factory.settings, "freellmapi_api_key", "")

    entries = factory.build_ai_provider_entries()

    assert [entry.name for entry in entries] == ["gemini", "groq", "freellmapi"]


def test_build_ai_provider_entries_includes_freellmapi_when_configured_last(monkeypatch):
    monkeypatch.setattr(
        factory.settings,
        "ai_provider_order",
        "gemini,groq,openrouter,cloudflare,freellmapi",
    )

    entries = factory.build_ai_provider_entries()

    assert [entry.name for entry in entries] == [
        "gemini",
        "groq",
        "openrouter",
        "cloudflare",
        "freellmapi",
    ]
    assert isinstance(entries[-1].provider, FreeLLMAPIProvider)


def test_create_default_ai_provider_manager_with_freellmapi_last(monkeypatch):
    monkeypatch.setattr(
        factory.settings,
        "ai_provider_order",
        "gemini,groq,openrouter,cloudflare,freellmapi",
    )

    manager = factory.create_default_ai_provider_manager()

    assert [health.name for health in manager.get_health()] == [
        "gemini",
        "groq",
        "openrouter",
        "cloudflare",
        "freellmapi",
    ]


def test_create_default_ai_provider_manager_auto_includes_freellmapi_from_key_alone(
    monkeypatch,
):
    """The end-to-end contract JarvisCore actually relies on: a pre-existing
    JARVIS_AI_PROVIDER_ORDER that never mentions FreeLLMAPI, plus only the
    API key configured, is enough for create_default_ai_provider_manager()
    (Core's actual default provider) to include it as the final entry - no
    second env var needed."""
    monkeypatch.setattr(
        factory.settings, "ai_provider_order", "gemini,groq,openrouter,cloudflare"
    )
    monkeypatch.setattr(factory.settings, "freellmapi_api_key", "fake-unified-key")

    manager = factory.create_default_ai_provider_manager()

    assert [health.name for health in manager.get_health()] == [
        "gemini",
        "groq",
        "openrouter",
        "cloudflare",
        "freellmapi",
    ]


def test_create_default_ai_provider_manager_omits_freellmapi_without_a_key(monkeypatch):
    monkeypatch.setattr(
        factory.settings, "ai_provider_order", "gemini,groq,openrouter,cloudflare"
    )
    monkeypatch.setattr(factory.settings, "freellmapi_api_key", "")

    manager = factory.create_default_ai_provider_manager()

    assert [health.name for health in manager.get_health()] == [
        "gemini",
        "groq",
        "openrouter",
        "cloudflare",
    ]


# --- build_ai_provider_entries / create_default_ai_provider_manager --------


def test_build_ai_provider_entries_follows_configured_order(monkeypatch):
    monkeypatch.setattr(factory.settings, "ai_provider_order", "cloudflare,gemini")

    entries = factory.build_ai_provider_entries()

    assert [entry.name for entry in entries] == ["cloudflare", "gemini"]
    assert isinstance(entries[0].provider, CloudflareProvider)
    assert isinstance(entries[1].provider, GeminiProvider)


def test_build_ai_provider_entries_raises_on_unknown_name(monkeypatch):
    monkeypatch.setattr(factory.settings, "ai_provider_order", "gemini,not-a-provider")

    with pytest.raises(factory.UnknownProviderError, match="not-a-provider"):
        factory.build_ai_provider_entries()


def test_create_default_ai_provider_manager_returns_a_manager(monkeypatch):
    monkeypatch.setattr(factory.settings, "ai_provider_order", "gemini,groq")

    manager = factory.create_default_ai_provider_manager()

    assert isinstance(manager, AIProviderManager)
    assert [health.name for health in manager.get_health()] == ["gemini", "groq"]


def test_create_default_ai_provider_manager_uses_configured_retry_and_cooldown(
    monkeypatch,
):
    monkeypatch.setattr(factory.settings, "ai_provider_order", "gemini")
    monkeypatch.setattr(factory.settings, "ai_provider_retry_count", 5)
    monkeypatch.setattr(factory.settings, "ai_provider_retry_delay_seconds", 1.5)
    monkeypatch.setattr(factory.settings, "ai_provider_cooldown_seconds", 120)

    manager = factory.create_default_ai_provider_manager()

    assert manager.retry_count == 5
    assert manager.retry_delay_seconds == 1.5
    assert manager.cooldown_seconds == 120
