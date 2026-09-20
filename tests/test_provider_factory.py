"""
Tests for the AI provider registry/factory (app/providers/factory.py):
selection by stable name, unknown-name handling, and the
configured/available metadata used for documentation. No HTTP calls are
made anywhere in this file.
"""

import pytest

from app.providers import factory
from app.providers.cloudflare import CloudflareProvider
from app.providers.gemini import GeminiProvider
from app.providers.groq import GroqProvider
from app.providers.mistral import MistralProvider
from app.providers.openrouter import OpenRouterProvider
from app.providers.xai import XAIProvider

EXPECTED_PROVIDERS = {
    "gemini": GeminiProvider,
    "groq": GroqProvider,
    "openrouter": OpenRouterProvider,
    "cloudflare": CloudflareProvider,
    "mistral": MistralProvider,
    "xai": XAIProvider,
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
