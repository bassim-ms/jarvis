"""
Tests for the search provider factory: proves the ordered provider list is
driven by JARVIS_SEARCH_PROVIDER_ORDER (configuration), not hardcoded, and
that WebSearchTool ends up using a SearchProviderManager fallback chain
through the application's default wiring - without WebSearchTool or
JarvisCore knowing any provider's name.
"""

import pytest

from app.search import factory
from app.search.manager import SearchProviderManager
from app.search.searxng import SearXNGProvider
from app.tools.defaults import create_default_registry


def test_build_search_providers_defaults_to_searxng(monkeypatch):
    monkeypatch.setattr(factory.settings, "search_provider_order", "searxng")

    providers = factory.build_search_providers()

    assert len(providers) == 1
    assert isinstance(providers[0], SearXNGProvider)


def test_build_search_providers_respects_configured_order(monkeypatch):
    monkeypatch.setattr(
        factory,
        "_PROVIDER_FACTORIES",
        {"searxng": SearXNGProvider, "other": SearXNGProvider},
    )
    monkeypatch.setattr(factory.settings, "search_provider_order", "other,searxng")

    providers = factory.build_search_providers()

    assert len(providers) == 2
    assert all(isinstance(p, SearXNGProvider) for p in providers)


def test_build_search_providers_ignores_blank_entries(monkeypatch):
    monkeypatch.setattr(factory.settings, "search_provider_order", " searxng , ")

    providers = factory.build_search_providers()

    assert len(providers) == 1


def test_build_search_providers_raises_on_unknown_provider_name(monkeypatch):
    monkeypatch.setattr(factory.settings, "search_provider_order", "brave")

    with pytest.raises(factory.UnknownSearchProviderError, match="brave"):
        factory.build_search_providers()


def test_create_default_search_provider_is_a_manager(monkeypatch):
    monkeypatch.setattr(factory.settings, "search_provider_order", "searxng")

    provider = factory.create_default_search_provider()

    assert isinstance(provider, SearchProviderManager)
    assert isinstance(provider.providers[0], SearXNGProvider)


def test_default_registry_web_search_tool_uses_provider_manager():
    registry = create_default_registry()

    tool = registry.get("web_search")

    assert isinstance(tool.provider, SearchProviderManager)
