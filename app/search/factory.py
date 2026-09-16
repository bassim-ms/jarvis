"""
Builds the application's default SearchProvider and SearchCache from
configuration.

This is the one place a search provider's name is ever checked against a
string. It turns the ordered, comma-separated names in
JARVIS_SEARCH_PROVIDER_ORDER into provider instances and wraps them in a
SearchProviderManager - so the provider order is configuration, not code,
and WebSearchTool / JarvisCore never see a provider name at all.
"""

from app.config import settings
from app.search.base import SearchProvider
from app.search.cache import SearchCache
from app.search.manager import SearchProviderManager
from app.search.searxng import SearXNGProvider
from app.search.sqlite_cache import SQLiteSearchCache

# Adding a new provider (Brave, Tavily, ...) means adding one entry here -
# no changes needed to SearchProviderManager, WebSearchTool, or JarvisCore.
_PROVIDER_FACTORIES: dict[str, type[SearchProvider]] = {
    "searxng": SearXNGProvider,
}


class UnknownSearchProviderError(ValueError):
    """Raised when JARVIS_SEARCH_PROVIDER_ORDER names a provider that isn't registered."""


def build_search_providers() -> list[SearchProvider]:
    """Instantiate the providers named in settings.search_provider_order, in order."""
    names = [
        name.strip()
        for name in settings.search_provider_order.split(",")
        if name.strip()
    ]

    providers = []
    for name in names:
        try:
            factory = _PROVIDER_FACTORIES[name]
        except KeyError:
            raise UnknownSearchProviderError(
                f"Unknown search provider '{name}' in JARVIS_SEARCH_PROVIDER_ORDER."
            ) from None
        providers.append(factory())

    return providers


def create_default_search_provider() -> SearchProvider:
    """Build the application's default SearchProvider: an ordered fallback
    chain over every provider named in JARVIS_SEARCH_PROVIDER_ORDER."""
    return SearchProviderManager(build_search_providers())


def create_default_search_cache() -> SearchCache:
    """Build the application's default SearchCache from configuration."""
    return SQLiteSearchCache(
        db_path=settings.search_cache_db_path,
        ttl_seconds=settings.search_cache_ttl,
    )
