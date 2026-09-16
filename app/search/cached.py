"""
Caching wrapper around a SearchProvider.

Implements SearchProvider itself, so WebSearchTool (or SearchProviderManager,
or anything else that takes a SearchProvider) can use a CachedSearchProvider
as a drop-in provider without any change. Caching logic lives entirely here,
not in SearXNGProvider or SearchProviderManager, so both stay cache-unaware.
"""

from app.search.base import SearchProvider, SearchResult
from app.search.cache import SearchCache


class CachedSearchProvider(SearchProvider):
    """Checks a SearchCache before delegating to a wrapped SearchProvider.

    Cache keys are the normalized query (stripped, lowercased) plus
    max_results, so different result counts are cached separately, and
    only an exact normalized match hits the cache - no embeddings or
    semantic similarity. A provider failure (SearchProviderError) is never
    cached and propagates unchanged, so a caller like
    SearchProviderManager still sees it as a normal failure on a miss.
    """

    def __init__(self, provider: SearchProvider, cache: SearchCache):
        self.provider = provider
        self.cache = cache

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        key = self._cache_key(query, max_results)

        cached = self.cache.get(key)
        if cached is not None:
            return cached

        results = self.provider.search(query, max_results=max_results)
        self.cache.set(key, results)
        return results

    @staticmethod
    def _cache_key(query: str, max_results: int) -> str:
        return f"{query.strip().lower()}::{max_results}"
