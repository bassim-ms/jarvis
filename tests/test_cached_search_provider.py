"""
Tests for CachedSearchProvider: proves it checks a SearchCache before
delegating to a wrapped SearchProvider, normalizes cache keys correctly,
caches successful results (including empty ones) but never failures, and
lets SearchProviderError propagate unchanged on a miss. Uses a fake
in-memory SearchCache and a fake SearchProvider - no SQLite or real network
involved.
"""

import pytest

from app.search.base import SearchProvider, SearchProviderError, SearchResult
from app.search.cache import SearchCache
from app.search.cached import CachedSearchProvider


class FakeCache(SearchCache):
    """Plain dict-backed SearchCache with no TTL - just enough to prove
    CachedSearchProvider's own hit/miss/key logic in isolation."""

    def __init__(self):
        self.store: dict[str, list[SearchResult]] = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, results):
        self.store[key] = results


class FakeProvider(SearchProvider):
    def __init__(self, results=None, error=None):
        self.results = results if results is not None else []
        self.error = error
        self.calls = 0

    def search(self, query, max_results=5):
        self.calls += 1
        if self.error:
            raise self.error
        return self.results


RESULT = SearchResult(title="Python", url="https://python.org", snippet="lang")


# --- Cache miss --------------------------------------------------------------


def test_cache_miss_calls_the_wrapped_provider():
    provider = FakeProvider(results=[RESULT])
    cached = CachedSearchProvider(provider, FakeCache())

    results = cached.search("python")

    assert results == [RESULT]
    assert provider.calls == 1


def test_successful_result_is_stored_in_the_cache():
    provider = FakeProvider(results=[RESULT])
    cache = FakeCache()
    cached = CachedSearchProvider(provider, cache)

    cached.search("python")

    assert cache.store == {"python::5": [RESULT]}


# --- Cache hit -----------------------------------------------------------------


def test_second_identical_search_uses_the_cache_not_the_provider():
    provider = FakeProvider(results=[RESULT])
    cached = CachedSearchProvider(provider, FakeCache())

    cached.search("python")
    results = cached.search("python")

    assert results == [RESULT]
    assert provider.calls == 1


# --- Cache key normalization -----------------------------------------------


def test_query_normalization_strips_and_lowercases_to_the_same_key():
    provider = FakeProvider(results=[RESULT])
    cached = CachedSearchProvider(provider, FakeCache())

    cached.search("  Python  ")
    cached.search("python")

    assert provider.calls == 1


def test_max_results_is_part_of_the_cache_key():
    provider = FakeProvider(results=[RESULT])
    cache = FakeCache()
    cached = CachedSearchProvider(provider, cache)

    cached.search("python", max_results=5)
    cached.search("python", max_results=10)

    assert provider.calls == 2
    assert set(cache.store) == {"python::5", "python::10"}


# --- Zero results --------------------------------------------------------------


def test_zero_results_is_cached_and_counts_as_a_hit():
    provider = FakeProvider(results=[])
    cached = CachedSearchProvider(provider, FakeCache())

    first = cached.search("no such thing")
    second = cached.search("no such thing")

    assert first == []
    assert second == []
    assert provider.calls == 1


# --- Failures are never cached, and propagate unchanged -----------------------


def test_provider_failure_is_not_cached():
    provider = FakeProvider(error=SearchProviderError("boom"))
    cache = FakeCache()
    cached = CachedSearchProvider(provider, cache)

    with pytest.raises(SearchProviderError):
        cached.search("python")

    assert cache.store == {}


def test_search_provider_error_propagates_unchanged():
    provider = FakeProvider(error=SearchProviderError("boom"))
    cached = CachedSearchProvider(provider, FakeCache())

    with pytest.raises(SearchProviderError, match="boom"):
        cached.search("python")


def test_failed_search_is_retried_against_the_provider_next_time():
    provider = FakeProvider(error=SearchProviderError("boom"))
    cached = CachedSearchProvider(provider, FakeCache())

    with pytest.raises(SearchProviderError):
        cached.search("python")

    provider.error = None
    provider.results = [RESULT]
    results = cached.search("python")

    assert results == [RESULT]
    assert provider.calls == 2
