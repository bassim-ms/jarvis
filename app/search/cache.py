"""
Search cache abstraction.

CachedSearchProvider (app/search/cached.py) depends only on this interface,
never on SQLite or any other storage engine, so the cache backend can be
swapped later without touching CachedSearchProvider or the providers it
wraps.
"""

from abc import ABC, abstractmethod

from app.search.base import SearchResult


class SearchCache(ABC):
    """Stores and retrieves cached search results, keyed by a plain string.

    Knows nothing about queries, providers, or normalization - callers
    decide what a key means. TTL is a concern of the concrete
    implementation, not this interface: `get` simply returns None once an
    entry is no longer valid, the same as a plain miss.
    """

    @abstractmethod
    def get(self, key: str) -> list[SearchResult] | None:
        """Return cached results for `key`, or None on a miss or expired entry."""

    @abstractmethod
    def set(self, key: str, results: list[SearchResult]) -> None:
        """Store `results` under `key` for this cache's configured TTL."""
