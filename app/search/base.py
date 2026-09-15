"""
Search provider abstraction.

WebSearchTool (app/tools/web_search.py) depends only on this interface,
never on a specific search backend's API shape, so providers (SearXNG now;
Brave, Tavily, etc. later) can be added or swapped without touching the
tool or Jarvis Core.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


class SearchProviderError(RuntimeError):
    """Raised when a search provider can't be reached or returns something unusable."""


@dataclass
class SearchResult:
    """One search hit, in a shape common to every provider."""

    title: str
    url: str
    snippet: str


class SearchProvider(ABC):
    """A single web search backend."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Run a query and return standardized results."""
