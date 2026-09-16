"""
Search result quality: deduplication by normalized URL.

Provider-agnostic by construction - dedup works purely off SearchResult.url
via app/search/url_utils.py, with no `if source == ...` branching and no
knowledge of which concrete SearchProvider produced a result. No ranking,
no credibility scoring, no AI - a plain, deterministic first-occurrence-wins
pass, safe to apply within a single provider call or across results already
combined from multiple search-mode passes (see app/tools/web_search.py).
"""

from app.search.base import SearchProvider, SearchResult
from app.search.url_utils import normalize_url


def deduplicate_results(results: list[SearchResult]) -> list[SearchResult]:
    """Remove duplicates by normalized URL, first occurrence wins.

    Preserves original ordering. Does no ranking, scoring, or merging of
    duplicate entries - just keeps the first one seen and drops the rest.
    """
    seen: set[str] = set()
    deduped: list[SearchResult] = []
    for result in results:
        key = normalize_url(result.url)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(result)
    return deduped


class ResultQualityProvider(SearchProvider):
    """Wraps a SearchProvider and deduplicates its results by normalized
    URL before returning them.

    Implements SearchProvider itself, so it composes with
    CachedSearchProvider/SearchProviderManager exactly like any other
    provider - WebSearchTool doesn't need to know it's there.
    """

    def __init__(self, provider: SearchProvider):
        self.provider = provider

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        results = self.provider.search(query, max_results=max_results)
        return deduplicate_results(results)
