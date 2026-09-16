"""
Search provider manager: ordered fallback across SearchProvider instances.

Implements SearchProvider itself, so anything that takes a SearchProvider
(WebSearchTool, or a future caller) can use a SearchProviderManager as a
drop-in provider without any change - it neither knows nor cares whether
it's talking to a single provider or a fallback chain of several.
"""

from app.search.base import SearchProvider, SearchProviderError, SearchResult


class SearchProviderManager(SearchProvider):
    """Tries providers in order, falling back to the next on failure.

    A provider only "fails" by raising SearchProviderError (network /
    provider errors) - a provider returning zero results is a valid,
    successful answer and is returned as-is, with no fallback to later
    providers. If every provider raises, one combined SearchProviderError
    is raised so callers (e.g. WebSearchTool) see a single clean error
    through the existing ToolResult error path, not a stack of them.
    """

    def __init__(self, providers: list[SearchProvider]):
        if not providers:
            raise ValueError("SearchProviderManager needs at least one provider.")
        self.providers = list(providers)

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        errors: list[str] = []
        for provider in self.providers:
            try:
                return provider.search(query, max_results=max_results)
            except SearchProviderError as exc:
                errors.append(f"{type(provider).__name__}: {exc}")

        raise SearchProviderError(
            "All search providers failed: " + "; ".join(errors)
        )
