"""
Minimal SearXNG client, behind the SearchProvider interface.

Talks to a SearXNG instance's JSON search API directly over HTTP using
httpx (already a project dependency), so no extra SDK is needed.
"""

import httpx

from app.config import settings
from app.search.base import SearchProvider, SearchProviderError, SearchResult


class SearXNGError(SearchProviderError):
    """Raised when SearXNG can't be reached or returns something unusable."""


class SearXNGProvider(SearchProvider):
    """SearchProvider implementation backed by a SearXNG instance."""

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        if not settings.searxng_url:
            raise SearXNGError("JARVIS_SEARXNG_URL is not set.")

        url = f"{settings.searxng_url.rstrip('/')}/search"

        try:
            response = httpx.get(
                url, params={"q": query, "format": "json"}, timeout=30
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise SearXNGError(f"Failed to reach SearXNG: {exc}") from exc

        data = response.json()
        try:
            raw_results = data["results"]
        except (KeyError, TypeError) as exc:
            raise SearXNGError(f"Unexpected SearXNG response: {data}") from exc

        results = []
        for item in raw_results[:max_results]:
            try:
                results.append(
                    SearchResult(
                        title=item["title"],
                        url=item["url"],
                        snippet=item.get("content", ""),
                        source="searxng",
                    )
                )
            except (KeyError, TypeError) as exc:
                raise SearXNGError(
                    f"Unexpected SearXNG result item: {item}"
                ) from exc

        return results
