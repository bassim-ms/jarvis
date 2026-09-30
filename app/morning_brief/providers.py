"""
The one optional source that can already be served by something that exists: news, through the EXISTING search
provider chain (app/search/). It is a thin adapter - one bounded query, at most a handful of results, each kept
with the site it came from. It is never a fact source: a headline is shown as a headline, attributed, and the
brief never treats its wording as anything but text. Weather and business sources have NO implementation here
on purpose (no such integration exists yet); their interfaces are in base.py.
"""

from __future__ import annotations

from urllib.parse import urlparse

from app.morning_brief.base import NewsItem, NewsProvider

DEFAULT_QUERY = "top news headlines today"


class SearchNewsProvider(NewsProvider):
    def __init__(self, search_provider, query: str = DEFAULT_QUERY):
        self.search_provider = search_provider
        self.query = query

    def headlines(self, limit: int) -> list[NewsItem]:
        results = self.search_provider.search(self.query, max_results=max(1, min(limit, 5)))
        items: list[NewsItem] = []
        for result in results[:limit]:
            host = urlparse(result.url).netloc.removeprefix("www.")
            items.append(NewsItem(title=result.title, source=host or result.source, url=result.url))
        return items
