"""
Web search tool, adapting a SearchProvider (SearXNG for now; Brave, Tavily,
etc. later) into the Tool interface. The Tool Registry and Jarvis Core
depend only on Tool, never on SearchProvider or any concrete provider -
see app/search/base.py.

Supports deterministic search modes (see app/search/modes.py) that decide
how many search passes to run and how many results each pass asks for by
default. Modes never change which provider answers, or how caching/fallback
behave - those stay entirely inside the injected SearchProvider chain.
"""

from app.search.base import SearchProvider, SearchProviderError, SearchResult
from app.search.modes import SEARCH_MODE_CONFIGS, SearchMode
from app.search.searxng import SearXNGProvider
from app.tools.base import (
    PermissionLevel,
    Tool,
    ToolParameter,
    ToolRequirements,
    ToolResult,
)


class WebSearchTool(Tool):
    """Searches the web via an injected SearchProvider and returns
    standardized results (title, url, snippet)."""

    name = "web_search"
    description = (
        "Search the web and return a list of results (title, url, snippet). "
        "Optional mode: fast, normal (default), research, or critical."
    )
    permission = PermissionLevel.READ
    requirements = ToolRequirements(network=True)

    def __init__(self, provider: SearchProvider | None = None):
        self.provider = provider or SearXNGProvider()

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="query", type="string", description="Search query"),
            ToolParameter(
                name="max_results",
                type="number",
                description=(
                    "Maximum results per search pass (defaults to the "
                    "mode's configured limit)"
                ),
                required=False,
                default=None,
            ),
            ToolParameter(
                name="mode",
                type="string",
                description="Search mode: fast, normal, research, or critical",
                required=False,
                default=SearchMode.NORMAL.value,
            ),
        ]

    def execute(
        self,
        query: str,
        max_results: int | None = None,
        mode: SearchMode | str = SearchMode.NORMAL,
    ) -> ToolResult:
        if isinstance(mode, str):
            mode = SearchMode(mode)
        config = SEARCH_MODE_CONFIGS[mode]
        limit = max_results if max_results is not None else config.result_limit

        try:
            results: list[SearchResult] = []
            for _ in range(config.passes):
                results.extend(self.provider.search(query, max_results=limit))
        except SearchProviderError as exc:
            return ToolResult.error(str(exc))

        data = [
            {"title": r.title, "url": r.url, "snippet": r.snippet} for r in results
        ]
        return ToolResult.success(data=data)
