"""
Web search tool, adapting a SearchProvider (SearXNG for now; Brave, Tavily,
etc. later) into the Tool interface. The Tool Registry and Jarvis Core
depend only on Tool, never on SearchProvider or any concrete provider -
see app/search/base.py.

Supports deterministic search modes (see app/search/modes.py) that pick a
result limit per query, and a deterministic SearchPlanner (see
app/search/planner.py) that decides WHICH queries to actually run for that
mode - RESEARCH/CRITICAL run more than one *distinct* query rather than
repeating the identical one. Neither modes nor the planner change which
provider answers, or how caching/fallback behave - those stay entirely
inside the injected SearchProvider chain.

Results from every planned query are deduplicated by normalized URL (see
app/search/quality.py) before the final result limit is applied, so a
multi-query mode's limit reflects unique results, not raw duplicate-laden
ones.

The successful ToolResult's `data` carries structured search context (see
app/search/response.py): the original query, resolved mode, every query
actually planned/run, and the final results - each still tagged with the
specific planned query that produced it (provenance stamped by the
concrete provider, e.g. SearXNGProvider, and preserved unchanged by every
layer above it).
"""

from app.search.base import SearchProvider, SearchProviderError, SearchResult
from app.search.modes import SEARCH_MODE_CONFIGS, SearchMode
from app.search.planner import DeterministicSearchPlanner, SearchPlanner
from app.search.quality import deduplicate_results
from app.search.response import SearchResponse
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

    def __init__(
        self,
        provider: SearchProvider | None = None,
        planner: SearchPlanner | None = None,
    ):
        self.provider = provider or SearXNGProvider()
        self.planner = planner or DeterministicSearchPlanner()

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

        plan = self.planner.plan(query, mode)

        try:
            results: list[SearchResult] = []
            for planned_query in plan.queries:
                results.extend(
                    self.provider.search(planned_query, max_results=limit)
                )
        except SearchProviderError as exc:
            return ToolResult.error(str(exc))

        # Deduplicate across every planned query before applying the final
        # limit, so duplicates (within one query's results, or repeated
        # across queries) never squeeze out otherwise-unique results.
        # deduplicate_results keeps the first occurrence untouched, so the
        # kept result's own `query` field - its provenance - survives too.
        combined = deduplicate_results(results)[:limit]

        response = SearchResponse(
            query=query,
            mode=mode,
            queries_used=plan.queries,
            results=combined,
        )

        data = {
            "query": response.query,
            "mode": response.mode.value,
            "queries_used": response.queries_used,
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "source": r.source,
                    "query": r.query,
                }
                for r in response.results
            ],
        }
        return ToolResult.success(data=data)
