"""
Web search tool, adapting a SearchProvider (SearXNG for now; Brave, Tavily,
etc. later) into the Tool interface. The Tool Registry and Jarvis Core
depend only on Tool, never on SearchProvider or any concrete provider -
see app/search/base.py.
"""

from app.search.base import SearchProvider, SearchProviderError
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
        "Search the web and return a list of results (title, url, snippet)."
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
                description="Maximum number of results to return",
                required=False,
                default=5,
            ),
        ]

    def execute(self, query: str, max_results: int = 5) -> ToolResult:
        try:
            results = self.provider.search(query, max_results=max_results)
        except SearchProviderError as exc:
            return ToolResult.error(str(exc))

        data = [
            {"title": r.title, "url": r.url, "snippet": r.snippet} for r in results
        ]
        return ToolResult.success(data=data)
