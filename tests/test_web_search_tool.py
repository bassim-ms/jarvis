"""
Tests for WebSearchTool: proves it adapts a SearchProvider into the Tool
interface correctly - including through the Tool Registry's permission
handling - without depending on SearXNG or any real network call.
"""

import pytest

from app.search.base import SearchProvider, SearchProviderError, SearchResult
from app.search.modes import SearchMode
from app.search.response import SearchResponse
from app.search.searxng import SearXNGProvider
from app.tools.base import PermissionLevel, ToolResultStatus
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool


class FakeSearchProvider(SearchProvider):
    """Fake SearchProvider for testing WebSearchTool without real HTTP."""

    def __init__(self, results=None, error=None):
        self.results = results or []
        self.error = error
        self.last_query = None
        self.last_max_results = None

    def search(self, query, max_results=5):
        self.last_query = query
        self.last_max_results = max_results
        if self.error:
            raise self.error
        return self.results


# --- Tool declaration ----------------------------------------------------


def test_web_search_tool_declares_name_description_and_read_permission():
    tool = WebSearchTool(provider=FakeSearchProvider())

    assert tool.name == "web_search"
    assert tool.description
    assert tool.permission == PermissionLevel.READ
    assert tool.requirements.network is True


def test_web_search_tool_declares_query_max_results_and_mode_parameters():
    tool = WebSearchTool(provider=FakeSearchProvider())

    names = {param.name for param in tool.parameters}

    assert names == {"query", "max_results", "mode"}


def test_default_provider_is_searxng():
    assert isinstance(WebSearchTool().provider, SearXNGProvider)


# --- Execution -------------------------------------------------------------


def test_execute_returns_standardized_results():
    provider = FakeSearchProvider(
        results=[
            SearchResult(title="Python", url="https://python.org", snippet="lang")
        ]
    )
    tool = WebSearchTool(provider=provider)

    result = tool.execute(query="python")

    assert result.status == ToolResultStatus.SUCCESS
    assert result.data == {
        "query": "python",
        "mode": "normal",
        "queries_used": ["python"],
        "results": [
            {
                "title": "Python",
                "url": "https://python.org",
                "snippet": "lang",
                "source": "",
                "query": "",
            }
        ],
    }
    assert provider.last_query == "python"


def test_execute_passes_max_results_through_to_provider():
    provider = FakeSearchProvider(results=[])
    tool = WebSearchTool(provider=provider)

    tool.execute(query="python", max_results=2)

    assert provider.last_max_results == 2


def test_execute_returns_error_result_on_provider_failure():
    provider = FakeSearchProvider(error=SearchProviderError("boom"))
    tool = WebSearchTool(provider=provider)

    result = tool.execute(query="python")

    assert result.status == ToolResultStatus.ERROR
    assert "boom" in result.message


# --- The typed SearchResponse seam (M6.9) -----------------------------------


def test_search_returns_a_typed_search_response():
    provider = FakeSearchProvider(
        results=[
            SearchResult(title="Python", url="https://python.org", snippet="lang")
        ]
    )
    tool = WebSearchTool(provider=provider)

    response = tool.search("python")

    assert isinstance(response, SearchResponse)
    assert response.query == "python"
    assert response.mode is SearchMode.NORMAL
    assert response.queries_used == ["python"]
    assert response.results == [
        SearchResult(title="Python", url="https://python.org", snippet="lang")
    ]


def test_search_accepts_mode_as_a_string_like_execute_does():
    provider = FakeSearchProvider(results=[])
    tool = WebSearchTool(provider=provider)

    response = tool.search("python", mode="fast")

    assert response.mode is SearchMode.FAST
    assert provider.last_max_results == 3  # FAST's configured result limit


def test_search_raises_search_provider_error_instead_of_returning_a_tool_result():
    """Unlike execute(), search() does not catch SearchProviderError - the
    caller (e.g. JarvisCore's SEARCH path) decides how to handle it."""
    provider = FakeSearchProvider(error=SearchProviderError("boom"))
    tool = WebSearchTool(provider=provider)

    with pytest.raises(SearchProviderError, match="boom"):
        tool.search("python")


def test_execute_builds_its_dict_from_the_same_search_response():
    """execute() and search() must agree - execute() is a thin wrapper, not
    a second implementation of the search pipeline."""
    provider = FakeSearchProvider(
        results=[
            SearchResult(title="Python", url="https://python.org", snippet="lang")
        ]
    )
    tool = WebSearchTool(provider=provider)

    response = tool.search("python")
    result = tool.execute(query="python")

    assert result.data["query"] == response.query
    assert result.data["mode"] == response.mode.value
    assert result.data["queries_used"] == response.queries_used
    assert [r["title"] for r in result.data["results"]] == [
        r.title for r in response.results
    ]


# --- Through the Tool Registry --------------------------------------------


def test_web_search_runs_through_registry_automatically_as_a_read_tool():
    provider = FakeSearchProvider(
        results=[
            SearchResult(title="Python", url="https://python.org", snippet="lang")
        ]
    )
    registry = ToolRegistry()
    registry.register(WebSearchTool(provider=provider))

    result = registry.execute("web_search", params={"query": "python"})

    assert result.status == ToolResultStatus.SUCCESS
    assert result.data["results"][0]["title"] == "Python"


def test_web_search_registry_execute_needs_no_confirmation():
    """READ permission means no confirmed=True is ever required, unlike
    CONFIRM/DANGEROUS tools (see test_tools.py)."""
    provider = FakeSearchProvider(results=[])
    registry = ToolRegistry()
    registry.register(WebSearchTool(provider=provider))

    result = registry.execute("web_search", params={"query": "python"})

    assert result.status != ToolResultStatus.CONFIRMATION_REQUIRED


def test_web_search_registry_isolates_provider_failure():
    provider = FakeSearchProvider(error=SearchProviderError("network down"))
    registry = ToolRegistry()
    registry.register(WebSearchTool(provider=provider))

    result = registry.execute("web_search", params={"query": "python"})

    assert result.status == ToolResultStatus.ERROR
    assert "network down" in result.message
