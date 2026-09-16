"""
Tests for WebSearchTool: proves it adapts a SearchProvider into the Tool
interface correctly - including through the Tool Registry's permission
handling - without depending on SearXNG or any real network call.
"""

from app.search.base import SearchProvider, SearchProviderError, SearchResult
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
    assert result.data == [
        {"title": "Python", "url": "https://python.org", "snippet": "lang"}
    ]
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
    assert result.data[0]["title"] == "Python"


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
