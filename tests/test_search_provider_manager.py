"""
Tests for SearchProviderManager: proves the generic fallback orchestration
(try in order, fall back on failure, zero results is success, one clean
error if everything fails) using lightweight fake providers - no real
SearXNG, Brave, or Tavily involved.
"""

import pytest

from app.search.base import SearchProvider, SearchProviderError, SearchResult
from app.search.manager import SearchProviderManager
from app.tools.base import PermissionLevel, ToolResultStatus
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool


class FakeProvider(SearchProvider):
    """Records whether/how it was called; succeeds, errors, or returns
    a fixed result list depending on how it's built."""

    def __init__(self, results=None, error=None):
        self.results = results if results is not None else []
        self.error = error
        self.calls = 0

    def search(self, query, max_results=5):
        self.calls += 1
        if self.error:
            raise self.error
        return self.results


RESULT = SearchResult(title="Python", url="https://python.org", snippet="lang")


# --- Construction ----------------------------------------------------------


def test_manager_requires_at_least_one_provider():
    with pytest.raises(ValueError):
        SearchProviderManager([])


# --- Ordering and short-circuiting -----------------------------------------


def test_first_provider_success_short_circuits_later_providers():
    first = FakeProvider(results=[RESULT])
    second = FakeProvider(results=[RESULT])
    manager = SearchProviderManager([first, second])

    results = manager.search("python")

    assert results == [RESULT]
    assert first.calls == 1
    assert second.calls == 0


def test_provider_order_is_respected():
    order = []

    class TrackingProvider(SearchProvider):
        def __init__(self, name):
            self.name = name

        def search(self, query, max_results=5):
            order.append(self.name)
            raise SearchProviderError(f"{self.name} failed")

    manager = SearchProviderManager(
        [TrackingProvider("a"), TrackingProvider("b"), TrackingProvider("c")]
    )

    with pytest.raises(SearchProviderError):
        manager.search("python")

    assert order == ["a", "b", "c"]


# --- Fallback on failure -----------------------------------------------------


def test_first_provider_failure_falls_back_to_second():
    first = FakeProvider(error=SearchProviderError("first down"))
    second = FakeProvider(results=[RESULT])
    manager = SearchProviderManager([first, second])

    results = manager.search("python")

    assert results == [RESULT]
    assert first.calls == 1
    assert second.calls == 1


def test_multiple_provider_failures_continue_to_next():
    first = FakeProvider(error=SearchProviderError("first down"))
    second = FakeProvider(error=SearchProviderError("second down"))
    third = FakeProvider(results=[RESULT])
    manager = SearchProviderManager([first, second, third])

    results = manager.search("python")

    assert results == [RESULT]
    assert first.calls == 1
    assert second.calls == 1
    assert third.calls == 1


def test_all_providers_failing_raises_one_clean_error():
    first = FakeProvider(error=SearchProviderError("first down"))
    second = FakeProvider(error=SearchProviderError("second down"))
    manager = SearchProviderManager([first, second])

    with pytest.raises(SearchProviderError) as excinfo:
        manager.search("python")

    message = str(excinfo.value)
    assert "first down" in message
    assert "second down" in message


# --- Zero results is success, not a failure ---------------------------------


def test_zero_results_from_first_provider_is_success_no_fallback():
    first = FakeProvider(results=[])
    second = FakeProvider(results=[RESULT])
    manager = SearchProviderManager([first, second])

    results = manager.search("python")

    assert results == []
    assert first.calls == 1
    assert second.calls == 0


# --- WebSearchTool through a manager, via the Tool Registry -----------------


def test_web_search_tool_falls_back_through_the_manager():
    first = FakeProvider(error=SearchProviderError("first down"))
    second = FakeProvider(results=[RESULT])
    tool = WebSearchTool(provider=SearchProviderManager([first, second]))

    result = tool.execute(query="python")

    assert result.status == ToolResultStatus.SUCCESS
    assert result.data == [
        {"title": "Python", "url": "https://python.org", "snippet": "lang"}
    ]
    assert first.calls == 1
    assert second.calls == 1


def test_web_search_tool_reports_one_clean_error_when_all_providers_fail():
    first = FakeProvider(error=SearchProviderError("first down"))
    second = FakeProvider(error=SearchProviderError("second down"))
    tool = WebSearchTool(provider=SearchProviderManager([first, second]))

    result = tool.execute(query="python")

    assert result.status == ToolResultStatus.ERROR
    assert "first down" in result.message
    assert "second down" in result.message


def test_web_search_permission_is_unchanged_when_backed_by_a_manager():
    """The manager is an implementation detail of the injected provider -
    it must not affect the tool's declared permission or the Registry's
    automatic (no-confirmation) execution of READ tools."""
    manager = SearchProviderManager([FakeProvider(results=[RESULT])])
    tool = WebSearchTool(provider=manager)
    registry = ToolRegistry()
    registry.register(tool)

    assert tool.permission == PermissionLevel.READ

    result = registry.execute("web_search", params={"query": "python"})

    assert result.status == ToolResultStatus.SUCCESS
