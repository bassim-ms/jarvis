"""
Tests for deterministic search modes: SearchMode/SearchModeConfig
(app/search/modes.py) and how WebSearchTool uses them to run one or more
search passes. Uses fake SearchProviders - no real network calls.
"""

import pytest

from app.search.base import SearchProvider, SearchProviderError, SearchResult
from app.search.cache import SearchCache
from app.search.cached import CachedSearchProvider
from app.search.manager import SearchProviderManager
from app.search.modes import SEARCH_MODE_CONFIGS, SearchMode
from app.tools.base import ToolResultStatus
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool


class CountingProvider(SearchProvider):
    """Records every call's max_results and returns one result per call,
    with a call-index-based URL so passes are distinguishable, countable,
    and - importantly - not deduplicated against each other by the M6.5
    result-quality layer (each call's result is a genuinely distinct URL)."""

    def __init__(self, error=None):
        self.error = error
        self.calls: list[int] = []

    def search(self, query, max_results=5):
        self.calls.append(max_results)
        if self.error:
            raise self.error
        return [
            SearchResult(
                title="r", url=f"https://example.com/{len(self.calls)}", snippet=""
            )
        ]


class FakeCache(SearchCache):
    def __init__(self):
        self.store: dict[str, list[SearchResult]] = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, results):
        self.store[key] = results


# --- SearchMode / SearchModeConfig ------------------------------------------


def test_all_four_search_modes_exist():
    assert {m.value for m in SearchMode} == {"fast", "normal", "research", "critical"}
    assert set(SEARCH_MODE_CONFIGS) == set(SearchMode)


def test_search_mode_constructs_from_its_string_value():
    assert SearchMode("fast") is SearchMode.FAST
    assert SearchMode("normal") is SearchMode.NORMAL
    assert SearchMode("research") is SearchMode.RESEARCH
    assert SearchMode("critical") is SearchMode.CRITICAL


# --- Mode selects pass count and default result limit -----------------------


def test_normal_is_the_default_mode():
    provider = CountingProvider()
    tool = WebSearchTool(provider=provider)

    tool.execute(query="python")

    assert provider.calls == [SEARCH_MODE_CONFIGS[SearchMode.NORMAL].result_limit]


def test_fast_performs_one_pass_with_its_configured_limit():
    provider = CountingProvider()
    tool = WebSearchTool(provider=provider)

    tool.execute(query="python", mode="fast")

    config = SEARCH_MODE_CONFIGS[SearchMode.FAST]
    assert config.passes == 1
    assert provider.calls == [config.result_limit]


def test_normal_performs_one_pass_with_its_configured_limit():
    provider = CountingProvider()
    tool = WebSearchTool(provider=provider)

    tool.execute(query="python", mode="normal")

    config = SEARCH_MODE_CONFIGS[SearchMode.NORMAL]
    assert config.passes == 1
    assert provider.calls == [config.result_limit]


def test_research_performs_multiple_deterministic_passes():
    """Since M6.6, the number of provider calls is driven by the
    SearchPlanner's distinct planned queries, not SearchModeConfig.passes
    (which stays in place for backward compatibility but no longer drives
    execution - see app/search/planner.py)."""
    provider = CountingProvider()
    tool = WebSearchTool(provider=provider)

    result = tool.execute(query="python", mode="research")

    config = SEARCH_MODE_CONFIGS[SearchMode.RESEARCH]
    query_count = len(tool.planner.plan("python", SearchMode.RESEARCH).queries)
    assert query_count > 1
    assert provider.calls == [config.result_limit] * query_count
    assert result.status == ToolResultStatus.SUCCESS
    assert len(result.data) == query_count  # one fake result per planned query


def test_critical_performs_multiple_deterministic_passes():
    provider = CountingProvider()
    tool = WebSearchTool(provider=provider)

    result = tool.execute(query="python", mode="critical")

    config = SEARCH_MODE_CONFIGS[SearchMode.CRITICAL]
    query_count = len(tool.planner.plan("python", SearchMode.CRITICAL).queries)
    assert query_count > 1
    assert provider.calls == [config.result_limit] * query_count
    assert result.status == ToolResultStatus.SUCCESS
    assert len(result.data) == query_count


def test_explicit_max_results_overrides_the_mode_default_limit():
    provider = CountingProvider()
    tool = WebSearchTool(provider=provider)

    tool.execute(query="python", mode="research", max_results=2)

    query_count = len(tool.planner.plan("python", SearchMode.RESEARCH).queries)
    assert provider.calls == [2] * query_count


# --- Backward compatibility --------------------------------------------------


def test_existing_callers_without_mode_behave_exactly_as_before():
    provider = CountingProvider()
    tool = WebSearchTool(provider=provider)

    result = tool.execute(query="python")

    assert provider.calls == [5]  # one pass, the pre-M6.4 default max_results
    assert result.status == ToolResultStatus.SUCCESS


def test_existing_callers_with_explicit_max_results_and_no_mode_are_unchanged():
    provider = CountingProvider()
    tool = WebSearchTool(provider=provider)

    tool.execute(query="python", max_results=2)

    assert provider.calls == [2]


# --- Invalid mode -------------------------------------------------------------


def test_invalid_mode_raises_on_direct_call():
    """No bespoke validation is added - an invalid mode fails the same way
    an invalid enum value always does in this codebase."""
    tool = WebSearchTool(provider=CountingProvider())

    with pytest.raises(ValueError):
        tool.execute(query="python", mode="bogus")


def test_invalid_mode_returns_error_result_through_registry():
    """Through the Tool Registry, the ValueError is caught by the
    Registry's existing generic exception handling (see
    test_tools.py::test_execute_isolates_exceptions_raised_by_a_tool), not
    by any new code in WebSearchTool."""
    registry = ToolRegistry()
    registry.register(WebSearchTool(provider=CountingProvider()))

    result = registry.execute(
        "web_search", params={"query": "python", "mode": "bogus"}
    )

    assert result.status == ToolResultStatus.ERROR


# --- Fallback and caching keep working underneath a mode ---------------------


def test_fallback_still_works_under_research_mode():
    failing = CountingProvider(error=SearchProviderError("down"))
    manager = SearchProviderManager([failing, CountingProvider()])
    tool = WebSearchTool(provider=manager)

    result = tool.execute(query="python", mode="research")

    assert result.status == ToolResultStatus.SUCCESS
    # each planned query calls manager.search() independently, and each of
    # those falls back from the failing provider to the working one
    query_count = len(tool.planner.plan("python", SearchMode.RESEARCH).queries)
    assert len(failing.calls) == query_count
    assert len(result.data) == query_count


def test_caching_works_independently_per_planned_query():
    """Since M6.6, RESEARCH/CRITICAL run distinct planned queries (see
    app/search/planner.py) rather than repeating the identical one, so
    each planned query is its own cache key: within a single execute()
    call, every planned query reaches the wrapped provider once (nothing
    to dedup-via-cache there anymore). Calling with the exact same
    query+mode again is where caching pays off - nothing new reaches the
    provider the second time."""
    provider = CountingProvider()
    cached = CachedSearchProvider(provider, FakeCache())
    tool = WebSearchTool(provider=cached)

    query_count = len(tool.planner.plan("python", SearchMode.RESEARCH).queries)

    first = tool.execute(query="python", mode="research")
    assert len(provider.calls) == query_count

    second = tool.execute(query="python", mode="research")
    assert len(provider.calls) == query_count  # no new calls - fully cached

    assert first.data == second.data
