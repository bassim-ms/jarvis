"""
Tests for M6.7 search context + provenance:
- SearchResponse structure (app/search/response.py)
- SearchResult.query provenance flowing through the whole chain:
  provider -> ResultQualityProvider -> CachedSearchProvider ->
  SearchProviderManager -> WebSearchTool
- deduplication preserving the first occurrence's provenance
- WebSearchTool's structured ToolResult.data (query/mode/queries_used/
  results) across all four SearchModes

Uses fake SearchProviders throughout - no real network calls.
"""

from app.search.base import SearchProvider, SearchProviderError, SearchResult
from app.search.cache import SearchCache
from app.search.cached import CachedSearchProvider
from app.search.manager import SearchProviderManager
from app.search.modes import SearchMode
from app.search.planner import DeterministicSearchPlanner, SearchPlan, SearchPlanner
from app.search.quality import ResultQualityProvider, deduplicate_results
from app.search.response import SearchResponse
from app.tools.web_search import WebSearchTool


class QueryStampingProvider(SearchProvider):
    """Mimics a real provider (like SearXNGProvider): stamps the exact
    query it was asked onto every result it returns."""

    def __init__(self, letters_by_query: dict[str, list[str]]):
        self.letters_by_query = letters_by_query
        self.calls: list[str] = []

    def search(self, query, max_results=5):
        self.calls.append(query)
        return [
            SearchResult(
                title=letter,
                url=f"https://example.com/{letter}",
                snippet="",
                source="fake",
                query=query,
            )
            for letter in self.letters_by_query.get(query, [])
        ]


class FakeCache(SearchCache):
    def __init__(self):
        self.store: dict[str, list[SearchResult]] = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, results):
        self.store[key] = results


class ScriptedPlanner(SearchPlanner):
    """Always plans the same fixed list of queries, regardless of mode."""

    def __init__(self, queries: list[str]):
        self.queries = list(queries)

    def plan(self, query, mode):
        return SearchPlan(queries=self.queries)


# --- SearchResponse structure ----------------------------------------------


def test_search_response_holds_query_mode_queries_used_and_results():
    result = SearchResult(title="A", url="https://a.com", snippet="", query="python")

    response = SearchResponse(
        query="python",
        mode=SearchMode.RESEARCH,
        queries_used=["python", "python 2026"],
        results=[result],
    )

    assert response.query == "python"
    assert response.mode is SearchMode.RESEARCH
    assert response.queries_used == ["python", "python 2026"]
    assert response.results == [result]


# --- SearchResult.query field -----------------------------------------------


def test_search_result_query_defaults_to_empty_string():
    result = SearchResult(title="t", url="https://x.com", snippet="s")

    assert result.query == ""


def test_search_result_query_is_settable():
    result = SearchResult(title="t", url="https://x.com", snippet="s", query="python")

    assert result.query == "python"


# --- Provenance through the full chain --------------------------------------


def test_provenance_survives_result_quality_provider():
    provider = QueryStampingProvider({"python": ["A", "B"]})
    quality = ResultQualityProvider(provider)

    results = quality.search("python")

    assert [r.query for r in results] == ["python", "python"]


def test_provenance_survives_cached_search_provider_on_a_miss():
    provider = QueryStampingProvider({"python": ["A"]})
    cached = CachedSearchProvider(provider, FakeCache())

    results = cached.search("python")

    assert results[0].query == "python"


def test_provenance_survives_cached_search_provider_on_a_hit():
    """A cache hit returns the exact SearchResult objects stored on the
    miss, so their `query` provenance is whatever was true when they were
    first cached."""
    provider = QueryStampingProvider({"python": ["A"]})
    cached = CachedSearchProvider(provider, FakeCache())

    cached.search("python")  # populates the cache
    results = cached.search("python")  # served from cache

    assert results[0].query == "python"
    assert len(provider.calls) == 1  # second call never reached the provider


class _FailingProvider(SearchProvider):
    def search(self, query, max_results=5):
        raise SearchProviderError("down")


def test_provenance_survives_search_provider_manager_fallback():
    working = QueryStampingProvider({"python": ["A"]})
    manager = SearchProviderManager([_FailingProvider(), working])

    results = manager.search("python")

    assert results[0].query == "python"


def test_provenance_survives_the_full_default_style_chain():
    """WebSearchTool -> ResultQualityProvider -> CachedSearchProvider ->
    SearchProviderManager -> provider - the application's real default
    wiring shape (see app/tools/defaults.py) - provenance intact end to
    end."""
    provider = QueryStampingProvider({"python": ["A"]})
    chain = ResultQualityProvider(
        CachedSearchProvider(SearchProviderManager([provider]), FakeCache())
    )
    tool = WebSearchTool(provider=chain)

    result = tool.execute(query="python")

    assert result.data["results"][0]["query"] == "python"


# --- Multiple planned queries retain their own provenance -------------------


def test_multiple_planned_queries_retain_different_provenance():
    provider = QueryStampingProvider(
        {"python": ["A"], "python 2026": ["B"], "python student": ["C"]}
    )
    planner = ScriptedPlanner(["python", "python 2026", "python student"])
    tool = WebSearchTool(provider=provider, planner=planner)

    result = tool.execute(query="python", mode="research", max_results=10)

    queries_by_title = {r["title"]: r["query"] for r in result.data["results"]}
    assert queries_by_title == {
        "A": "python",
        "B": "python 2026",
        "C": "python student",
    }


# --- Dedup preserves first-occurrence provenance -----------------------------


def test_deduplicate_results_keeps_the_first_occurrences_provenance():
    """Q1 -> Result A, Q2 -> Result A (same URL) -> after dedup, keep
    Result A with query=Q1."""
    from_q1 = SearchResult(title="A", url="https://a.com", snippet="", query="Q1")
    from_q2 = SearchResult(title="A dup", url="https://a.com", snippet="", query="Q2")

    deduped = deduplicate_results([from_q1, from_q2])

    assert deduped == [from_q1]
    assert deduped[0].query == "Q1"


def test_web_search_tool_dedup_keeps_first_planned_querys_provenance():
    provider = QueryStampingProvider({"python": ["A"], "python 2026": ["A"]})
    planner = ScriptedPlanner(["python", "python 2026"])
    tool = WebSearchTool(provider=provider, planner=planner)

    result = tool.execute(query="python", mode="research", max_results=10)

    assert len(result.data["results"]) == 1  # same URL "A" from both queries
    assert result.data["results"][0]["query"] == "python"  # first planned query wins


# --- Final result limit still works ------------------------------------------


def test_final_limit_still_applies_with_provenance_tracking():
    provider = QueryStampingProvider({"python": ["A", "B", "C", "D", "E"]})
    tool = WebSearchTool(provider=provider)

    result = tool.execute(query="python", max_results=3)

    assert len(result.data["results"]) == 3


# --- FAST/NORMAL/RESEARCH/CRITICAL structured behavior ------------------------


def test_fast_mode_structured_output():
    provider = QueryStampingProvider({"python": ["A"]})
    tool = WebSearchTool(provider=provider)

    result = tool.execute(query="python", mode="fast")

    assert result.data["mode"] == "fast"
    assert result.data["queries_used"] == ["python"]


def test_normal_mode_structured_output():
    provider = QueryStampingProvider({"python": ["A"]})
    tool = WebSearchTool(provider=provider)

    result = tool.execute(query="python", mode="normal")

    assert result.data["mode"] == "normal"
    assert result.data["queries_used"] == ["python"]


def test_research_mode_structured_output_matches_the_real_planner():
    planner = DeterministicSearchPlanner(current_year=2026)
    provider = QueryStampingProvider(
        {"python": ["A"], "python 2026": ["B"], "python student": ["C"]}
    )
    tool = WebSearchTool(provider=provider, planner=planner)

    result = tool.execute(query="python", mode="research", max_results=10)

    assert result.data["mode"] == "research"
    assert result.data["queries_used"] == ["python", "python 2026", "python student"]
    assert {r["title"] for r in result.data["results"]} == {"A", "B", "C"}


def test_critical_mode_structured_output_matches_the_real_planner():
    planner = DeterministicSearchPlanner(current_year=2026)
    provider = QueryStampingProvider(
        {
            "python": ["A"],
            "python 2026": ["B"],
            "python student": ["C"],
            "python official": ["D"],
        }
    )
    tool = WebSearchTool(provider=provider, planner=planner)

    result = tool.execute(query="python", mode="critical", max_results=10)

    assert result.data["mode"] == "critical"
    assert result.data["queries_used"] == [
        "python",
        "python 2026",
        "python student",
        "python official",
    ]
    assert {r["title"] for r in result.data["results"]} == {"A", "B", "C", "D"}


def test_response_query_is_the_original_user_query_not_a_planned_one():
    planner = DeterministicSearchPlanner(current_year=2026)
    provider = QueryStampingProvider(
        {"python": ["A"], "python 2026": ["B"], "python student": ["C"]}
    )
    tool = WebSearchTool(provider=provider, planner=planner)

    result = tool.execute(query="python", mode="research", max_results=10)

    assert result.data["query"] == "python"
