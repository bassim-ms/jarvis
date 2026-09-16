"""
Tests for M6.5 search result quality:
- SearchResult's `source` metadata field (app/search/base.py)
- conservative URL normalization (app/search/url_utils.py)
- deduplication by normalized URL (app/search/quality.py)
- ResultQualityProvider, the SearchProvider wrapper that applies it
- WebSearchTool combining + deduplicating multiple search-mode passes and
  applying the result limit only after deduplication

Uses fake SearchProviders throughout - no real network calls.
"""

from app.search.base import SearchProvider, SearchResult
from app.search.planner import SearchPlan, SearchPlanner
from app.search.quality import ResultQualityProvider, deduplicate_results
from app.search.url_utils import normalize_url
from app.tools.base import ToolResultStatus
from app.tools.web_search import WebSearchTool


def r(letter: str, source: str = "") -> SearchResult:
    """Shorthand for a SearchResult with a unique, deterministic URL."""
    return SearchResult(
        title=letter, url=f"https://example.com/{letter}", snippet="", source=source
    )


class FakeProvider(SearchProvider):
    """Always returns the same fixed results list, regardless of how many
    times or with what arguments it's called."""

    def __init__(self, results=None):
        self.results = results if results is not None else []
        self.calls: list[tuple[str, int]] = []

    def search(self, query, max_results=5):
        self.calls.append((query, max_results))
        return self.results


class ScriptedProvider(SearchProvider):
    """Returns a different, pre-scripted results list on each successive
    call, in order - used to simulate distinct search-mode passes."""

    def __init__(self, responses: list[list[SearchResult]]):
        self.responses = list(responses)
        self.calls = 0

    def search(self, query, max_results=5):
        response = self.responses[self.calls]
        self.calls += 1
        return response


class FixedPlanner(SearchPlanner):
    """Always plans the same fixed list of queries, regardless of mode -
    decouples these dedup/combine tests from any particular SearchMode's
    actual planned-query count (see app/search/planner.py for that)."""

    def __init__(self, queries: list[str]):
        self.queries = list(queries)

    def plan(self, query, mode):
        return SearchPlan(queries=self.queries)


# --- SearchResult source metadata -------------------------------------------


def test_search_result_source_defaults_to_empty_string():
    result = SearchResult(title="t", url="https://x.com", snippet="s")

    assert result.source == ""


def test_search_result_source_is_settable():
    result = SearchResult(title="t", url="https://x.com", snippet="s", source="searxng")

    assert result.source == "searxng"


# --- URL normalization -------------------------------------------------------


def test_normalize_url_strips_surrounding_whitespace():
    assert normalize_url("  https://example.com/page  ") == normalize_url(
        "https://example.com/page"
    )


def test_normalize_url_lowercases_scheme_and_host():
    assert normalize_url("HTTPS://Example.COM/Page") == normalize_url(
        "https://example.com/Page"
    )


def test_normalize_url_strips_a_trailing_slash():
    assert normalize_url("https://example.com/page/") == normalize_url(
        "https://example.com/page"
    )


def test_normalize_url_treats_root_and_bare_domain_as_equal():
    assert normalize_url("https://example.com/") == normalize_url(
        "https://example.com"
    )


def test_normalize_url_removes_utm_tracking_params():
    tracked = "https://example.com/page?utm_source=x&utm_medium=y&utm_campaign=z"
    assert normalize_url(tracked) == normalize_url("https://example.com/page")


def test_normalize_url_removes_fbclid_and_gclid():
    assert normalize_url("https://example.com/page?fbclid=abc") == normalize_url(
        "https://example.com/page"
    )
    assert normalize_url("https://example.com/page?gclid=abc") == normalize_url(
        "https://example.com/page"
    )


def test_normalize_url_preserves_non_tracking_query_params():
    a = normalize_url("https://example.com/search?q=python")
    b = normalize_url("https://example.com/search?q=java")

    assert a != b  # a real, meaningful query parameter is not stripped


def test_normalize_url_does_not_mutate_the_input():
    original = "  HTTPS://Example.com/Page/  "
    normalize_url(original)

    assert original == "  HTTPS://Example.com/Page/  "  # unchanged


# --- deduplicate_results: basics ---------------------------------------------


def test_deduplicate_empty_list_returns_empty_list():
    assert deduplicate_results([]) == []


def test_deduplicate_all_unique_results_are_kept():
    results = [r("A"), r("B"), r("C")]

    assert deduplicate_results(results) == results


def test_deduplicate_first_occurrence_wins():
    first = SearchResult(title="First", url="https://example.com/a", snippet="")
    second = SearchResult(title="Second", url="https://example.com/a", snippet="")

    assert deduplicate_results([first, second]) == [first]


def test_deduplicate_preserves_original_ordering():
    results = [r("A"), r("B"), r("A"), r("C"), r("B")]

    assert [x.title for x in deduplicate_results(results)] == ["A", "B", "C"]


def test_deduplicate_example_from_the_spec():
    """A, B, A, C, B -> A, B, C."""
    results = [r("A"), r("B"), r("A"), r("C"), r("B")]

    assert deduplicate_results(results) == [r("A"), r("B"), r("C")]


# --- deduplicate_results: normalization-driven duplicates --------------------


def test_deduplicate_detects_trailing_slash_duplicates():
    a = SearchResult(title="A", url="https://example.com/page", snippet="")
    a_slash = SearchResult(title="A dup", url="https://example.com/page/", snippet="")

    assert deduplicate_results([a, a_slash]) == [a]


def test_deduplicate_detects_tracking_parameter_duplicates():
    a = SearchResult(title="A", url="https://example.com/page", snippet="")
    a_tracked = SearchResult(
        title="A dup",
        url="https://example.com/page?utm_source=newsletter",
        snippet="",
    )

    assert deduplicate_results([a, a_tracked]) == [a]


def test_deduplicate_detects_whitespace_duplicates():
    a = SearchResult(title="A", url="https://example.com/page", snippet="")
    a_padded = SearchResult(
        title="A dup", url="  https://example.com/page  ", snippet=""
    )

    assert deduplicate_results([a, a_padded]) == [a]


def test_deduplicate_preserves_the_original_usable_url():
    """The kept result's `url` is exactly what the provider gave it -
    normalization (here: uppercase scheme/host, a trailing slash) only
    affects the internal comparison key, never the stored/returned URL."""
    original = SearchResult(
        title="A", url="HTTPS://Example.com/page/", snippet=""
    )
    later_dup = SearchResult(title="A dup", url="https://example.com/page", snippet="")

    deduped = deduplicate_results([original, later_dup])

    assert deduped == [original]
    assert deduped[0].url == "HTTPS://Example.com/page/"  # untouched


# --- deduplicate_results: multi-pass combination ------------------------------


def test_deduplicate_across_multiple_passes_example():
    """Pass 1: A,B,C,D,E. Pass 2: C,D,E,F,G. Combined: A,B,C,D,E,F,G."""
    pass_1 = [r("A"), r("B"), r("C"), r("D"), r("E")]
    pass_2 = [r("C"), r("D"), r("E"), r("F"), r("G")]

    combined = deduplicate_results(pass_1 + pass_2)

    assert [x.title for x in combined] == ["A", "B", "C", "D", "E", "F", "G"]


# --- ResultQualityProvider ----------------------------------------------------


def test_result_quality_provider_dedupes_a_single_calls_results():
    wrapped = FakeProvider(results=[r("A"), r("B"), r("A")])
    quality = ResultQualityProvider(wrapped)

    results = quality.search("python")

    assert [x.title for x in results] == ["A", "B"]


def test_result_quality_provider_passes_query_and_max_results_through():
    wrapped = FakeProvider(results=[])
    quality = ResultQualityProvider(wrapped)

    quality.search("python", max_results=7)

    assert wrapped.calls == [("python", 7)]


def test_result_quality_provider_ignores_source_when_deduplicating():
    """Two results sharing a URL but differing `source` are still treated
    as duplicates - proves there is no `if source == "..."` branching."""
    from_searxng = r("A", source="searxng")
    same_url_different_source = SearchResult(
        title="A dup", url=from_searxng.url, snippet="", source="tavily"
    )
    quality = ResultQualityProvider(
        FakeProvider(results=[from_searxng, same_url_different_source])
    )

    results = quality.search("python")

    assert results == [from_searxng]


def test_result_quality_provider_on_empty_results():
    quality = ResultQualityProvider(FakeProvider(results=[]))

    assert quality.search("python") == []


# --- WebSearchTool: limit applied after dedup, across passes -----------------


def test_web_search_tool_applies_limit_after_dedup_not_before():
    """Provider returns: A, B, B, C, C, D, E. With limit 5: A, B, C, D, E -
    not just the first 3 unique items from a naive pre-dedup slice."""
    provider = FakeProvider(
        results=[r("A"), r("B"), r("B"), r("C"), r("C"), r("D"), r("E")]
    )
    tool = WebSearchTool(provider=provider)

    result = tool.execute(query="python", max_results=5)

    assert result.status == ToolResultStatus.SUCCESS
    assert [item["title"] for item in result.data] == ["A", "B", "C", "D", "E"]


def test_web_search_tool_combines_and_dedupes_multiple_planned_queries():
    provider = ScriptedProvider(
        [
            [r("A"), r("B"), r("C"), r("D"), r("E")],
            [r("C"), r("D"), r("E"), r("F"), r("G")],
        ]
    )
    planner = FixedPlanner(["python", "python 2026"])
    tool = WebSearchTool(provider=provider, planner=planner)

    result = tool.execute(query="python", mode="research", max_results=10)

    assert result.status == ToolResultStatus.SUCCESS
    assert [item["title"] for item in result.data] == [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
    ]


def test_web_search_tool_final_limit_truncates_the_deduped_combined_list():
    provider = ScriptedProvider(
        [
            [r("A"), r("B"), r("C"), r("D"), r("E")],
            [r("C"), r("D"), r("E"), r("F"), r("G")],
        ]
    )
    planner = FixedPlanner(["python", "python 2026"])
    tool = WebSearchTool(provider=provider, planner=planner)

    result = tool.execute(query="python", mode="research", max_results=3)

    assert [item["title"] for item in result.data] == ["A", "B", "C"]


def test_web_search_tool_on_empty_results_stays_empty():
    provider = FakeProvider(results=[])
    tool = WebSearchTool(provider=provider)

    result = tool.execute(query="python")

    assert result.status == ToolResultStatus.SUCCESS
    assert result.data == []


def test_web_search_tool_leaves_already_unique_results_untouched():
    provider = FakeProvider(results=[r("A"), r("B"), r("C")])
    tool = WebSearchTool(provider=provider)

    result = tool.execute(query="python", max_results=10)

    assert [item["title"] for item in result.data] == ["A", "B", "C"]


# --- Future source/link metadata availability ---------------------------------


def test_web_search_tool_output_exposes_source_and_the_original_clickable_url():
    """A future text UI needs result.title / result.url / result.source to
    render something like "Source 1 - Wikipedia" as a clickable link -
    prove all three survive through to the tool's structured output."""
    provider = FakeProvider(
        results=[
            SearchResult(
                title="NASA",
                url="https://www.nasa.gov/news",
                snippet="Official site",
                source="searxng",
            )
        ]
    )
    tool = WebSearchTool(provider=provider)

    result = tool.execute(query="nasa")

    assert result.data == [
        {
            "title": "NASA",
            "url": "https://www.nasa.gov/news",
            "snippet": "Official site",
            "source": "searxng",
        }
    ]
