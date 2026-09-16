"""
Tests for deterministic search query planning (app/search/planner.py):
SearchPlan/SearchPlanner/DeterministicSearchPlanner in isolation, and that
WebSearchTool actually uses an injected planner. Cross-query aggregation,
deduplication, and cache-compatibility integration tests live in
test_search_modes.py and test_search_quality.py.
"""

from datetime import date

from app.search.base import SearchProvider
from app.search.modes import SearchMode
from app.search.planner import DeterministicSearchPlanner, SearchPlan, SearchPlanner
from app.tools.web_search import WebSearchTool


# --- DeterministicSearchPlanner: per-mode output ----------------------------


def test_fast_plans_the_original_query_only():
    planner = DeterministicSearchPlanner(current_year=2026)

    plan = planner.plan("python", SearchMode.FAST)

    assert plan.queries == ["python"]


def test_normal_plans_the_original_query_only():
    planner = DeterministicSearchPlanner(current_year=2026)

    plan = planner.plan("python", SearchMode.NORMAL)

    assert plan.queries == ["python"]


def test_research_plans_original_current_and_student():
    planner = DeterministicSearchPlanner(current_year=2026)

    plan = planner.plan("python", SearchMode.RESEARCH)

    assert plan.queries == ["python", "python 2026", "python student"]


def test_critical_plans_original_current_student_and_official():
    planner = DeterministicSearchPlanner(current_year=2026)

    plan = planner.plan("python", SearchMode.CRITICAL)

    assert plan.queries == [
        "python",
        "python 2026",
        "python student",
        "python official",
    ]


# --- Query counts --------------------------------------------------------------


def test_query_counts_per_mode():
    planner = DeterministicSearchPlanner(current_year=2026)

    counts = {mode: len(planner.plan("python", mode).queries) for mode in SearchMode}

    assert counts == {
        SearchMode.FAST: 1,
        SearchMode.NORMAL: 1,
        SearchMode.RESEARCH: 3,
        SearchMode.CRITICAL: 4,
    }


# --- No generic filler words ----------------------------------------------------


def test_planner_never_appends_generic_filler_words():
    planner = DeterministicSearchPlanner(current_year=2026)

    for mode in SearchMode:
        for planned_query in planner.plan("python", mode).queries:
            assert "research" not in planned_query
            assert "information" not in planned_query


# --- Injected current year ------------------------------------------------------


def test_current_year_is_injectable():
    planner = DeterministicSearchPlanner(current_year=1999)

    plan = planner.plan("python", SearchMode.RESEARCH)

    assert "python 1999" in plan.queries


def test_current_year_defaults_to_the_real_current_year():
    planner = DeterministicSearchPlanner()

    assert planner.current_year == date.today().year


# --- Determinism -------------------------------------------------------------


def test_plan_is_deterministic_across_calls():
    planner = DeterministicSearchPlanner(current_year=2026)

    first = planner.plan("python", SearchMode.CRITICAL)
    second = planner.plan("python", SearchMode.CRITICAL)

    assert first == second


# --- Whitespace / duplicate handling --------------------------------------------


def test_plan_strips_surrounding_whitespace():
    planner = DeterministicSearchPlanner(current_year=2026)

    plan = planner.plan("  python  ", SearchMode.RESEARCH)

    assert plan.queries == ["python", "python 2026", "python student"]


def test_plan_never_contains_duplicate_queries():
    """Defensive: if a transformation happened to coincide with an earlier
    one for some input, the plan still only lists it once."""
    planner = DeterministicSearchPlanner(current_year=2026)

    for mode in SearchMode:
        queries = planner.plan("python", mode).queries
        assert len(queries) == len(set(queries))


# --- WebSearchTool uses the injected SearchPlanner ------------------------------


class RecordingProvider(SearchProvider):
    """Records every query string it was asked to search; always returns
    no results, since these tests only care about what was requested."""

    def __init__(self):
        self.queries: list[str] = []

    def search(self, query, max_results=5):
        self.queries.append(query)
        return []


class TwoQueryPlanner(SearchPlanner):
    """A trivial custom planner, proving WebSearchTool works with any
    SearchPlanner implementation, not just DeterministicSearchPlanner."""

    def plan(self, query, mode):
        return SearchPlan(queries=[query, f"{query} extra"])


def test_web_search_tool_executes_every_planned_query():
    provider = RecordingProvider()
    tool = WebSearchTool(provider=provider, planner=TwoQueryPlanner())

    tool.execute(query="python")

    assert provider.queries == ["python", "python extra"]


def test_web_search_tool_defaults_to_a_deterministic_planner():
    tool = WebSearchTool()

    assert isinstance(tool.planner, DeterministicSearchPlanner)


def test_web_search_tool_research_mode_sends_the_planners_exact_queries():
    provider = RecordingProvider()
    tool = WebSearchTool(
        provider=provider, planner=DeterministicSearchPlanner(current_year=2026)
    )

    tool.execute(query="python", mode="research")

    assert provider.queries == ["python", "python 2026", "python student"]


def test_web_search_tool_fast_mode_sends_only_the_original_query():
    provider = RecordingProvider()
    tool = WebSearchTool(
        provider=provider, planner=DeterministicSearchPlanner(current_year=2026)
    )

    tool.execute(query="python", mode="fast")

    assert provider.queries == ["python"]
