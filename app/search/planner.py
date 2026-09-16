"""
Deterministic search query planning.

SearchPlanner decides WHAT queries to run for a given mode - never how many
times to call a provider for one query, which provider answers, or how
caching/fallback/deduplication behave. A planner imports nothing from
app/search/base.py, app/search/cache.py, app/search/manager.py, or
app/search/quality.py - it only knows about SearchMode and plain strings.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

from app.search.modes import SearchMode


@dataclass
class SearchPlan:
    """An ordered list of queries to run for one logical search request."""

    queries: list[str]


class SearchPlanner(ABC):
    """Turns one user query + mode into an ordered SearchPlan."""

    @abstractmethod
    def plan(self, query: str, mode: SearchMode) -> SearchPlan:
        """Return the ordered list of queries to run for `mode`."""


class DeterministicSearchPlanner(SearchPlanner):
    """Fixed, non-AI query variations per mode - no query-generation model,
    just a small set of deterministic intent transformations:

    - PRIMARY: the original query, unchanged
    - CURRENT: "{query} {current_year}" - biases toward recent results
    - STUDENT: "{query} student" - biases toward educational/explainer
      content
    - OFFICIAL: "{query} official" - biases toward official/primary
      sources

    FAST and NORMAL plan PRIMARY only (unchanged from M6.4/M6.5 behavior).
    RESEARCH adds CURRENT and STUDENT; CRITICAL adds OFFICIAL on top of
    that. This exists specifically so RESEARCH/CRITICAL stop searching the
    exact same query multiple times (see app/tools/web_search.py) without
    inventing an AI query generator - each variation is a fixed suffix, not
    a generic filler word like "research" or "information".
    """

    def __init__(self, current_year: int | None = None):
        self.current_year = (
            current_year if current_year is not None else date.today().year
        )

    def plan(self, query: str, mode: SearchMode) -> SearchPlan:
        query = query.strip()
        candidates = [query]

        if mode in (SearchMode.RESEARCH, SearchMode.CRITICAL):
            candidates.append(f"{query} {self.current_year}")
            candidates.append(f"{query} student")

        if mode is SearchMode.CRITICAL:
            candidates.append(f"{query} official")

        # Defensive, deterministic dedup - first occurrence wins - in case
        # a transformation happens to coincide with an earlier one for a
        # particular input.
        seen: set[str] = set()
        queries: list[str] = []
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            queries.append(candidate)

        return SearchPlan(queries=queries)
