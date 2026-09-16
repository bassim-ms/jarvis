"""
Structured search context for one logical web_search request.

Provider-agnostic: SearchResponse only references SearchMode and
SearchResult, never a concrete provider, Gemini, or any presentation
concern. It exists so a future answer-generation layer can see not just
the final results but the resolved mode and the exact queries that
produced them, rather than just a flat result list - see
app/tools/web_search.py, which builds one per execute() call.
"""

from dataclasses import dataclass

from app.search.base import SearchResult
from app.search.modes import SearchMode


@dataclass
class SearchResponse:
    """The full structured outcome of one web_search request.

    `query` is the original, unmodified user query. `queries_used` is
    every query the SearchPlanner actually ran (see app/search/planner.py)
    - for FAST/NORMAL this is just [query]; for RESEARCH/CRITICAL it also
    includes the deterministic variations. `results` is the final,
    deduplicated, limit-applied list - each SearchResult retains its own
    `query` field recording which planned query actually produced it
    (not necessarily this SearchResponse's `query`).
    """

    query: str
    mode: SearchMode
    queries_used: list[str]
    results: list[SearchResult]
