"""
Deterministic search modes.

SearchMode/SearchModeConfig decide HOW MANY search passes WebSearchTool
runs and HOW MANY results each pass asks for - never which provider
answers them, and never how fallback or caching behave. This module never
imports or knows about SearXNG (or any concrete provider); WebSearchTool
still talks only to the SearchProvider it was given.

RESEARCH and CRITICAL are plain repeated passes over the same provider
chain, not an autonomous research agent: no new queries are generated (by
an LLM or otherwise), so every pass uses the identical query. Combining is
a simple concatenation with no deduplication or ranking - see
app/tools/web_search.py - both are explicitly left for a later milestone.
"""

from dataclasses import dataclass
from enum import Enum


class SearchMode(Enum):
    """How thoroughly web_search should search, chosen by the caller.

    Plain Enum with string values (matching PermissionLevel's convention),
    so a mode can be constructed from the plain string a caller passes,
    e.g. SearchMode("fast") -> SearchMode.FAST.
    """

    FAST = "fast"
    NORMAL = "normal"
    RESEARCH = "research"
    CRITICAL = "critical"


@dataclass(frozen=True)
class SearchModeConfig:
    """Deterministic parameters for one SearchMode.

    `passes` is how many times WebSearchTool calls its SearchProvider;
    `result_limit` is the default max_results per pass, used unless the
    caller explicitly overrides max_results.
    """

    passes: int
    result_limit: int


# Minimal, deterministic numbers - chosen to keep NORMAL's behavior
# identical to web_search before search modes existed (one pass, 5
# results), FAST cheaper for quick lookups, and RESEARCH/CRITICAL doing
# more passes for broader/redundant coverage without inventing new
# queries or providers.
SEARCH_MODE_CONFIGS: dict[SearchMode, SearchModeConfig] = {
    SearchMode.FAST: SearchModeConfig(passes=1, result_limit=3),
    SearchMode.NORMAL: SearchModeConfig(passes=1, result_limit=5),
    SearchMode.RESEARCH: SearchModeConfig(passes=2, result_limit=5),
    SearchMode.CRITICAL: SearchModeConfig(passes=3, result_limit=5),
}
