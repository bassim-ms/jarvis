"""
Deterministic search routing.

SearchRouter decides DIRECT vs SEARCH for a user message using fixed
keyword/phrase matching only - never a Gemini/LLM call. It knows nothing
about SearchProvider, SearchMode, WebSearchTool, or JarvisCore; it only
looks at the message text, so it stays trivially unit-testable and has no
dependency on the rest of the search stack.

Explicit search intent always wins: if any search-triggering pattern is
present, the decision is SEARCH regardless of what else is in the message
- e.g. "search the web and explain quantum entanglement" is SEARCH, even
though "quantum entanglement" alone would be DIRECT. This falls out
naturally from doing one unconditional OR-match over every pattern, not
from any special-cased "override" logic.

This is deliberately coarse (plain keyword matching, not intent
understanding) - see the README's Search Routing section for the accepted
trade-offs and what's explicitly left for a later, smarter router.
"""

import re
from enum import Enum


class SearchDecision(Enum):
    """Where a user message should be routed."""

    DIRECT = "direct"
    SEARCH = "search"


# Explicit requests to search/look something up. Deliberately NOT a bare
# `\bsearch\b`: that would match "search" used as a plain noun/algorithm
# name ("explain binary search", "depth-first search") - very plausible
# for this kind of assistant - and wrongly route a stable question to
# SEARCH. Requiring an actual request phrasing keeps the "explicit intent"
# rule meaningful without adding any non-deterministic heuristics.
_EXPLICIT_SEARCH_PATTERNS = (
    r"\bsearch\s+(?:for|the\s+web|online|up)\b",
    r"\b(?:please|can\s+you|could\s+you)\s+search\b",
    r"\blook\s+up\b",
    r"\blook\s+(?:it|this|that)\s+up\b",
    r"\bgoogle\s+(?:it|this|that)\b",
    r"\bweb\s*search\b",
)

# Freshness/currentness indicators.
_FRESHNESS_PATTERNS = (
    r"\blatest\b",
    r"\bcurrent(?:ly)?\b",
    r"\btoday\b",
    r"\byesterday\b",
    r"\brecent(?:ly)?\b",
    r"\bnow\b",
    r"\bthis\s+(?:week|month|year)\b",
)

# Live/external information requests.
_LIVE_INFO_PATTERNS = (
    r"\bweather\b",
    r"\bforecast\b",
    r"\bscores?\b",
    r"\bresults?\b",
    r"\bprices?\b",
    r"\bavailab(?:le|ility)\b",
)

_SEARCH_REGEX = re.compile(
    "|".join(_EXPLICIT_SEARCH_PATTERNS + _FRESHNESS_PATTERNS + _LIVE_INFO_PATTERNS),
    re.IGNORECASE,
)


_EXPLICIT_SEARCH_REGEX = re.compile("|".join(_EXPLICIT_SEARCH_PATTERNS), re.IGNORECASE)


def is_explicit_search_request(message: str) -> bool:
    """Whether the message explicitly ASKS for a search ("search for ...",
    "look it up", "web search") - as opposed to being routed to SEARCH only
    because it contains a freshness word such as "now" or "today". Pure."""
    return bool(_EXPLICIT_SEARCH_REGEX.search(message))


class SearchRouter:
    """Deterministically decides whether a message needs a web search."""

    def route(self, message: str) -> SearchDecision:
        if _SEARCH_REGEX.search(message):
            return SearchDecision.SEARCH
        return SearchDecision.DIRECT
