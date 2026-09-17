"""
Deterministic conversational query enrichment for the SEARCH path.

SearchQueryBuilder decides WHAT TEXT to actually search once SearchRouter
(app/search/router.py) has already decided a message needs SEARCH - it
plays no part in that DIRECT/SEARCH decision and never sees a message
before the router does. It runs strictly after routing, inside
JarvisCore._handle_search (app/core.py).

Deliberately narrow and fixed-pattern, matching SearchRouter's own
philosophy: no Gemini call, no network call, no embeddings/semantic
similarity, no ranking, no autonomous research. It recognizes a small,
explicit set of context-dependent follow-up phrasings ("what about that",
"the second one", ...) and, only for those, prepends the most recent prior
user message found in the bounded conversation context (the same
M6.10 context JarvisCore already builds for SearchAnswerGenerator - no new
memory/context system). Every other message - including any self-contained
SEARCH query - passes through completely unchanged.

This is intentionally not a general coreference resolver: many genuine
follow-ups ("and Blue Origin?") won't be caught, by design - see the
README's Search Query Understanding section for the accepted trade-offs.
"""

import re
from abc import ABC, abstractmethod

# Explicit, narrow set of context-dependent follow-up phrasings. Each one
# only makes sense as a reference to something already established in the
# conversation ("that", "this", "it", "them", "the second one") - never a
# bare topic word, so this can't misfire on an ordinary self-contained
# question the way a broader/fuzzy heuristic could.
_CONTEXT_DEPENDENT_PATTERNS = (
    r"\bon\s+(?:that|this|it)\b",
    r"\babout\s+(?:that|this|it|them)\b",
    r"\b(?:that|this)\s+(?:one|topic|story)\b",
    r"\bthe\s+(?:first|second|third|last|other)\s+one\b",
    r"\bwhat\s+about\b",
    r"\bhow\s+about\b",
    r"\band\s+(?:that|this|them)\b",
)

_CONTEXT_DEPENDENT_REGEX = re.compile(
    "|".join(_CONTEXT_DEPENDENT_PATTERNS), re.IGNORECASE
)

_USER_LINE_PREFIX = "user: "


class SearchQueryBuilder(ABC):
    """Turns a SEARCH-bound message into the actual text to search."""

    @abstractmethod
    def build(self, message: str, recent_context: str) -> str:
        """Return the query to hand to WebSearchTool.search().

        `message` is the raw current message - the exact same string
        SearchRouter already routed to SEARCH. `recent_context` is the
        same bounded, plain-text conversation context JarvisCore builds
        for SearchAnswerGenerator (M6.10): possibly empty, and never a
        source of new facts here either - only used to recover the
        subject of a context-dependent follow-up.
        """


class DeterministicSearchQueryBuilder(SearchQueryBuilder):
    """Fixed-pattern query enrichment - no AI call, no query-generation
    model.

    Enrichment only happens when BOTH hold:
    1. `message` matches one of `_CONTEXT_DEPENDENT_PATTERNS` (it reads as
       a reference to something already discussed, not a fresh topic).
    2. `recent_context` contains a prior user message to recover as that
       reference's subject.

    Otherwise `message` is returned byte-for-byte unchanged - this is the
    common case, covering every self-contained SEARCH query.
    """

    def build(self, message: str, recent_context: str) -> str:
        if not recent_context:
            return message

        if not _CONTEXT_DEPENDENT_REGEX.search(message):
            return message

        previous_user_message = self._last_user_message(recent_context)
        if not previous_user_message or previous_user_message == message:
            return message

        return f"{message} {previous_user_message}"

    @staticmethod
    def _last_user_message(recent_context: str) -> str:
        """The most recent "user: ..." line in the bounded context - the
        immediately preceding user turn, since the current message hasn't
        been added to history yet when `_handle_search` builds this
        context (see app/core.py). Returns "" if none is found (e.g. the
        window only contains stored facts, or the first turn ever)."""
        last = ""
        for line in recent_context.splitlines():
            if line.startswith(_USER_LINE_PREFIX):
                last = line[len(_USER_LINE_PREFIX) :]
        return last
