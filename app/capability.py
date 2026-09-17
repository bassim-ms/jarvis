"""
Capability Decision Layer (M6.13, extended in M6.14 with TOOL).

The high-level "who should handle this message" decision - DIRECT, SEARCH,
and (M6.14) TOOL today, more capabilities (MEMORY, EMAIL, CALENDAR,
PRINTER, ROBOTIC_ARM, ...) later - lives here, not in JarvisCore.
JarvisCore (app/core.py) only orchestrates: it asks a CapabilityRouter for
a CapabilityDecision and dispatches on it, never encoding routing rules
itself.

Only DIRECT, SEARCH, and TOOL are implemented - no other Capability member
is added "for the future"; see the README's Postponed decisions for why.
Adding a real future capability means adding an enum member plus whatever
decision logic picks it, without touching JarvisCore's dispatch mechanics.

DeterministicCapabilityRouter generalizes M6.9's SearchRouter
(app/search/router.py) into this layer by delegating to it for DIRECT vs
SEARCH rather than duplicating its regex rules - SearchRouter (and its own
tests, tests/test_search_router.py) stays the single source of truth for
that matching; it's no longer the top-level interface JarvisCore talks to
directly for the decision. TOOL detection (M6.14) is a separate, narrow,
purely textual check - see DeterministicCapabilityRouter's docstring.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from app.search.router import SearchDecision, SearchRouter


class Capability(Enum):
    """A handler Jarvis can dispatch a message to.

    Only DIRECT, SEARCH, and TOOL exist today. MEMORY/EMAIL/CALENDAR/
    PRINTER/ROBOTIC_ARM/etc. are deliberately not added as members until a
    milestone actually implements the decision logic for them - an unused
    enum value isn't "future-facing architecture," just dead code.
    """

    DIRECT = "direct"
    SEARCH = "search"
    TOOL = "tool"


@dataclass
class CapabilityDecision:
    """The result of a CapabilityRouter decision.

    `capability` is the only thing JarvisCore needs to dispatch DIRECT or
    SEARCH correctly - both use JarvisCore's own already-loaded
    message/history/facts, no extra payload required.

    `tool_name` (M6.14) is used only when `capability` is `TOOL`: the name
    of the registered `Tool` (app/tools/base.py) to dispatch to via
    `ToolRegistry.execute(...)`, exactly as the user's message named it -
    never guessed, inferred, or fuzzy-matched. It's `None` when TOOL was
    clearly the right capability (an explicit tool-invocation phrasing was
    used) but no concrete tool name could be identified from the message
    (e.g. "use tool" with nothing named) - an explicit, typed way to
    represent "TOOL, but nothing to dispatch to" rather than guessing a
    tool or silently falling back to DIRECT. It's also `None`, and simply
    unused, for DIRECT/SEARCH decisions.
    """

    capability: Capability
    tool_name: str | None = None


class CapabilityRouter(ABC):
    """Decides which capability should handle a message.

    Answers only "which capability?" - it never performs the capability
    itself (that's JarvisCore's dispatch + the capability's own handler,
    e.g. WebSearchTool/SearchAnswerGenerator for SEARCH).
    """

    @abstractmethod
    def route(self, message: str) -> CapabilityDecision:
        """Return the CapabilityDecision for `message`.

        `message` is the raw current message only, matching every
        deterministic router in this codebase (SearchRouter, M6.9) - no
        conversation context is used to decide WHICH capability handles a
        request. Context-dependent behavior after a capability is already
        chosen (e.g. SEARCH's query enrichment, M6.12) is a separate,
        lower-level concern this router never touches.
        """


#: Deliberately conservative (M6.14): only an explicit, command-like
#: invocation at the start of the message counts as a TOOL request - never
#: a message that merely mentions "a tool" in passing ("what tool should I
#: use for this job?" stays DIRECT). The tool name itself, if given, is
#: captured verbatim - no fuzzy matching, no case-folding, no synonym
#: mapping. There is currently no safe, general way to map arbitrary
#: natural language to a specific registered tool, so none is invented
#: here; this pattern is the entire extent of TOOL "understanding" for
#: M6.14.
_TOOL_INVOCATION_PATTERN = re.compile(
    r"^\s*(?:use|run|call|invoke)\s+tool\b\s*:?\s*(\S+)?", re.IGNORECASE
)


class DeterministicCapabilityRouter(CapabilityRouter):
    """The only capability decision maker for now: TOOL, then DIRECT vs
    SEARCH, all without any Gemini/LLM call.

    TOOL is checked first, via a narrow explicit-invocation phrasing
    (`_TOOL_INVOCATION_PATTERN`) - the most deliberate, unambiguous signal
    of the three, so it takes priority even over a message that also
    happens to contain a SEARCH-triggering word (e.g. "use tool weather"
    is TOOL, not SEARCH, even though "weather" alone would be SEARCH).
    This check is pure text parsing - it never looks at what tools are
    actually registered (that's ToolRegistry's job, at dispatch time in
    JarvisCore._handle_tool - see app/core.py) and never executes
    anything.

    Everything that doesn't match the TOOL pattern is delegated to an
    injected SearchRouter (defaulting to a bare SearchRouter()) for the
    actual DIRECT/SEARCH keyword/pattern matching, exactly as in M6.13.

    Injecting a `search_router` here is how JarvisCore keeps its existing
    `router`/`SearchRouter` constructor parameter meaningful (see
    app/core.py): whatever SearchRouter instance Core was given still
    makes the DIRECT/SEARCH decision, just reached through this layer
    instead of Core calling it directly.
    """

    def __init__(self, search_router: SearchRouter | None = None):
        self.search_router = search_router or SearchRouter()

    def route(self, message: str) -> CapabilityDecision:
        tool_match = _TOOL_INVOCATION_PATTERN.match(message)
        if tool_match:
            return CapabilityDecision(
                capability=Capability.TOOL, tool_name=tool_match.group(1)
            )

        decision = self.search_router.route(message)
        capability = (
            Capability.SEARCH
            if decision is SearchDecision.SEARCH
            else Capability.DIRECT
        )
        return CapabilityDecision(capability=capability)
