"""
Capability Decision Layer (M6.13).

The high-level "who should handle this message" decision - DIRECT vs
SEARCH today, more capabilities (MEMORY, TOOL, EMAIL, CALENDAR, PRINTER,
ROBOTIC_ARM, ...) later - lives here, not in JarvisCore. JarvisCore
(app/core.py) only orchestrates: it asks a CapabilityRouter for a
CapabilityDecision and dispatches on it, never encoding routing rules
itself.

M6.13 only implements DIRECT and SEARCH - no other Capability member is
added "for the future"; see the README's Postponed decisions for why.
Adding a real future capability means adding an enum member plus whatever
decision logic picks it, without touching JarvisCore's dispatch mechanics.

DeterministicCapabilityRouter generalizes M6.9's SearchRouter
(app/search/router.py) into this layer by delegating to it rather than
duplicating its regex rules - SearchRouter (and its own tests,
tests/test_search_router.py) stays the single source of truth for the
actual DIRECT/SEARCH keyword matching; it's no longer the top-level
interface JarvisCore talks to directly for that decision.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from app.search.router import SearchDecision, SearchRouter


class Capability(Enum):
    """A handler Jarvis can dispatch a message to.

    Only DIRECT and SEARCH exist today. MEMORY/TOOL/EMAIL/CALENDAR/
    PRINTER/ROBOTIC_ARM/etc. are deliberately not added as members until a
    milestone actually implements the decision logic for them - an unused
    enum value isn't "future-facing architecture," just dead code.
    """

    DIRECT = "direct"
    SEARCH = "search"


@dataclass
class CapabilityDecision:
    """The result of a CapabilityRouter decision.

    `capability` is the only thing JarvisCore needs to dispatch correctly
    today - DIRECT and SEARCH both dispatch using JarvisCore's own
    already-loaded message/history/facts, no extra payload required. Kept
    as its own (currently trivial) type rather than returning a bare
    Capability so a future capability that needs to carry extra dispatch
    data can extend this dataclass without changing CapabilityRouter's
    return type.
    """

    capability: Capability


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


class DeterministicCapabilityRouter(CapabilityRouter):
    """The only capability decision maker for now: DIRECT vs SEARCH,
    delegating to an injected SearchRouter (defaulting to a bare
    SearchRouter()) for the actual keyword/pattern matching - never a
    Gemini/LLM call, same as SearchRouter itself.

    Injecting a `search_router` here is how JarvisCore keeps its existing
    `router`/`SearchRouter` constructor parameter meaningful (see
    app/core.py): whatever SearchRouter instance Core was given still
    makes the actual decision, just reached through this layer instead of
    Core calling it directly.
    """

    def __init__(self, search_router: SearchRouter | None = None):
        self.search_router = search_router or SearchRouter()

    def route(self, message: str) -> CapabilityDecision:
        decision = self.search_router.route(message)
        capability = (
            Capability.SEARCH
            if decision is SearchDecision.SEARCH
            else Capability.DIRECT
        )
        return CapabilityDecision(capability=capability)
