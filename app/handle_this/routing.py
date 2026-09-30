"""
Routing into Handle This: an adapter around the EXISTING capability router.

`ObjectiveAwareCapabilityRouter` asks the router it wraps for its decision
first, exactly as before, and then - only for the few decisions where it is
allowed to - asks the objective detector (app/handle_this/objective.py) one
question: "is this a delegated high-level objective?". If so, the decision
becomes TASK, which is the capability JarvisCore already sends to Handle This
(app/core.py `_handle_task`); the Job, planning and execution machinery is
untouched. Nothing here understands the objective, checks the boundary, reads
context or runs anything.

PRIORITY (the existing order is kept; this only acts AFTER it):

  1. Explicit intents the wrapped router already recognised - a tool command,
     a confirmation reply, self-development, a task-status question - are
     returned unchanged. A delegated-objective look can never override them.
  2. Otherwise, a message the router called DIRECT or SEARCH is promoted to
     TASK only if it is clearly a delegated objective. A plain question, a
     simple command, or an explicit search ("Search Google for the definition
     of CoreXY") is not, so it keeps its existing path.
  3. A message the router already called TASK stays TASK - with one exception.
     The router's goal-verb rule sends anything that STARTS with "research"
     to TASK, including plain information requests ("Research what a black
     hole is"). Now that TASK means "understood, then an autonomous Job",
     such a message is handed back to the existing search/chat path unless it
     really is a delegated objective. It is NOT handed back if Handle This's
     own scan would have refused it (an attempt to loosen limits, a
     transaction, a file deletion...): those keep reaching Handle This so the
     refusal stays explicit. The other goal verbs (prepare, organize, plan,
     arrange, set up, put together) and the explicit "create a goal:" phrasing
     are unchanged.

When in doubt the answer is "existing behaviour": any fault inside this layer
returns the wrapped router's decision untouched. It adds no AI call and does
no I/O. Install it together with a configured Handle This (app/main.py); a
promoted TASK without one would be the older, un-understood path.
"""

from __future__ import annotations

import re

from app.capability import Capability, CapabilityDecision, CapabilityRouter
from app.handle_this.objective import ObjectiveRoute, detect_objective
from app.handle_this.signals import detect_unsupported
from app.search.router import SearchDecision, SearchRouter

#: Decisions the wrapped router made on an explicit signal: never overridden.
_EXPLICIT = frozenset(
    {
        Capability.TOOL,
        Capability.CONFIRMATION,
        Capability.SELF_DEVELOPMENT,
        Capability.SELF_DEV_APPROVAL,
        Capability.TASK_STATUS,
    }
)

#: The one goal verb whose plain use is usually an information request.
_RESEARCH_VERB = re.compile(r"^\s*research\b", re.IGNORECASE)


class ObjectiveAwareCapabilityRouter(CapabilityRouter):
    """See the module docstring."""

    def __init__(
        self,
        inner: CapabilityRouter,
        *,
        detector=detect_objective,
        search_router: SearchRouter | None = None,
    ):
        self.inner = inner
        self.detector = detector
        self.search_router = search_router or getattr(inner, "search_router", None) or SearchRouter()

    def route(self, message: str) -> CapabilityDecision:
        decision = self.inner.route(message)
        try:
            return self._adjust(message, decision)
        except Exception:  # noqa: BLE001 - any fault here means "existing behaviour"
            return decision

    def _adjust(self, message: str, decision: CapabilityDecision) -> CapabilityDecision:
        if decision.capability in _EXPLICIT:
            return decision

        verdict = self.detector(message)
        if verdict.route is ObjectiveRoute.MORNING_BRIEF:
            # A read-only review, not a delegated objective: never a Job. The goal-verb rule would turn
            # "prepare my morning brief" into TASK; it goes to the ordinary conversation path instead,
            # where JarvisCore hands it to the read-only Morning Brief. DIRECT / SEARCH are left as they are.
            if decision.capability is Capability.TASK:
                return CapabilityDecision(capability=Capability.DIRECT)
            return decision

        is_objective = verdict.route is ObjectiveRoute.HANDLE_THIS

        if decision.capability in (Capability.DIRECT, Capability.SEARCH):
            if is_objective:
                return CapabilityDecision(capability=Capability.TASK, goal=message.strip())
            return decision

        if decision.capability is Capability.TASK and self._is_bare_research_task(message, decision):
            if not is_objective and detect_unsupported(message) is None:
                return self._information_decision(message)
        return decision

    @staticmethod
    def _is_bare_research_task(message: str, decision: CapabilityDecision) -> bool:
        """TASK produced by the goal-verb rule (the goal IS the whole message)
        for a message that starts with "research" - not the explicit
        "create a goal: ..." phrasing, whose goal is only the text after it."""
        return (decision.goal or "").strip() == message.strip() and bool(_RESEARCH_VERB.match(message))

    def _information_decision(self, message: str) -> CapabilityDecision:
        searching = self.search_router.route(message) is SearchDecision.SEARCH
        return CapabilityDecision(capability=Capability.SEARCH if searching else Capability.DIRECT)
