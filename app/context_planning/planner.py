"""
ContextPlanner (M16.4): the abstraction and LLM-backed implementation
that turns (raw_request, RequestContext, ContextSnapshot, ReasoningResult)
into a validated PlanningDirective.

STRICTLY READ-ONLY AND NON-EXECUTING: nothing here ever calls
ToolRegistry.execute, creates a Task/Job, mutates the World Model,
sends a notification, or controls a browser/computer/hardware - it only
ever reads what it's handed and returns a PlanningDirective for
"existing response generation" (app/core.py) to use. It never
independently queries the World Model or MemoryStore, and it never
builds a competing executable Plan/PlanStep - M15 (app/orchestration/)
remains the sole authoritative executable planning/execution system;
see app/context_planning/base.py:PlanningDirective's own docstring for
how `requires_execution` reaches M15 without this module ever touching
it directly.

Uses the existing AIProvider/provider-routing architecture (app/
providers/) exactly as app.context_reasoning.reasoner.LLMContextReasoner
already does for M16.3's own structured-output needs - via
`call_structured` (app/providers/manager.py) with
RequestRequirements(structured_output=True): the providers are asked in
the fixed configured order, and a reply that fails this module's own
strict validation sends the request on to the next one (nothing is
promoted for the task) - never a second AI client, never a
hardcoded provider class here. `LLMContextPlanner` takes its AIProvider
as a required constructor argument, so it is always routed through
whatever provider/provider-manager the caller (normally JarvisCore)
already has - see app/context_planning/factory.py.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from app.context_engine.base import RequestContext
from app.context_planning.base import (
    GroundedReference,
    PlanningDirective,
    PlanningDirectiveError,
    PlanningType,
    PlanningTypeError,
    parse_planning_type,
)
from app.context_planning.prompt import build_planning_prompt
from app.context_reasoning.base import ReasoningResult
from app.context_reasoning.deterministic import DeterministicAnalysis, analyze
from app.context_reasoning.prompt import collect_allowed_provenance
from app.providers.base import AIProvider, ProviderError
from app.providers.manager import call_structured
from app.providers.requirements import RequestRequirements, TaskType
from app.world_model.base import ContextSnapshot


class ContextPlanningError(RuntimeError):
    """Raised when a ContextPlanner can't produce a valid
    PlanningDirective - a failed AI call, malformed output, output that
    fails strict schema validation, output citing provenance outside
    what was actually supplied (the anti-hallucination gate), or a
    deterministic coherence violation (see PlanningDirectiveError).
    Never silently guessed around - the same ContextReasonerError
    convention (app/context_reasoning/reasoner.py) applied to M16.4's
    own boundary."""


class ContextPlanner(ABC):
    """Answers "given the request, the context M16.2 selected, and what
    M16.3 concluded, what planning STRATEGY does this call for?" - never
    executes anything, never queries the World Model/MemoryStore itself,
    and never builds a competing executable Plan (M15 remains the sole
    authority for that)."""

    @abstractmethod
    def plan(
        self,
        raw_request: str,
        request_context: RequestContext,
        snapshot: ContextSnapshot,
        reasoning_result: ReasoningResult,
    ) -> PlanningDirective:
        """Return a validated PlanningDirective. Raises
        ContextPlanningError on failure - never a fabricated or
        partially-guessed PlanningDirective."""


def _context_is_empty(snapshot: ContextSnapshot, reasoning_result: ReasoningResult) -> bool:
    """True when there is nothing worth reasoning strategically about -
    an empty ContextSnapshot AND an empty ReasoningResult (see
    ReasoningResult.is_empty(), app/context_reasoning/base.py). Mirrors
    LLMContextReasoner's own empty-ContextSnapshot short circuit one
    layer up (app/context_reasoning/reasoner.py) - never calls the
    provider when there's genuinely nothing to plan around."""
    snapshot_fields = (
        "active_projects",
        "relevant_goals",
        "relevant_tasks",
        "upcoming_commitments",
        "upcoming_events",
        "active_blockers",
        "relevant_resources",
        "relevant_preferences",
        "relevant_memory",
    )
    snapshot_empty = all(not getattr(snapshot, f) for f in snapshot_fields)
    return snapshot_empty and reasoning_result.is_empty()


def _none_directive() -> PlanningDirective:
    """The fixed, minimal PlanningDirective returned with NO provider
    call when `_context_is_empty` - there is nothing to strategize
    about. `is_empty()` is true for this result (see
    app/context_planning/base.py), which is what lets app/core.py's own
    `_format_planning_directive` preserve byte-identical prompt text for
    a request with nothing relevant."""
    return PlanningDirective(
        objective="No planning-relevant context was available for this request.",
        planning_type=PlanningType.NONE,
    )


_VALID_TOP_LEVEL_KEYS = {
    "objective",
    "planning_type",
    "contextual_constraints",
    "relevant_resources",
    "dependencies",
    "assumptions",
    "suggested_approach",
    "requires_execution",
    "confidence",
}

_GROUNDED_LIST_FIELDS = ("contextual_constraints", "relevant_resources", "dependencies")

# The model often wraps JSON in a markdown code fence even when told not
# to; stripping one is tolerant formatting cleanup, not guessing at
# content - the same tolerance app.context_reasoning.reasoner already
# applies to its own strict-JSON provider.
_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _parse_grounded_list(
    payload: dict, field_name: str, allowed_provenance: set[str], raw: str
) -> tuple[GroundedReference, ...]:
    items = payload.get(field_name, [])
    if not isinstance(items, list):
        raise ContextPlanningError(f"'{field_name}' must be a JSON array, got: {raw!r}")

    parsed = []
    for entry in items:
        if not isinstance(entry, dict):
            raise ContextPlanningError(
                f"Each '{field_name}' entry must be a JSON object, got: {raw!r}"
            )
        extra = set(entry) - {"text", "provenance"}
        if extra:
            raise ContextPlanningError(
                f"Unexpected fields in a '{field_name}' entry: {sorted(extra)}"
            )
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ContextPlanningError(
                f"Each '{field_name}' entry needs a non-empty 'text', got: {raw!r}"
            )
        provenance = entry.get("provenance")
        if (
            not isinstance(provenance, list)
            or not provenance
            or any(not isinstance(p, str) or not p for p in provenance)
        ):
            raise ContextPlanningError(
                f"Each '{field_name}' entry needs a non-empty 'provenance' array of "
                f"strings (every claim must be grounded), got: {raw!r}"
            )
        unsupported = [p for p in provenance if p not in allowed_provenance]
        if unsupported:
            raise ContextPlanningError(
                f"'{field_name}' entry cites provenance not present in the supplied "
                f"context (possible hallucination): {unsupported!r}. Full response: {raw!r}"
            )
        parsed.append(GroundedReference(text=text, provenance=tuple(provenance)))
    return tuple(parsed)


def _parse_and_validate(raw: str, allowed_provenance: set[str]) -> PlanningDirective:
    cleaned = _CODE_FENCE_PATTERN.sub("", raw.strip()).strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ContextPlanningError(f"Invalid JSON from context planner: {raw!r}") from exc

    if not isinstance(payload, dict):
        raise ContextPlanningError(f"Expected a JSON object, got: {raw!r}")

    extra_keys = set(payload) - _VALID_TOP_LEVEL_KEYS
    if extra_keys:
        raise ContextPlanningError(
            f"Unexpected fields in planning directive: {sorted(extra_keys)}"
        )

    objective = payload.get("objective")
    if not isinstance(objective, str) or not objective.strip():
        raise ContextPlanningError(f"Missing or invalid 'objective': {raw!r}")

    planning_type_raw = payload.get("planning_type")
    if not isinstance(planning_type_raw, str):
        raise ContextPlanningError(f"Missing or invalid 'planning_type': {raw!r}")
    try:
        planning_type = parse_planning_type(planning_type_raw)
    except PlanningTypeError as exc:
        raise ContextPlanningError(f"Invalid 'planning_type': {exc}") from exc

    grounded_fields = {
        field_name: _parse_grounded_list(payload, field_name, allowed_provenance, raw)
        for field_name in _GROUNDED_LIST_FIELDS
    }

    assumptions_raw = payload.get("assumptions", [])
    if not isinstance(assumptions_raw, list) or any(
        not isinstance(a, str) or not a.strip() for a in assumptions_raw
    ):
        raise ContextPlanningError(
            f"'assumptions' must be a JSON array of non-empty strings, got: {raw!r}"
        )

    suggested_approach = payload.get("suggested_approach")
    if suggested_approach is not None and (
        not isinstance(suggested_approach, str) or not suggested_approach.strip()
    ):
        raise ContextPlanningError(f"Invalid 'suggested_approach': {raw!r}")

    requires_execution = payload.get("requires_execution")
    if not isinstance(requires_execution, bool):
        raise ContextPlanningError(
            f"Missing or invalid 'requires_execution' (must be a bool): {raw!r}"
        )

    confidence = payload.get("confidence")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not (0.0 <= confidence <= 1.0)
    ):
        raise ContextPlanningError(
            f"'confidence' must be a number in [0.0, 1.0], got: {raw!r}"
        )

    try:
        return PlanningDirective(
            objective=objective,
            planning_type=planning_type,
            contextual_constraints=grounded_fields["contextual_constraints"],
            relevant_resources=grounded_fields["relevant_resources"],
            dependencies=grounded_fields["dependencies"],
            assumptions=tuple(assumptions_raw),
            suggested_approach=suggested_approach,
            requires_execution=requires_execution,
            confidence=float(confidence),
        )
    except PlanningDirectiveError as exc:
        raise ContextPlanningError(f"Planning directive failed validation: {exc}") from exc


class LLMContextPlanner(ContextPlanner):
    """ContextPlanner backed by the existing AIProvider/provider-routing
    architecture - see this module's own docstring. `provider` is
    required (no internal default/hardcoded vendor).

    `plan()` first checks `_context_is_empty` - if so, returns
    `_none_directive()` with NO provider call. Otherwise it recomputes
    the same app.context_reasoning.deterministic.analyze() pass M16.3
    already ran (pure, no AI call) purely to derive the allowed-
    provenance whitelist (`collect_allowed_provenance`), builds the
    dedicated planning prompt (app/context_planning/prompt.py), calls
    the provider with a structured-output requirement, then strictly
    parses/validates the response - including that every provenance
    identifier cited actually appears in what was supplied, and that
    `requires_execution` is coherent with `planning_type` - before
    constructing a PlanningDirective. Any failure raises
    ContextPlanningError."""

    def __init__(self, provider: AIProvider):
        self.provider = provider

    def plan(
        self,
        raw_request: str,
        request_context: RequestContext,
        snapshot: ContextSnapshot,
        reasoning_result: ReasoningResult,
    ) -> PlanningDirective:
        if _context_is_empty(snapshot, reasoning_result):
            return _none_directive()

        analysis: DeterministicAnalysis = analyze(snapshot, request_context)
        allowed_provenance = collect_allowed_provenance(snapshot, analysis)

        prompt = build_planning_prompt(
            raw_request, request_context, snapshot, reasoning_result, analysis
        )

        requirements = RequestRequirements(
            task_type=TaskType.STRUCTURED_OUTPUT,
            structured_output=True,
            reasoning=True,
        )
        try:
            # A reply that fails `_parse_and_validate` sends the request to the next provider in the fixed order.
            return call_structured(
                self.provider,
                prompt,
                lambda raw: _parse_and_validate(raw, allowed_provenance),
                requirements=requirements,
            )
        except ProviderError as exc:
            raise ContextPlanningError(
                f"Failed to get a planning directive: {exc}"
            ) from exc
