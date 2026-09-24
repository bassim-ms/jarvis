"""
ContextReasoner (M16.3): the abstraction and LLM-backed implementation
that turns (raw_request, RequestContext, ContextSnapshot) into a
validated ReasoningResult.

STRICTLY READ-ONLY AND NON-EXECUTING: nothing here ever calls
ToolRegistry.execute, creates a Task/Job, mutates the World Model, or
performs any external action - it only ever reads what it's handed and
returns a ReasoningResult for "existing response generation" (app/
core.py) to use. It never independently queries the World Model or
MemoryStore - see app/context_reasoning/deterministic.py's own read-only
discipline, one level downstream, for why this module never bypasses
M16.2.

Uses the existing AIProvider/provider-routing architecture (app/
providers/) exactly as app.capability.GeminiCapabilityDecisionProvider
and app.tools.decision.GeminiToolCallProvider already do for their own
strict-JSON needs - via `call_with_requirements` with
RequestRequirements(structured_output=True), which ProviderRouter
(app/providers/router.py) hard-gates to a provider that's actually been
engineered and validated for that (today, Gemini - see
app/providers/capabilities.py) - never a second AI client, never a
hardcoded provider class here. `LLMContextReasoner` takes its
AIProvider as a required constructor argument rather than defaulting to
one internally, so it is always routed through whatever provider/
provider-manager the caller (normally JarvisCore) already has - see
app/context_reasoning/factory.py.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

from app.context_engine.base import RequestContext
from app.context_reasoning.base import (
    Conclusion,
    Conflict,
    Constraint,
    ContextReasoningError,
    Opportunity,
    RelevantFact,
    ReasoningResult,
)
from app.context_reasoning.deterministic import DeterministicAnalysis, analyze
from app.context_reasoning.prompt import build_reasoning_prompt, collect_allowed_provenance
from app.providers.base import AIProvider, ProviderError
from app.providers.manager import call_with_requirements
from app.providers.requirements import RequestRequirements, TaskType
from app.world_model.base import ContextSnapshot


class ContextReasonerError(RuntimeError):
    """Raised when a ContextReasoner can't produce a valid ReasoningResult
    - a failed AI call, malformed output, output that fails strict schema
    validation, or output citing provenance outside what was actually
    supplied (the anti-hallucination gate). Never silently guessed
    around - the same CapabilityDecisionError/ToolCallError convention
    (app/capability.py, app/tools/decision.py) applied to M16.3's own
    boundary."""


class ContextReasoner(ABC):
    """Answers "given what the user asked and the context M16.2 already
    selected, what does that context mean for this request?" - never
    executes anything, never queries the World Model/MemoryStore itself
    (only ever reads what it's given), and never re-derives or bypasses
    the ContextSnapshot it's handed (M16.2 remains the sole retrieval
    authority)."""

    @abstractmethod
    def reason(
        self,
        raw_request: str,
        request_context: RequestContext,
        snapshot: ContextSnapshot,
    ) -> ReasoningResult:
        """Return a validated ReasoningResult for `raw_request`, given
        the `request_context`/`snapshot` M16.2 already built (never
        re-derived here). Raises ContextReasonerError on failure - never
        a fabricated or partially-guessed ReasoningResult."""


_SNAPSHOT_FIELDS = (
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


def _snapshot_is_empty(snapshot: ContextSnapshot) -> bool:
    """True when every ContextSnapshot category is empty - nothing was
    found relevant to this request anywhere in the World Model. Mirrors
    the same "empty means nothing to add" judgment app.core.JarvisCore.
    _format_world_model_context already makes for M16.2 - this module
    must never import app.core, so it re-states the same nine-field
    check independently rather than sharing that method."""
    return all(not getattr(snapshot, field_name) for field_name in _SNAPSHOT_FIELDS)


def _trivial_result() -> ReasoningResult:
    """The fixed, minimal ReasoningResult returned with NO provider call
    when `_snapshot_is_empty` - there is nothing selected to reason
    about, so calling an LLM anyway would only invite fabrication and
    cost for no benefit. `is_empty()` is true for this result (see
    app/context_reasoning/base.py), which is exactly what lets app/
    core.py's own `_format_reasoning_result` preserve byte-identical
    prompt text for a request whose ContextSnapshot had nothing relevant
    - this milestone's own explicit empty-context regression
    requirement, at the layer directly above ContextEngine's own."""
    return ReasoningResult(
        interpretation="No relevant world-model context was available for this request."
    )


_VALID_TOP_LEVEL_KEYS = {
    "interpretation",
    "relevant_facts",
    "constraints",
    "conflicts",
    "opportunities",
    "conclusions",
    "recommended_next_step",
    "confidence",
    "requires_clarification",
    "clarification_question",
}

_LIST_FIELD_TYPES: dict[str, type] = {
    "relevant_facts": RelevantFact,
    "constraints": Constraint,
    "conflicts": Conflict,
    "opportunities": Opportunity,
    "conclusions": Conclusion,
}

# The model often wraps JSON in a markdown code fence even when told not
# to; stripping one is tolerant formatting cleanup, not guessing at
# content - the same tolerance app.capability/app.tools.decision already
# apply to their own strict-JSON providers.
_CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _parse_item_list(
    payload: dict,
    field_name: str,
    item_cls: type,
    allowed_provenance: set[str],
    raw: str,
) -> tuple:
    items = payload.get(field_name, [])
    if not isinstance(items, list):
        raise ContextReasonerError(f"'{field_name}' must be a JSON array, got: {raw!r}")

    parsed = []
    for entry in items:
        if not isinstance(entry, dict):
            raise ContextReasonerError(
                f"Each '{field_name}' entry must be a JSON object, got: {raw!r}"
            )
        extra = set(entry) - {"text", "provenance"}
        if extra:
            raise ContextReasonerError(
                f"Unexpected fields in a '{field_name}' entry: {sorted(extra)}"
            )
        text = entry.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ContextReasonerError(
                f"Each '{field_name}' entry needs a non-empty 'text', got: {raw!r}"
            )
        provenance = entry.get("provenance")
        if (
            not isinstance(provenance, list)
            or not provenance
            or any(not isinstance(p, str) or not p for p in provenance)
        ):
            raise ContextReasonerError(
                f"Each '{field_name}' entry needs a non-empty 'provenance' array of "
                f"strings (every claim must be grounded), got: {raw!r}"
            )
        unsupported = [p for p in provenance if p not in allowed_provenance]
        if unsupported:
            raise ContextReasonerError(
                f"'{field_name}' entry cites provenance not present in the supplied "
                f"context (possible hallucination): {unsupported!r}. Full response: {raw!r}"
            )
        parsed.append(item_cls(text=text, provenance=tuple(provenance)))
    return tuple(parsed)


def _parse_and_validate(raw: str, allowed_provenance: set[str]) -> ReasoningResult:
    cleaned = _CODE_FENCE_PATTERN.sub("", raw.strip()).strip()

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ContextReasonerError(f"Invalid JSON from context reasoner: {raw!r}") from exc

    if not isinstance(payload, dict):
        raise ContextReasonerError(f"Expected a JSON object, got: {raw!r}")

    extra_keys = set(payload) - _VALID_TOP_LEVEL_KEYS
    if extra_keys:
        raise ContextReasonerError(
            f"Unexpected fields in reasoning result: {sorted(extra_keys)}"
        )

    interpretation = payload.get("interpretation")
    if not isinstance(interpretation, str) or not interpretation.strip():
        raise ContextReasonerError(f"Missing or invalid 'interpretation': {raw!r}")

    list_fields = {
        field_name: _parse_item_list(payload, field_name, item_cls, allowed_provenance, raw)
        for field_name, item_cls in _LIST_FIELD_TYPES.items()
    }

    recommended_next_step = payload.get("recommended_next_step")
    if recommended_next_step is not None and (
        not isinstance(recommended_next_step, str) or not recommended_next_step.strip()
    ):
        raise ContextReasonerError(f"Invalid 'recommended_next_step': {raw!r}")

    confidence = payload.get("confidence")
    if (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not (0.0 <= confidence <= 1.0)
    ):
        raise ContextReasonerError(
            f"'confidence' must be a number in [0.0, 1.0], got: {raw!r}"
        )

    requires_clarification = payload.get("requires_clarification")
    if not isinstance(requires_clarification, bool):
        raise ContextReasonerError(
            f"Missing or invalid 'requires_clarification' (must be a bool): {raw!r}"
        )

    clarification_question = payload.get("clarification_question")
    if requires_clarification:
        if not isinstance(clarification_question, str) or not clarification_question.strip():
            raise ContextReasonerError(
                f"'clarification_question' must be a non-empty string when "
                f"requires_clarification is true: {raw!r}"
            )
    elif clarification_question is not None:
        raise ContextReasonerError(
            f"'clarification_question' must be null when requires_clarification is "
            f"false, got: {raw!r}"
        )

    try:
        return ReasoningResult(
            interpretation=interpretation,
            relevant_facts=list_fields["relevant_facts"],
            constraints=list_fields["constraints"],
            conflicts=list_fields["conflicts"],
            opportunities=list_fields["opportunities"],
            conclusions=list_fields["conclusions"],
            recommended_next_step=recommended_next_step,
            confidence=float(confidence),
            requires_clarification=requires_clarification,
            clarification_question=clarification_question,
        )
    except ContextReasoningError as exc:
        raise ContextReasonerError(f"Reasoning result failed validation: {exc}") from exc


class LLMContextReasoner(ContextReasoner):
    """ContextReasoner backed by the existing AIProvider/provider-routing
    architecture - see this module's own docstring. `provider` is
    required (no internal default/hardcoded vendor): the caller supplies
    whichever AIProvider/AIProviderManager it already has.

    `reason()` first runs app.context_reasoning.deterministic.analyze()
    (always, no AI call). If the ContextSnapshot is entirely empty (see
    `_snapshot_is_empty`), returns `_trivial_result()` with NO provider
    call - see that function's own docstring for why. Otherwise, builds
    the dedicated reasoning prompt (app/context_reasoning/prompt.py) and
    calls the provider with a structured-output requirement, then
    strictly parses/validates the response - including that every
    provenance identifier cited actually appears in what was supplied
    (`collect_allowed_provenance`) - before constructing a
    ReasoningResult. Any failure raises ContextReasonerError."""

    def __init__(self, provider: AIProvider):
        self.provider = provider

    def reason(
        self,
        raw_request: str,
        request_context: RequestContext,
        snapshot: ContextSnapshot,
    ) -> ReasoningResult:
        analysis: DeterministicAnalysis = analyze(snapshot, request_context)

        if _snapshot_is_empty(snapshot):
            return _trivial_result()

        prompt = build_reasoning_prompt(raw_request, request_context, snapshot, analysis)
        allowed_provenance = collect_allowed_provenance(snapshot, analysis)

        requirements = RequestRequirements(
            task_type=TaskType.STRUCTURED_OUTPUT,
            structured_output=True,
            reasoning=True,
        )
        try:
            raw = call_with_requirements(self.provider, prompt, requirements)
        except ProviderError as exc:
            raise ContextReasonerError(
                f"Failed to get a reasoning result: {exc}"
            ) from exc

        return _parse_and_validate(raw, allowed_provenance)
