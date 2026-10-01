"""
Planning prompt construction (M16.4) - the one place raw_request/
RequestContext/ContextSnapshot/ReasoningResult get turned into text sent
to an AI provider for strategic planning classification. Every
serializer here is a controlled, explicit mapping from a known field to
a line of text - never `repr()`/`str()` of an arbitrary internal object.

Reuses app.context_reasoning.deterministic.analyze and
app.context_reasoning.prompt.collect_allowed_provenance directly (both
already public, pure, deterministic functions) rather than
re-implementing deterministic analysis or the provenance whitelist a
second time - M16.4 reasons over the SAME grounded context M16.3 already
validated its own output against, one layer up."""

from __future__ import annotations

from app.context_engine.base import RequestContext
from app.context_reasoning.base import ReasoningResult
from app.context_reasoning.deterministic import DeterministicAnalysis
from app.context_reasoning.prompt import collect_allowed_provenance, serialize_sending_device
from app.world_model.base import ContextSnapshot


def _serialize_request_context(request_context: RequestContext) -> str:
    lines = [f"  intent: {request_context.context_intent.value}"]
    for field_name in (
        "mentioned_projects",
        "mentioned_goals",
        "mentioned_tasks",
        "mentioned_commitments",
        "mentioned_resources",
    ):
        values = getattr(request_context, field_name)
        if values:
            lines.append(f"  {field_name}: {', '.join(values)}")
    scope = request_context.time_scope
    if scope.kind.value != "none":
        lines.append(
            f"  time_scope: kind={scope.kind.value} start={scope.start} end={scope.end}"
        )
    return "\n".join(lines)


def _serialize_reasoning_result(reasoning_result: ReasoningResult) -> str:
    lines = [f"  interpretation: {reasoning_result.interpretation}"]
    for field_name, label in (
        ("relevant_facts", "relevant_facts"),
        ("constraints", "constraints"),
        ("conflicts", "conflicts"),
        ("opportunities", "opportunities"),
        ("conclusions", "conclusions"),
    ):
        items = getattr(reasoning_result, field_name)
        for item in items:
            lines.append(f"  {label} [{', '.join(item.provenance)}]: {item.text}")
    if reasoning_result.recommended_next_step:
        lines.append(f"  recommended_next_step: {reasoning_result.recommended_next_step}")
    lines.append(f"  confidence: {reasoning_result.confidence}")
    lines.append(f"  requires_clarification: {reasoning_result.requires_clarification}")
    if reasoning_result.clarification_question:
        lines.append(f"  clarification_question: {reasoning_result.clarification_question}")
    return "\n".join(lines)


_INSTRUCTIONS = (
    "You are a bounded, strategic context-aware PLANNING layer for a "
    "personal assistant. You are given the user's raw request, how it "
    "was classified, the context M16.2 already selected as relevant, "
    "and the reasoning M16.3 already produced about it. Your job is "
    "ONLY to classify what kind of planning strategy this request calls "
    "for and describe it - you never execute anything, never create a "
    "task/job, and never produce an executable step-by-step plan of "
    "your own (that is a separate, existing system's job).\n\n"
    "Choose exactly one planning_type:\n"
    '- "NONE": purely informational - nothing to plan.\n'
    '- "DECISION": the user is choosing between options - frame the '
    "decision, never pick one for them.\n"
    '- "ACTION": one self-contained thing to get done.\n'
    '- "MULTI_STEP": accomplishing this genuinely involves several '
    "dependent parts, especially once weighed against other context "
    "(competing commitments, blockers, resources).\n\n"
    "Strict rules:\n"
    "1. Never invent a project, goal, task, commitment, event, blocker, "
    "resource, preference, memory, deadline, or relationship that isn't "
    "explicitly present in the ALLOWED PROVENANCE list below. Every "
    "entry in contextual_constraints/relevant_resources/dependencies "
    "must cite at least one identifier from that exact list in its "
    '"provenance" array.\n'
    "2. requires_execution may only be true when planning_type is "
    '"ACTION" or "MULTI_STEP" - it must be false for "NONE"/"DECISION".\n'
    "3. confidence must be a number from 0.0 to 1.0.\n"
    "4. If there isn't enough information for a field, leave its list "
    "empty or its value null - never fabricate content to fill it.\n\n"
    "Respond with ONLY one JSON object, no prose, no markdown formatting, "
    "in exactly this shape:\n"
    '{"objective": "<string>", '
    '"planning_type": "NONE|DECISION|ACTION|MULTI_STEP", '
    '"contextual_constraints": [{"text": "<string>", "provenance": ["<id>", ...]}], '
    '"relevant_resources": [{"text": "<string>", "provenance": ["<id>", ...]}], '
    '"dependencies": [{"text": "<string>", "provenance": ["<id>", ...]}], '
    '"assumptions": ["<string>", ...], '
    '"suggested_approach": "<string or null>", '
    '"requires_execution": <true|false>, '
    '"confidence": <0.0-1.0>}'
)


def build_planning_prompt(
    raw_request: str,
    request_context: RequestContext,
    snapshot: ContextSnapshot,
    reasoning_result: ReasoningResult,
    analysis: DeterministicAnalysis,
) -> str:
    """Build the dedicated planning prompt - the ONLY place raw_request/
    RequestContext/ContextSnapshot/ReasoningResult are combined into
    text sent to an AI provider for M16.4's own classification.
    `analysis` is the SAME app.context_reasoning.deterministic.
    DeterministicAnalysis the caller (app/context_planning/planner.py)
    already computed once for the allowed-provenance whitelist -
    computed once, reused for both, never a second, independent
    analysis architecture."""
    allowed = sorted(collect_allowed_provenance(snapshot, analysis))
    return (
        "USER REQUEST:\n"
        f"  {raw_request}\n\n"
        "REQUEST CONTEXT:\n"
        f"{_serialize_request_context(request_context)}\n\n"
        f"{serialize_sending_device(request_context, snapshot)}"
        "REASONING (M16.3 - already produced, do not re-derive):\n"
        f"{_serialize_reasoning_result(reasoning_result)}\n\n"
        "ALLOWED PROVENANCE (the only identifiers you may cite in any "
        "\"provenance\" array):\n"
        f"  {', '.join(allowed) if allowed else '(none - request only)'}\n\n"
        f"{_INSTRUCTIONS}"
    )
