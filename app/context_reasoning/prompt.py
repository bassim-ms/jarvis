"""
Reasoning prompt construction (M16.3) - the one place raw_request/
RequestContext/ContextSnapshot/DeterministicAnalysis get turned into text
sent to an AI provider. Every serializer here is a controlled, explicit
mapping from a known field to a line of text - never `repr()`/`str()` of
an arbitrary internal object, and never anything beyond what these four
inputs actually hold (no secrets, no unrelated system state, no World
Model access of its own).

`allowed_provenance` (see `collect_allowed_provenance`) is included in
the prompt verbatim, in its own labeled section, specifically so the
instructions can tell the model to cite ONLY these identifiers - this is
what app/context_reasoning/reasoner.py's post-hoc validation then checks
the model's actual response against (belt and suspenders: told AND
enforced, never enforcement alone with no explanation of the rule)."""

from __future__ import annotations

from app.context_engine.base import RequestContext
from app.context_reasoning.deterministic import DeterministicAnalysis
from app.world_model.base import ContextSnapshot

#: Every ContextSnapshot category this prompt builder knows how to
#: serialize, together with the provenance prefix used for its items -
#: mirrors app.core.JarvisCore._WORLD_MODEL_CATEGORIES in spirit, but
#: exposes each item's id (never shown to the end user, but required
#: here so the model can cite it in `provenance` - see this module's own
#: docstring) and is entirely independent of that formatter (this module
#: must never import app.core).
_SNAPSHOT_CATEGORIES: tuple = (
    ("active_projects", "project", lambda p: f"{p.name} (status={p.status.value}, priority={p.priority.value})"),
    ("relevant_goals", "goal", lambda g: f"{g.title} (status={g.status.value}, priority={g.priority.value}, target_date={g.target_date})"),
    ("relevant_tasks", "task_link", lambda t: f"{t.ref_type.value}:{t.ref_id}"),
    ("upcoming_commitments", "commitment", lambda c: f"{c.title} (status={c.status.value}, priority={c.priority.value}, due_at={c.due_at})"),
    ("upcoming_events", "event", lambda e: f"{e.title} (starts_at={e.starts_at}, ends_at={e.ends_at})"),
    ("active_blockers", "blocker", lambda b: f"{b.title} (severity={b.severity.value})"),
    ("relevant_resources", "resource", lambda r: f"{r.name} (status={r.status.value})"),
    ("relevant_preferences", "preference", lambda p: f"{p.key}={p.value}"),
)


def collect_allowed_provenance(
    snapshot: ContextSnapshot, analysis: DeterministicAnalysis
) -> set[str]:
    """Every provenance identifier a ReasoningResult is allowed to cite -
    one per ContextSnapshot item actually selected (`"<category>:<id>"`),
    plus every provenance string DeterministicAnalysis already emitted
    (already grounded in the same snapshot - see app/context_reasoning/
    deterministic.py), plus the two generic tags `"request"` (a claim
    grounded in the raw request/RequestContext itself, not one specific
    World Model item) and `"memory"` (MemoryStore facts, which carry no
    id - see ContextSnapshot.relevant_memory's own docstring). Nothing
    else is ever valid - see app/context_reasoning/reasoner.py's
    anti-hallucination validation, which rejects any provenance string
    outside this set."""
    allowed: set[str] = {"request"}
    for field_name, prefix, _describe in _SNAPSHOT_CATEGORIES:
        for item in getattr(snapshot, field_name):
            allowed.add(f"{prefix}:{item.id}")
    if snapshot.relevant_memory:
        allowed.add("memory")
    for fact in analysis.facts:
        allowed.update(fact.provenance)
    for conflict in analysis.conflicts:
        allowed.update(conflict.provenance)
    return allowed


def _serialize_snapshot(snapshot: ContextSnapshot) -> str:
    lines = []
    for field_name, prefix, describe in _SNAPSHOT_CATEGORIES:
        items = getattr(snapshot, field_name)
        if not items:
            continue
        for item in items:
            lines.append(f"  [{prefix}:{item.id}] {describe(item)}")
    if snapshot.relevant_memory:
        for line in snapshot.relevant_memory:
            lines.append(f"  [memory] {line}")
    if not lines:
        return "(nothing was selected as relevant)"
    return "\n".join(lines)


def _serialize_request_context(request_context: RequestContext) -> str:
    lines = [f"  intent: {request_context.context_intent.value}"]
    for field_name, label in (
        ("mentioned_projects", "mentioned_projects"),
        ("mentioned_goals", "mentioned_goals"),
        ("mentioned_tasks", "mentioned_tasks"),
        ("mentioned_commitments", "mentioned_commitments"),
        ("mentioned_resources", "mentioned_resources"),
    ):
        values = getattr(request_context, field_name)
        if values:
            lines.append(f"  {label}: {', '.join(values)}")
    scope = request_context.time_scope
    if scope.kind.value != "none":
        lines.append(
            f"  time_scope: kind={scope.kind.value} start={scope.start} end={scope.end}"
        )
    return "\n".join(lines)


def _serialize_deterministic_analysis(analysis: DeterministicAnalysis) -> str:
    if analysis.is_empty():
        return "(no deterministic facts or conflicts were found)"
    lines = []
    for fact in analysis.facts:
        lines.append(f"  fact [{', '.join(fact.provenance)}]: {fact.text}")
    for conflict in analysis.conflicts:
        lines.append(f"  conflict [{', '.join(conflict.provenance)}]: {conflict.text}")
    return "\n".join(lines)


_INSTRUCTIONS = (
    "You are a bounded context-reasoning layer for a personal assistant. "
    "You are given the user's raw request, how it was classified, the "
    "context already selected as relevant from the user's Personal World "
    "Model, and facts already derived deterministically from that "
    "context. Your job is ONLY to reason about what this context means "
    "for the request - you never execute anything, never create tasks, "
    "and never mutate any state.\n\n"
    "Strict rules:\n"
    "1. Never invent a project, goal, task, commitment, event, blocker, "
    "resource, preference, memory, deadline, or relationship that isn't "
    "explicitly present in the ALLOWED PROVENANCE list below. Every item "
    "in relevant_facts/constraints/conflicts/opportunities/conclusions "
    "must cite at least one identifier from that exact list in its "
    "\"provenance\" array - never an id, name, or phrase you made up.\n"
    "2. If the supplied context does not resolve a genuine ambiguity in "
    "the request (e.g. it doesn't say which of two named things to "
    "prefer), set requires_clarification to true and give a concrete "
    "clarification_question - never silently pick one. Do not invent "
    "ambiguity that the context already resolves.\n"
    "3. confidence must be a number from 0.0 to 1.0 reflecting your "
    "confidence in the conclusion, never a claim of certainty.\n"
    "4. If there isn't enough information for a field, leave its list "
    "empty or its value null - never fabricate content to fill it.\n\n"
    "Respond with ONLY one JSON object, no prose, no markdown formatting, "
    "in exactly this shape:\n"
    '{"interpretation": "<string>", '
    '"relevant_facts": [{"text": "<string>", "provenance": ["<id>", ...]}], '
    '"constraints": [{"text": "<string>", "provenance": ["<id>", ...]}], '
    '"conflicts": [{"text": "<string>", "provenance": ["<id>", ...]}], '
    '"opportunities": [{"text": "<string>", "provenance": ["<id>", ...]}], '
    '"conclusions": [{"text": "<string>", "provenance": ["<id>", ...]}], '
    '"recommended_next_step": "<string or null>", '
    '"confidence": <0.0-1.0>, '
    '"requires_clarification": <true|false>, '
    '"clarification_question": "<string or null - required if requires_clarification '
    'is true, must be null otherwise>"}'
)


def build_reasoning_prompt(
    raw_request: str,
    request_context: RequestContext,
    snapshot: ContextSnapshot,
    analysis: DeterministicAnalysis,
) -> str:
    """Build the dedicated reasoning prompt - the ONLY place raw_request/
    RequestContext/ContextSnapshot/DeterministicAnalysis are combined
    into text sent to an AI provider (M16.3's own "dedicated, inspectable
    reasoning prompt" requirement). Clearly separates: (1) the user
    request, (2) RequestContext, (3) the selected ContextSnapshot, (4)
    deterministic analysis, (5) allowed provenance, (6) instructions/
    required output - in that exact order, matching this module's own
    docstring."""
    allowed = sorted(collect_allowed_provenance(snapshot, analysis))
    return (
        "USER REQUEST:\n"
        f"  {raw_request}\n\n"
        "REQUEST CONTEXT:\n"
        f"{_serialize_request_context(request_context)}\n\n"
        "SELECTED CONTEXT (from the user's Personal World Model):\n"
        f"{_serialize_snapshot(snapshot)}\n\n"
        "DETERMINISTIC ANALYSIS (already computed, no need to re-derive):\n"
        f"{_serialize_deterministic_analysis(analysis)}\n\n"
        "ALLOWED PROVENANCE (the only identifiers you may cite in any "
        "\"provenance\" array):\n"
        f"  {', '.join(allowed) if allowed else '(none - request only)'}\n\n"
        f"{_INSTRUCTIONS}"
    )
