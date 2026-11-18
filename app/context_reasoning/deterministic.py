"""
Deterministic analysis (M16.3) - facts derivable directly, reliably, and
explainably from a ContextSnapshot (M16.2) and its own RequestContext,
with NO AI call. `analyze()` is the one entry point; everything else here
is a private helper it uses.

This intentionally reuses app.context_reasoning.base.RelevantFact/Conflict
as its own output shape - a deterministic fact IS a RelevantFact, just one
produced by plain Python comparisons instead of an LLM, and reusing the
same type means app/context_reasoning/reasoner.py can feed these straight
into a ReasoningResult (for the empty-LLM-call short circuit) or into the
reasoning prompt (for the LLM to build on) without a second parallel shape.

Every fact/conflict's `provenance` names the exact ContextSnapshot item(s)
it came from (e.g. "commitment:<id>", "event:<id>") or "request" for one
derived from RequestContext.time_scope itself - never a made-up id, and
never anything not already present in the snapshot/request this module
was given. This module never queries the World Model or MemoryStore
itself - it only ever reads the ContextSnapshot/RequestContext it's
handed, exactly like app.context_engine.engine.ContextEngine's own
read-only discipline, one level downstream.

Bounded scope, matching this milestone's own explicit list: overdue
items, approaching deadlines, active blockers, project/task and
goal/task relationships, commitment/event-time conflicts, explicit
priorities, relevant time windows, and dependency relationships (read
off ContextSnapshot.retrieval_metadata's own REASON_DEPENDENCY signal -
see app/context_engine/scoring.py - never a fresh World Model query for
GoalDependency rows, which app.context_reasoning must never make)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.context_engine.base import RequestContext, TimeScopeKind
from app.context_reasoning.base import Conflict, RelevantFact
from app.world_model.base import (
    Blocker,
    Commitment,
    CommitmentStatus,
    ContextSnapshot,
    Event,
    Goal,
    Priority,
    Project,
)

#: How close to `now` a due/target/start time has to be to count as an
#: "approaching deadline" - a fixed, documented window, the same
#: "small fixed number, never inferred" discipline app.context_engine.
#: scoring.RelevanceWeights already uses.
_APPROACHING_WINDOW = timedelta(hours=48)

_REASON_DEPENDENCY = "dependency"


def _parse(timestamp: str | None) -> datetime | None:
    """An AWARE datetime, so every comparison here is aware-with-aware: a value without a timezone is read as UTC (this
    module's `now` is the snapshot's UTC time), and a plain date ("2026-11-30" - a goal's target date) means the end of that
    day. Before this, a date-only target date compared with the aware `now` raised a TypeError and failed the chat request."""
    if not timestamp:
        return None
    text = timestamp.strip()
    try:
        value = datetime.fromisoformat(text)
    except ValueError:
        return None
    if len(text) == 10 and "T" not in text:
        value = value.replace(hour=23, minute=59, second=59)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


@dataclass(frozen=True)
class DeterministicAnalysis:
    """The bounded set of facts/conflicts this module could derive
    without an AI call, for one (ContextSnapshot, RequestContext) pair.
    Both fields reuse app.context_reasoning.base's own RelevantFact/
    Conflict shape - see this module's own docstring for why."""

    facts: tuple[RelevantFact, ...] = ()
    conflicts: tuple[Conflict, ...] = ()

    def is_empty(self) -> bool:
        return not self.facts and not self.conflicts


def _overdue_commitments(commitments: tuple[Commitment, ...], now: datetime) -> list[RelevantFact]:
    facts = []
    for c in commitments:
        due = _parse(c.due_at)
        if due is not None and due < now and c.status is CommitmentStatus.OPEN:
            facts.append(
                RelevantFact(
                    text=f"Commitment '{c.title}' is overdue (was due {c.due_at}).",
                    provenance=(f"commitment:{c.id}",),
                )
            )
    return facts


def _approaching_deadlines(
    goals: tuple[Goal, ...],
    commitments: tuple[Commitment, ...],
    events: tuple[Event, ...],
    now: datetime,
) -> list[RelevantFact]:
    facts = []
    for g in goals:
        target = _parse(g.target_date)
        if target is not None and now <= target <= now + _APPROACHING_WINDOW:
            facts.append(
                RelevantFact(
                    text=f"Goal '{g.title}' has an approaching target date ({g.target_date}).",
                    provenance=(f"goal:{g.id}",),
                )
            )
    for c in commitments:
        due = _parse(c.due_at)
        if (
            due is not None
            and now <= due <= now + _APPROACHING_WINDOW
            and c.status is CommitmentStatus.OPEN
        ):
            facts.append(
                RelevantFact(
                    text=f"Commitment '{c.title}' is due soon ({c.due_at}).",
                    provenance=(f"commitment:{c.id}",),
                )
            )
    for e in events:
        starts = _parse(e.starts_at)
        if starts is not None and now <= starts <= now + _APPROACHING_WINDOW:
            facts.append(
                RelevantFact(
                    text=f"Event '{e.title}' starts soon ({e.starts_at}).",
                    provenance=(f"event:{e.id}",),
                )
            )
    return facts


def _active_blockers(blockers: tuple[Blocker, ...]) -> list[RelevantFact]:
    return [
        RelevantFact(
            text=f"Blocker '{b.title}' is active (severity: {b.severity.value}).",
            provenance=(f"blocker:{b.id}",),
        )
        for b in blockers
    ]


def _relationships(
    projects: tuple[Project, ...],
    goals: tuple[Goal, ...],
) -> list[RelevantFact]:
    project_names = {p.id: p.name for p in projects}
    facts = []
    for g in goals:
        project_name = project_names.get(g.project_id)
        if project_name is not None:
            facts.append(
                RelevantFact(
                    text=f"Goal '{g.title}' belongs to project '{project_name}'.",
                    provenance=(f"goal:{g.id}", f"project:{g.project_id}"),
                )
            )
    return facts


def _explicit_priorities(
    projects: tuple[Project, ...],
    goals: tuple[Goal, ...],
    commitments: tuple[Commitment, ...],
) -> list[RelevantFact]:
    facts = []
    for p in projects:
        if p.priority is Priority.HIGH:
            facts.append(
                RelevantFact(
                    text=f"Project '{p.name}' is a high priority.",
                    provenance=(f"project:{p.id}",),
                )
            )
    for g in goals:
        if g.priority is Priority.HIGH:
            facts.append(
                RelevantFact(
                    text=f"Goal '{g.title}' is a high priority.",
                    provenance=(f"goal:{g.id}",),
                )
            )
    for c in commitments:
        if c.priority is Priority.HIGH:
            facts.append(
                RelevantFact(
                    text=f"Commitment '{c.title}' is a high priority.",
                    provenance=(f"commitment:{c.id}",),
                )
            )
    return facts


def _time_window(request_context: RequestContext) -> list[RelevantFact]:
    scope = request_context.time_scope
    if scope.kind is TimeScopeKind.NONE:
        return []
    description = f"The request implies a time window of kind '{scope.kind.value}'"
    if scope.start or scope.end:
        description += f" ({scope.start or '...'} to {scope.end or '...'})"
    description += "."
    return [RelevantFact(text=description, provenance=("request",))]


def _dependency_relationships(
    goals: tuple[Goal, ...], snapshot: ContextSnapshot
) -> list[RelevantFact]:
    """Reads ContextSnapshot.retrieval_metadata["goals"] (populated by
    ContextEngine - app/context_engine/engine.py) for the REASON_DEPENDENCY
    signal it already computed - never a fresh GoalDependency query, which
    app.context_reasoning must never make (M16.3's own World Model
    boundary)."""
    goal_metadata = snapshot.retrieval_metadata.get("goals")
    if goal_metadata is None:
        return []
    facts = []
    for g in goals:
        provenance = goal_metadata.items.get(g.id)
        if provenance is not None and _REASON_DEPENDENCY in provenance.reasons:
            facts.append(
                RelevantFact(
                    text=f"Goal '{g.title}' has a dependency relationship with another goal.",
                    provenance=(f"goal:{g.id}",),
                )
            )
    return facts


def _event_conflicts(events: tuple[Event, ...]) -> list[Conflict]:
    conflicts = []
    parsed = [(e, _parse(e.starts_at), _parse(e.ends_at)) for e in events]
    for i, (event_a, start_a, end_a) in enumerate(parsed):
        if start_a is None or end_a is None:
            continue
        for event_b, start_b, end_b in parsed[i + 1 :]:
            if start_b is None or end_b is None:
                continue
            if start_a < end_b and start_b < end_a:
                conflicts.append(
                    Conflict(
                        text=f"Event '{event_a.title}' overlaps with event '{event_b.title}'.",
                        provenance=(f"event:{event_a.id}", f"event:{event_b.id}"),
                    )
                )
    return conflicts


def _commitment_event_conflicts(
    commitments: tuple[Commitment, ...], events: tuple[Event, ...]
) -> list[Conflict]:
    conflicts = []
    for c in commitments:
        due = _parse(c.due_at)
        if due is None or c.status is not CommitmentStatus.OPEN:
            continue
        for e in events:
            starts, ends = _parse(e.starts_at), _parse(e.ends_at)
            if starts is not None and ends is not None and starts <= due < ends:
                conflicts.append(
                    Conflict(
                        text=(
                            f"Commitment '{c.title}' is due during event '{e.title}'."
                        ),
                        provenance=(f"commitment:{c.id}", f"event:{e.id}"),
                    )
                )
    return conflicts


def analyze(
    snapshot: ContextSnapshot,
    request_context: RequestContext,
) -> DeterministicAnalysis:
    """Derive every deterministic fact/conflict this module knows how to
    find in `snapshot`/`request_context` - bounded, explainable, no AI
    call. `snapshot.current_time` is used as `now` (the same value
    ContextEngine.build_snapshot already anchored every relevance score
    to - see app/context_engine/engine.py), never a fresh clock read, so
    this analysis is always consistent with the snapshot it's analyzing."""
    # `snapshot.current_time` is always a valid ISO 8601 UTC timestamp in
    # practice (app/context_engine/engine.py's `_now_iso`) - the `or`
    # fallback is defensive only, never a naive local clock read.
    now = _parse(snapshot.current_time) or datetime.now(timezone.utc)

    facts: list[RelevantFact] = []
    facts += _overdue_commitments(tuple(snapshot.upcoming_commitments), now)
    facts += _approaching_deadlines(
        tuple(snapshot.relevant_goals),
        tuple(snapshot.upcoming_commitments),
        tuple(snapshot.upcoming_events),
        now,
    )
    facts += _active_blockers(tuple(snapshot.active_blockers))
    facts += _relationships(tuple(snapshot.active_projects), tuple(snapshot.relevant_goals))
    facts += _explicit_priorities(
        tuple(snapshot.active_projects),
        tuple(snapshot.relevant_goals),
        tuple(snapshot.upcoming_commitments),
    )
    facts += _time_window(request_context)
    facts += _dependency_relationships(tuple(snapshot.relevant_goals), snapshot)

    conflicts: list[Conflict] = []
    conflicts += _event_conflicts(tuple(snapshot.upcoming_events))
    conflicts += _commitment_event_conflicts(
        tuple(snapshot.upcoming_commitments), tuple(snapshot.upcoming_events)
    )

    return DeterministicAnalysis(facts=tuple(facts), conflicts=tuple(conflicts))
