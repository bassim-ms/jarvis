"""
Deterministic relevance scoring (M16.2) - no LLM, no embeddings, no
vector similarity, no generic graph traversal. Every function here is a
pure, additive combination of a small, bounded, documented set of
weighted signals (RelevanceWeights) - the same "small fixed numbers,
never inferred, never learned" discipline app/providers/router.py's own
soft-scoring already uses for provider selection.

The one rule every signal below is designed to uphold: **explicit user
intent must dominate inferred relevance**. RelevanceWeights.EXPLICIT_MATCH
is deliberately far larger than the sum of every other signal a single
entity could plausibly accumulate, so a candidate the user actually named
always outranks one merely inferred to be relevant - see
`test_explicit_match_dominates_other_signals` (tests/test_context_engine_
scoring.py) for the property this is meant to guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from app.world_model.base import (
    Blocker,
    BlockerStatus,
    Commitment,
    Event,
    Goal,
    GoalStatus,
    Priority,
    Project,
    ProjectStatus,
    Resource,
)


@dataclass(frozen=True)
class RelevanceWeights:
    """Every relevance weight this module uses, in one place - never a
    magic number scattered across scoring functions. Bounded and
    documented: EXPLICIT_MATCH is intentionally far above the ceiling any
    combination of every other signal could reach for one entity (at
    most, roughly RELATIONSHIP + CURRENT_STATE_ACTIVE + PRIORITY_HIGH +
    TIME_NOW + DEPENDENCY + BLOCKER_PRESENT + RESOURCE_MATCH ≈ 96 <
    100), so an explicitly-named candidate always outranks a merely
    inferred one, regardless of how many inferred signals stack up."""

    EXPLICIT_MATCH: float = 100.0
    RELATIONSHIP: float = 20.0
    CURRENT_STATE_ACTIVE: float = 10.0
    CURRENT_STATE_PAUSED: float = 4.0
    PRIORITY_HIGH: float = 8.0
    PRIORITY_MEDIUM: float = 4.0
    PRIORITY_LOW: float = 1.0
    TIME_NOW: float = 32.0
    TIME_OVERDUE: float = 28.0
    TIME_TODAY: float = 24.0
    TIME_TOMORROW: float = 16.0
    TIME_THIS_WEEK: float = 10.0
    TIME_FAR_FUTURE: float = 2.0
    DEPENDENCY: float = 6.0
    BLOCKER_PRESENT: float = 15.0
    RESOURCE_MATCH: float = 5.0


#: The single shared weights instance every scoring function below uses -
#: a caller needing different weights constructs their own
#: RelevanceWeights and passes it through, rather than this module
#: growing per-caller special cases.
DEFAULT_WEIGHTS = RelevanceWeights()

# Fixed, deterministic reason identifiers - never LLM-generated prose
# (M16.2's own explicit requirement). Each one names exactly one signal
# from the list above so a caller can trace a score back to what
# produced it.
REASON_EXPLICIT_MATCH = "explicit_match"
REASON_ACTIVE_STATE = "active_state"
REASON_PAUSED_STATE = "paused_state"
REASON_PRIORITY_HIGH = "priority_high"
REASON_PRIORITY_MEDIUM = "priority_medium"
REASON_PRIORITY_LOW = "priority_low"
REASON_TIME_NOW = "time_now"
REASON_TIME_OVERDUE = "time_overdue"
REASON_TIME_TODAY = "time_today"
REASON_TIME_TOMORROW = "time_tomorrow"
REASON_TIME_THIS_WEEK = "time_this_week"
REASON_TIME_FAR_FUTURE = "time_far_future"
REASON_DEPENDENCY = "dependency"
REASON_BLOCKER_PRESENT = "has_active_blocker"
REASON_RESOURCE_MATCH = "resource_match"
REASON_RELATIONSHIP_PROJECT = "relationship_to_project"
REASON_RELATIONSHIP_GOAL = "relationship_to_goal"


def matches_hint(hint: str, name: str) -> bool:
    """Deterministic, case-insensitive containment match - the ONLY
    matching strategy this module uses (no fuzzy/similarity matching; no
    existing name-matching utility exists elsewhere in this codebase to
    reuse - see app/context_engine/request_builder.py's own module
    docstring). Either direction counts a match: a short hint fully
    contained in a longer name ("robotics" in "Robotics Arm Project"),
    or a hint that already contains the whole name (the user typed the
    full name plus extra words)."""
    h = hint.strip().lower()
    n = name.strip().lower()
    if not h or not n:
        return False
    return h in n or n in h


def any_hint_matches(hints: list[str], name: str) -> bool:
    return any(matches_hint(hint, name) for hint in hints)


def priority_relevance(
    priority: Priority, weights: RelevanceWeights = DEFAULT_WEIGHTS
) -> tuple[float, str]:
    if priority is Priority.HIGH:
        return weights.PRIORITY_HIGH, REASON_PRIORITY_HIGH
    if priority is Priority.MEDIUM:
        return weights.PRIORITY_MEDIUM, REASON_PRIORITY_MEDIUM
    return weights.PRIORITY_LOW, REASON_PRIORITY_LOW


def project_state_relevance(
    status: ProjectStatus, weights: RelevanceWeights = DEFAULT_WEIGHTS
) -> tuple[float, str] | None:
    if status is ProjectStatus.ACTIVE:
        return weights.CURRENT_STATE_ACTIVE, REASON_ACTIVE_STATE
    if status is ProjectStatus.PAUSED:
        return weights.CURRENT_STATE_PAUSED, REASON_PAUSED_STATE
    return None  # COMPLETED/ARCHIVED - only ever seen here via explicit override


def goal_state_relevance(
    status: GoalStatus, weights: RelevanceWeights = DEFAULT_WEIGHTS
) -> tuple[float, str] | None:
    if status is GoalStatus.ACTIVE:
        return weights.CURRENT_STATE_ACTIVE, REASON_ACTIVE_STATE
    if status is GoalStatus.PAUSED:
        return weights.CURRENT_STATE_PAUSED, REASON_PAUSED_STATE
    return None


def time_relevance(
    timestamp: str | None,
    current_time: str,
    weights: RelevanceWeights = DEFAULT_WEIGHTS,
) -> tuple[float, str] | None:
    """Deterministic time-proximity scoring for a due/start timestamp
    against `current_time` (both ISO 8601 strings - this codebase's
    existing timestamp convention). Returns None when `timestamp` is
    missing or unparseable, so a caller with no date to score simply
    adds nothing rather than a fabricated default.

    Overdue is deliberately scored HIGH (`weights.TIME_OVERDUE`), never
    zero/excluded - an item whose due time already passed is often MORE
    relevant, not less (M16.2's own explicit requirement)."""
    if not timestamp:
        return None
    try:
        ts = datetime.fromisoformat(timestamp)
        now = datetime.fromisoformat(current_time)
    except ValueError:
        return None
    # A plain date ("2026-11-30" - a goal's target date) means the END of that day: due that day, not overdue from its first
    # minute. A value without a timezone is read in the same zone as the other one, so a naive and an aware timestamp compare
    # (before this, a goal with a date-only target date made every chat request fail with a TypeError).
    if "T" not in timestamp.strip() and len(timestamp.strip()) == 10:
        ts = ts.replace(hour=23, minute=59, second=59)
    if ts.tzinfo is None and now.tzinfo is not None:
        ts = ts.replace(tzinfo=now.tzinfo)
    elif now.tzinfo is None and ts.tzinfo is not None:
        now = now.replace(tzinfo=ts.tzinfo)

    if ts < now:
        return weights.TIME_OVERDUE, REASON_TIME_OVERDUE
    if ts - now <= timedelta(hours=1):
        return weights.TIME_NOW, REASON_TIME_NOW
    if ts.date() == now.date():
        return weights.TIME_TODAY, REASON_TIME_TODAY
    if ts.date() == (now + timedelta(days=1)).date():
        return weights.TIME_TOMORROW, REASON_TIME_TOMORROW
    if ts - now <= timedelta(days=7):
        return weights.TIME_THIS_WEEK, REASON_TIME_THIS_WEEK
    return weights.TIME_FAR_FUTURE, REASON_TIME_FAR_FUTURE


def score_project(
    project: Project,
    *,
    explicit_hints: list[str],
    current_time: str,
    has_active_blocker: bool = False,
    weights: RelevanceWeights = DEFAULT_WEIGHTS,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    if any_hint_matches(explicit_hints, project.name):
        score += weights.EXPLICIT_MATCH
        reasons.append(REASON_EXPLICIT_MATCH)

    state = project_state_relevance(project.status, weights)
    if state is not None:
        s, reason = state
        score += s
        reasons.append(reason)

    p_score, p_reason = priority_relevance(project.priority, weights)
    score += p_score
    reasons.append(p_reason)

    if has_active_blocker:
        score += weights.BLOCKER_PRESENT
        reasons.append(REASON_BLOCKER_PRESENT)

    return score, reasons


def score_goal(
    goal: Goal,
    *,
    explicit_hints: list[str],
    project_explicitly_matched: bool,
    current_time: str,
    has_active_blocker: bool = False,
    has_dependency: bool = False,
    weights: RelevanceWeights = DEFAULT_WEIGHTS,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    if any_hint_matches(explicit_hints, goal.title):
        score += weights.EXPLICIT_MATCH
        reasons.append(REASON_EXPLICIT_MATCH)
    elif project_explicitly_matched:
        # The goal's OWN title wasn't named, but its parent project was -
        # a direct Project -> Goal relationship (app/world_model/base.py),
        # not an explicit match on the goal itself.
        score += weights.RELATIONSHIP
        reasons.append(REASON_RELATIONSHIP_PROJECT)

    state = goal_state_relevance(goal.status, weights)
    if state is not None:
        s, reason = state
        score += s
        reasons.append(reason)

    p_score, p_reason = priority_relevance(goal.priority, weights)
    score += p_score
    reasons.append(p_reason)

    time_score = time_relevance(goal.target_date, current_time, weights)
    if time_score is not None:
        s, reason = time_score
        score += s
        reasons.append(reason)

    if has_active_blocker:
        score += weights.BLOCKER_PRESENT
        reasons.append(REASON_BLOCKER_PRESENT)
    if has_dependency:
        score += weights.DEPENDENCY
        reasons.append(REASON_DEPENDENCY)

    return score, reasons


def score_commitment(
    commitment: Commitment,
    *,
    explicit_hints: list[str],
    project_explicitly_matched: bool,
    current_time: str,
    weights: RelevanceWeights = DEFAULT_WEIGHTS,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    if any_hint_matches(explicit_hints, commitment.title):
        score += weights.EXPLICIT_MATCH
        reasons.append(REASON_EXPLICIT_MATCH)
    elif project_explicitly_matched:
        score += weights.RELATIONSHIP
        reasons.append(REASON_RELATIONSHIP_PROJECT)

    p_score, p_reason = priority_relevance(commitment.priority, weights)
    score += p_score
    reasons.append(p_reason)

    time_score = time_relevance(commitment.due_at, current_time, weights)
    if time_score is not None:
        s, reason = time_score
        score += s
        reasons.append(reason)

    return score, reasons


def score_event(
    event: Event,
    *,
    explicit_hints: list[str],
    project_explicitly_matched: bool,
    current_time: str,
    weights: RelevanceWeights = DEFAULT_WEIGHTS,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    if any_hint_matches(explicit_hints, event.title):
        score += weights.EXPLICIT_MATCH
        reasons.append(REASON_EXPLICIT_MATCH)
    elif project_explicitly_matched:
        score += weights.RELATIONSHIP
        reasons.append(REASON_RELATIONSHIP_PROJECT)

    time_score = time_relevance(event.starts_at, current_time, weights)
    if time_score is not None:
        s, reason = time_score
        score += s
        reasons.append(reason)

    return score, reasons


def score_blocker(
    blocker: Blocker,
    *,
    explicit_hints: list[str],
    linked_to_explicit_entity: bool,
    weights: RelevanceWeights = DEFAULT_WEIGHTS,
) -> tuple[float, list[str]]:
    score = weights.BLOCKER_PRESENT
    reasons = [REASON_BLOCKER_PRESENT]

    if any_hint_matches(explicit_hints, blocker.title):
        score += weights.EXPLICIT_MATCH
        reasons.append(REASON_EXPLICIT_MATCH)
    elif linked_to_explicit_entity:
        score += weights.RELATIONSHIP
        reasons.append(REASON_RELATIONSHIP_PROJECT)

    # Severity contributes the same way priority does elsewhere -
    # reuses the identical HIGH/MEDIUM/LOW weight tier plus a CRITICAL
    # tier one step above HIGH, via the shared PRIORITY_* constants for
    # HIGH/MEDIUM/LOW and EXPLICIT/2 headroom for CRITICAL specifically,
    # so severity never accidentally outweighs an explicit match.
    from app.world_model.base import BlockerSeverity

    if blocker.severity is BlockerSeverity.CRITICAL:
        score += weights.PRIORITY_HIGH + weights.CURRENT_STATE_ACTIVE
        reasons.append("severity_critical")
    elif blocker.severity is BlockerSeverity.HIGH:
        score += weights.PRIORITY_HIGH
        reasons.append("severity_high")
    elif blocker.severity is BlockerSeverity.MEDIUM:
        score += weights.PRIORITY_MEDIUM
        reasons.append("severity_medium")
    else:
        score += weights.PRIORITY_LOW
        reasons.append("severity_low")

    return score, reasons


def score_resource(
    resource: Resource,
    *,
    explicit_hints: list[str],
    linked_to_explicit_entity: bool,
    weights: RelevanceWeights = DEFAULT_WEIGHTS,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    if any_hint_matches(explicit_hints, resource.name):
        score += weights.EXPLICIT_MATCH
        reasons.append(REASON_EXPLICIT_MATCH)
    elif linked_to_explicit_entity:
        score += weights.RESOURCE_MATCH
        reasons.append(REASON_RESOURCE_MATCH)

    return score, reasons


def tie_break_key(
    *,
    priority: Priority | None,
    proximity_timestamp: str | None,
    updated_at: str,
    entity_id: str,
) -> tuple:
    """Deterministic tie-break ordering for candidates with an equal
    relevance score, in this exact priority: (1) higher Priority first,
    (2) closer due/start time first (missing timestamps sort last), (3)
    more recently updated first, (4) stable ascending id as the final,
    always-deterministic fallback. Returned as a tuple meant to be
    sorted ASCENDING (paired with `-score` by the caller), so each
    component is pre-negated/inverted as needed to make "better" sort
    first.
    """
    priority_rank = {Priority.HIGH: 0, Priority.MEDIUM: 1, Priority.LOW: 2}.get(
        priority, 3
    )
    # A present timestamp always sorts before a missing one; among
    # present timestamps, the closer (lexicographically smaller ISO
    # string) one sorts first.
    timestamp_key = (0, proximity_timestamp) if proximity_timestamp else (1, "")
    return (priority_rank, timestamp_key, _recency_rank(updated_at), entity_id)


def _recency_rank(updated_at: str) -> float:
    """Negated epoch seconds, so sorting this component ASCENDING puts
    the more-recently-updated timestamp first - avoids a separate
    `reverse=True` sort pass, which would also reverse the other
    tie-break components this key is combined with. Falls back to 0.0
    (sorts after every parseable timestamp) for a value that isn't a
    valid ISO 8601 timestamp, rather than raising."""
    try:
        return -datetime.fromisoformat(updated_at).timestamp()
    except ValueError:
        return 0.0
