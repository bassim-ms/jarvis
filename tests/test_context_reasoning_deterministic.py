"""
Tests for app/context_reasoning/deterministic.py: analyze() derives
bounded, explainable facts/conflicts from a ContextSnapshot/RequestContext
alone - no AI call, no World Model query of its own.
"""

from app.context_engine.base import (
    CategoryRetrieval,
    ContextWindow,
    ItemProvenance,
    RequestContext,
    TimeScope,
    TimeScopeKind,
)
from app.context_reasoning.deterministic import analyze
from app.world_model.base import (
    Blocker,
    BlockerSeverity,
    BlockerStatus,
    Commitment,
    CommitmentStatus,
    ContextSnapshot,
    EntitySource,
    Event,
    Goal,
    GoalStatus,
    Priority,
    Project,
    ProjectStatus,
)

_NOW = "2026-01-01T12:00:00+00:00"


def _rc(message: str = "hello", **overrides) -> RequestContext:
    return RequestContext(raw_request=message, **overrides)


def _project(id_="project-1", priority=Priority.MEDIUM) -> Project:
    return Project(
        id=id_,
        name="Robotics Arm",
        description="",
        status=ProjectStatus.ACTIVE,
        priority=priority,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _goal(id_="goal-1", project_id="project-1", target_date=None, priority=Priority.MEDIUM) -> Goal:
    return Goal(
        id=id_,
        project_id=project_id,
        title="Print gripper",
        description="",
        status=GoalStatus.ACTIVE,
        priority=priority,
        target_date=target_date,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _commitment(id_="commitment-1", due_at=None, status=CommitmentStatus.OPEN, priority=Priority.MEDIUM) -> Commitment:
    return Commitment(
        id=id_,
        title="Submit assignment",
        description="",
        status=status,
        priority=priority,
        due_at=due_at,
        source=EntitySource.USER,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _event(id_="event-1", starts_at=_NOW, ends_at=_NOW, title="Robotics meeting") -> Event:
    return Event(
        id=id_,
        title=title,
        description="",
        starts_at=starts_at,
        ends_at=ends_at,
        location="",
        source=EntitySource.USER,
        external_id=None,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _blocker(id_="blocker-1", severity=BlockerSeverity.MEDIUM) -> Blocker:
    return Blocker(
        id=id_,
        title="Printer nozzle isn't heating",
        description="",
        status=BlockerStatus.ACTIVE,
        severity=severity,
        detected_at=_NOW,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _snapshot(**overrides) -> ContextSnapshot:
    defaults = dict(current_time=_NOW)
    defaults.update(overrides)
    return ContextSnapshot(**defaults)


# -- empty context -------------------------------------------------------


def test_empty_snapshot_produces_no_facts_or_conflicts():
    analysis = analyze(_snapshot(), _rc())
    assert analysis.is_empty()


# -- overdue items ---------------------------------------------------------


def test_overdue_open_commitment_is_a_fact():
    commitment = _commitment(due_at="2025-12-31T00:00:00+00:00")
    analysis = analyze(_snapshot(upcoming_commitments=[commitment]), _rc())
    texts = [f.text for f in analysis.facts]
    assert any("overdue" in t for t in texts)
    fact = next(f for f in analysis.facts if "overdue" in f.text)
    assert fact.provenance == (f"commitment:{commitment.id}",)


def test_overdue_fulfilled_commitment_is_not_flagged():
    commitment = _commitment(
        due_at="2025-12-31T00:00:00+00:00", status=CommitmentStatus.FULFILLED
    )
    analysis = analyze(_snapshot(upcoming_commitments=[commitment]), _rc())
    assert not any("overdue" in f.text for f in analysis.facts)


def test_future_commitment_is_not_overdue():
    commitment = _commitment(due_at="2099-01-01T00:00:00+00:00")
    analysis = analyze(_snapshot(upcoming_commitments=[commitment]), _rc())
    assert not any("overdue" in f.text for f in analysis.facts)


# -- approaching deadlines -------------------------------------------------


def test_goal_target_date_within_48h_is_approaching():
    goal = _goal(target_date="2026-01-02T00:00:00+00:00")
    analysis = analyze(_snapshot(relevant_goals=[goal]), _rc())
    assert any("approaching target date" in f.text for f in analysis.facts)


def test_goal_target_date_far_future_is_not_approaching():
    goal = _goal(target_date="2099-01-01T00:00:00+00:00")
    analysis = analyze(_snapshot(relevant_goals=[goal]), _rc())
    assert not any("approaching target date" in f.text for f in analysis.facts)


def test_commitment_due_soon_is_approaching():
    commitment = _commitment(due_at="2026-01-02T00:00:00+00:00")
    analysis = analyze(_snapshot(upcoming_commitments=[commitment]), _rc())
    assert any("due soon" in f.text for f in analysis.facts)


def test_event_starting_soon_is_approaching():
    event = _event(starts_at="2026-01-02T00:00:00+00:00", ends_at="2026-01-02T01:00:00+00:00")
    analysis = analyze(_snapshot(upcoming_events=[event]), _rc())
    assert any("starts soon" in f.text for f in analysis.facts)


# -- active blockers --------------------------------------------------------


def test_active_blocker_is_a_fact_with_severity():
    blocker = _blocker(severity=BlockerSeverity.CRITICAL)
    analysis = analyze(_snapshot(active_blockers=[blocker]), _rc())
    fact = next(f for f in analysis.facts if "Printer nozzle" in f.text)
    assert "critical" in fact.text
    assert fact.provenance == (f"blocker:{blocker.id}",)


# -- relationships -----------------------------------------------------------


def test_goal_belonging_to_active_project_is_a_relationship_fact():
    project = _project()
    goal = _goal(project_id=project.id)
    analysis = analyze(_snapshot(active_projects=[project], relevant_goals=[goal]), _rc())
    fact = next(f for f in analysis.facts if "belongs to project" in f.text)
    assert f"project:{project.id}" in fact.provenance
    assert f"goal:{goal.id}" in fact.provenance


def test_goal_whose_project_is_not_selected_has_no_relationship_fact():
    goal = _goal(project_id="project-not-in-snapshot")
    analysis = analyze(_snapshot(relevant_goals=[goal]), _rc())
    assert not any("belongs to project" in f.text for f in analysis.facts)


# -- explicit priorities ------------------------------------------------------


def test_high_priority_project_is_flagged():
    project = _project(priority=Priority.HIGH)
    analysis = analyze(_snapshot(active_projects=[project]), _rc())
    assert any("high priority" in f.text and project.name in f.text for f in analysis.facts)


def test_medium_priority_project_is_not_flagged():
    project = _project(priority=Priority.MEDIUM)
    analysis = analyze(_snapshot(active_projects=[project]), _rc())
    assert not any("high priority" in f.text for f in analysis.facts)


# -- time constraints (RequestContext.time_scope) -----------------------------


def test_time_scope_produces_a_request_grounded_fact():
    request_context = _rc(time_scope=TimeScope(kind=TimeScopeKind.TODAY, start=_NOW, end=_NOW))
    analysis = analyze(_snapshot(), request_context)
    fact = next(f for f in analysis.facts if "time window" in f.text)
    assert fact.provenance == ("request",)


def test_no_time_scope_produces_no_time_window_fact():
    analysis = analyze(_snapshot(), _rc())
    assert not any("time window" in f.text for f in analysis.facts)


# -- dependency relationships (via ContextSnapshot.retrieval_metadata) -------


def test_goal_with_dependency_reason_in_retrieval_metadata_is_flagged():
    goal = _goal()
    metadata = {
        "goals": CategoryRetrieval(
            considered=1,
            selected=1,
            items={goal.id: ItemProvenance(relevance_score=6.0, reasons=("dependency",))},
        )
    }
    analysis = analyze(
        _snapshot(relevant_goals=[goal], retrieval_metadata=metadata), _rc()
    )
    assert any("dependency relationship" in f.text for f in analysis.facts)


def test_goal_without_dependency_reason_is_not_flagged():
    goal = _goal()
    metadata = {
        "goals": CategoryRetrieval(
            considered=1,
            selected=1,
            items={goal.id: ItemProvenance(relevance_score=6.0, reasons=("active_state",))},
        )
    }
    analysis = analyze(
        _snapshot(relevant_goals=[goal], retrieval_metadata=metadata), _rc()
    )
    assert not any("dependency relationship" in f.text for f in analysis.facts)


# -- event/time conflicts -----------------------------------------------------


def test_overlapping_events_are_a_conflict():
    event_a = _event(id_="event-a", starts_at="2026-01-01T10:00:00+00:00", ends_at="2026-01-01T11:00:00+00:00", title="Meeting A")
    event_b = _event(id_="event-b", starts_at="2026-01-01T10:30:00+00:00", ends_at="2026-01-01T11:30:00+00:00", title="Meeting B")
    analysis = analyze(_snapshot(upcoming_events=[event_a, event_b]), _rc())
    assert len(analysis.conflicts) == 1
    conflict = analysis.conflicts[0]
    assert "overlaps with" in conflict.text
    assert f"event:{event_a.id}" in conflict.provenance
    assert f"event:{event_b.id}" in conflict.provenance


def test_non_overlapping_events_are_not_a_conflict():
    event_a = _event(id_="event-a", starts_at="2026-01-01T10:00:00+00:00", ends_at="2026-01-01T11:00:00+00:00")
    event_b = _event(id_="event-b", starts_at="2026-01-01T12:00:00+00:00", ends_at="2026-01-01T13:00:00+00:00")
    analysis = analyze(_snapshot(upcoming_events=[event_a, event_b]), _rc())
    assert analysis.conflicts == ()


def test_commitment_due_during_event_is_a_conflict():
    event = _event(starts_at="2026-01-01T10:00:00+00:00", ends_at="2026-01-01T11:00:00+00:00")
    commitment = _commitment(due_at="2026-01-01T10:30:00+00:00")
    analysis = analyze(
        _snapshot(upcoming_events=[event], upcoming_commitments=[commitment]), _rc()
    )
    conflict = next(c for c in analysis.conflicts if "due during event" in c.text)
    assert f"commitment:{commitment.id}" in conflict.provenance
    assert f"event:{event.id}" in conflict.provenance


def test_commitment_not_overlapping_event_is_not_a_conflict():
    event = _event(starts_at="2026-01-01T10:00:00+00:00", ends_at="2026-01-01T11:00:00+00:00")
    commitment = _commitment(due_at="2026-01-01T15:00:00+00:00")
    analysis = analyze(
        _snapshot(upcoming_events=[event], upcoming_commitments=[commitment]), _rc()
    )
    assert not any("due during event" in c.text for c in analysis.conflicts)


# -- determinism ---------------------------------------------------------


def test_analyze_is_deterministic_for_the_same_inputs():
    project = _project()
    goal = _goal(project_id=project.id, target_date="2026-01-02T00:00:00+00:00")
    snapshot = _snapshot(active_projects=[project], relevant_goals=[goal])
    request_context = _rc()

    first = analyze(snapshot, request_context)
    second = analyze(snapshot, request_context)
    assert first == second


# --- regression: date-only / naive timestamps made analyze() raise, failing every chat request (found 2026-11-18) -------------


def test_a_date_only_goal_target_date_is_compared_without_failing_and_counts_as_that_whole_day():
    """World Model goals carry target dates like "2026-01-02" (no time, no zone); comparing them with the snapshot's aware time
    raised "can't compare offset-naive and offset-aware datetimes"."""
    goal = _goal(target_date="2026-01-02")

    analysis = analyze(_snapshot(relevant_goals=[goal]), _rc())

    assert any("approaching target date" in fact.text for fact in analysis.facts)


def test_naive_commitment_and_event_times_are_read_as_utc_without_failing():
    overdue = _commitment(id_="c-old", due_at="2025-12-31T00:00:00")
    soon = _commitment(id_="c-soon", due_at="2026-01-01T12:00:00")

    analysis = analyze(_snapshot(upcoming_commitments=[overdue, soon]), _rc())

    texts = " ".join(fact.text for fact in analysis.facts)
    assert "is overdue" in texts and "is due soon" in texts
