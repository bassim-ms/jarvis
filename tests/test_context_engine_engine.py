"""
Integration tests for ContextEngine.build_snapshot
(app/context_engine/engine.py) against a real WorldModelService/
SQLiteWorldModelStore (never mocked) - the same "prove it against real
persistence" convention every other store-backed test in this codebase
uses (see e.g. tests/test_orchestration_store.py).
"""

from app.context_engine.base import (
    ContextBudget,
    ContextIntent,
    RequestContext,
    RequiredContextCategory,
    TimeScope,
    TimeScopeKind,
    default_required_context,
)
from app.context_engine.engine import ContextEngine
from app.context_engine.request_builder import build_request_context
from app.world_model.base import (
    BlockerSeverity,
    BlockerStatus,
    GoalStatus,
    LinkedEntityKind,
    Priority,
    ProjectStatus,
    ResourceType,
    TaskRefType,
)
from app.world_model.service import WorldModelService
from app.world_model.sqlite_store import SQLiteWorldModelStore

_NOW = "2026-03-04T12:00:00+00:00"


def make_engine(db_path: str = ":memory:") -> tuple[WorldModelService, ContextEngine]:
    service = WorldModelService(SQLiteWorldModelStore(db_path))
    return service, ContextEngine(service)


def all_categories_context(**overrides) -> RequestContext:
    defaults = dict(
        raw_request="test",
        required_context=frozenset(RequiredContextCategory) - {RequiredContextCategory.MEMORY},
    )
    defaults.update(overrides)
    return RequestContext(**defaults)


# -- empty World Model --------------------------------------------------


def test_empty_world_model_returns_an_empty_snapshot():
    _, engine = make_engine()
    snapshot = engine.build_snapshot(all_categories_context(), now=_NOW)

    assert snapshot.active_projects == []
    assert snapshot.relevant_goals == []
    assert snapshot.relevant_tasks == []
    assert snapshot.upcoming_commitments == []
    assert snapshot.upcoming_events == []
    assert snapshot.active_blockers == []
    assert snapshot.relevant_resources == []
    assert snapshot.current_time == _NOW


def test_snapshot_is_read_only_and_never_mutates_the_world_model():
    service, engine = make_engine()
    project = service.create_project("Robotics arm")
    before = service.get_project(project.id)

    engine.build_snapshot(all_categories_context(), now=_NOW)

    after = service.get_project(project.id)
    assert after == before


# -- required_context gating (don't retrieve unnecessarily) -----------


def test_only_requested_categories_are_populated():
    service, engine = make_engine()
    service.create_project("Robotics arm")
    context = RequestContext(
        raw_request="x", required_context=frozenset({RequiredContextCategory.PROJECTS})
    )

    snapshot = engine.build_snapshot(context, now=_NOW)

    assert len(snapshot.active_projects) == 1
    assert snapshot.relevant_goals == []
    assert "projects" in snapshot.retrieval_metadata
    assert "goals" not in snapshot.retrieval_metadata


# -- context-intent driven requests (via build_request_context) --------


def test_general_request_populates_the_general_category_set():
    service, engine = make_engine()
    service.create_project("Robotics arm")
    context = build_request_context("hello there", now=None)
    snapshot = engine.build_snapshot(context, now=_NOW)
    assert len(snapshot.active_projects) == 1


def test_project_focus_request():
    service, engine = make_engine()
    service.create_project("Robotics Arm")
    context = build_request_context("tell me about my robotics arm project")
    snapshot = engine.build_snapshot(context, now=_NOW)
    assert context.context_intent is ContextIntent.PROJECT_FOCUS
    assert len(snapshot.active_projects) == 1


def test_task_execution_request():
    service, engine = make_engine()
    project = service.create_project("Robotics arm")
    goal = service.create_goal(project.id, "Print gripper")
    service.link_task(TaskRefType.JOB, "job-1", goal_id=goal.id)
    context = build_request_context("finish the gripper print")
    snapshot = engine.build_snapshot(context, now=_NOW)
    assert context.context_intent is ContextIntent.TASK_EXECUTION
    assert RequiredContextCategory.TASKS in context.required_context


def test_planning_request():
    context = build_request_context("what should I work on next?")
    assert context.context_intent is ContextIntent.PLANNING
    assert RequiredContextCategory.PROJECTS in context.required_context
    assert RequiredContextCategory.COMMITMENTS in context.required_context


def test_scheduling_request():
    service, engine = make_engine()
    service.create_event(
        "Robotics meeting", starts_at="2026-03-04T18:00:00+00:00", ends_at="2026-03-04T19:00:00+00:00"
    )
    context = build_request_context("what's on my calendar today?")
    snapshot = engine.build_snapshot(context, now=_NOW)
    assert context.context_intent is ContextIntent.SCHEDULING
    assert len(snapshot.upcoming_events) == 1


def test_status_request():
    service, engine = make_engine()
    blocker = service.create_blocker("Printer nozzle isn't heating")
    context = build_request_context("what is blocking my project?")
    snapshot = engine.build_snapshot(context, now=_NOW)
    assert context.context_intent is ContextIntent.STATUS
    assert [b.id for b in snapshot.active_blockers] == [blocker.id]


def test_decision_support_request():
    context = build_request_context(
        "should I work on the robotics project or the physics assignment?"
    )
    assert context.context_intent is ContextIntent.DECISION_SUPPORT
    assert RequiredContextCategory.PREFERENCES in context.required_context


# -- explicit entity matches ---------------------------------------------


def test_explicit_project_match_outranks_an_unmentioned_higher_priority_project():
    service, engine = make_engine()
    mentioned = service.create_project("Robotics Arm", priority=Priority.LOW)
    unmentioned = service.create_project("Home Automation", priority=Priority.HIGH)
    context = all_categories_context(mentioned_projects=["robotics arm"])

    snapshot = engine.build_snapshot(context, now=_NOW)

    assert snapshot.active_projects[0].id == mentioned.id
    assert "explicit_match" in snapshot.retrieval_metadata["projects"].items[mentioned.id].reasons


def test_explicit_goal_match():
    service, engine = make_engine()
    project = service.create_project("Robotics arm")
    mentioned = service.create_goal(project.id, "Print gripper", priority=Priority.LOW)
    service.create_goal(project.id, "Wire motors", priority=Priority.HIGH)
    context = all_categories_context(mentioned_goals=["gripper"])

    snapshot = engine.build_snapshot(context, now=_NOW)

    assert snapshot.relevant_goals[0].id == mentioned.id
    assert (
        "explicit_match"
        in snapshot.retrieval_metadata["goals"].items[mentioned.id].reasons
    )


def test_explicit_task_match_via_label_provider():
    service = WorldModelService(SQLiteWorldModelStore(":memory:"))
    project = service.create_project("Robotics arm")
    link = service.link_task(TaskRefType.JOB, "job-1", project_id=project.id)

    labels = {link.id: "Print the gripper"}
    engine = ContextEngine(
        service, task_label_provider=lambda l: labels.get(l.id)
    )
    context = all_categories_context(mentioned_tasks=["gripper"])

    snapshot = engine.build_snapshot(context, now=_NOW)

    assert [t.id for t in snapshot.relevant_tasks] == [link.id]
    assert "explicit_match" in snapshot.retrieval_metadata["tasks"].items[link.id].reasons


def test_explicit_commitment_match():
    service, engine = make_engine()
    mentioned = service.create_commitment("Submit physics assignment", priority=Priority.LOW)
    service.create_commitment("Buy groceries", priority=Priority.HIGH)
    context = all_categories_context(mentioned_commitments=["physics"])

    snapshot = engine.build_snapshot(context, now=_NOW)

    assert snapshot.upcoming_commitments[0].id == mentioned.id


def test_ambiguous_entity_reference_preserves_all_plausible_matches():
    service, engine = make_engine()
    a = service.create_project("Robotics Arm Alpha")
    b = service.create_project("Robotics Arm Beta")
    context = all_categories_context(mentioned_projects=["robotics arm"])

    snapshot = engine.build_snapshot(context, now=_NOW)

    ids = {p.id for p in snapshot.active_projects}
    assert ids == {a.id, b.id}
    assert all(
        "explicit_match" in snapshot.retrieval_metadata["projects"].items[i].reasons
        for i in ids
    )


def test_no_matching_entity_does_not_error_and_returns_normal_candidates():
    service, engine = make_engine()
    service.create_project("Robotics arm")
    context = all_categories_context(mentioned_projects=["completely unrelated name"])

    snapshot = engine.build_snapshot(context, now=_NOW)

    assert len(snapshot.active_projects) == 1
    assert (
        "explicit_match"
        not in snapshot.retrieval_metadata["projects"].items[snapshot.active_projects[0].id].reasons
    )


# -- time relevance / overdue -------------------------------------------


def test_time_relevance_orders_commitments_by_due_date():
    service, engine = make_engine()
    far = service.create_commitment("far", due_at="2027-01-01T00:00:00+00:00")
    soon = service.create_commitment("soon", due_at="2026-03-04T13:00:00+00:00")
    context = all_categories_context()

    snapshot = engine.build_snapshot(context, now=_NOW)

    ids = [c.id for c in snapshot.upcoming_commitments]
    assert ids.index(soon.id) < ids.index(far.id)


def test_overdue_commitment_is_not_excluded_and_scores_highly():
    service, engine = make_engine()
    overdue = service.create_commitment("overdue thing", due_at="2026-01-01T00:00:00+00:00")
    far = service.create_commitment("far thing", due_at="2027-01-01T00:00:00+00:00")
    context = all_categories_context()

    snapshot = engine.build_snapshot(context, now=_NOW)

    ids = [c.id for c in snapshot.upcoming_commitments]
    assert overdue.id in ids
    assert ids.index(overdue.id) < ids.index(far.id)
    assert "time_overdue" in snapshot.retrieval_metadata["commitments"].items[overdue.id].reasons


# -- blocker / relationship relevance -------------------------------


def test_blocker_relevance_boosts_the_linked_project():
    service, engine = make_engine()
    blocked = service.create_project("Robotics arm", priority=Priority.LOW)
    unblocked = service.create_project("Home automation", priority=Priority.LOW)
    blocker = service.create_blocker("Printer nozzle isn't heating")
    service.add_blocker_link(blocker.id, LinkedEntityKind.PROJECT, blocked.id)
    context = all_categories_context()

    snapshot = engine.build_snapshot(context, now=_NOW)

    blocked_score = snapshot.retrieval_metadata["projects"].items[blocked.id].relevance_score
    unblocked_score = snapshot.retrieval_metadata["projects"].items[unblocked.id].relevance_score
    assert blocked_score > unblocked_score
    assert "has_active_blocker" in snapshot.retrieval_metadata["projects"].items[blocked.id].reasons


def test_relationship_relevance_goal_inherits_explicit_project_match():
    service, engine = make_engine()
    project = service.create_project("Robotics Arm")
    goal = service.create_goal(project.id, "Print gripper")
    context = all_categories_context(mentioned_projects=["robotics arm"])

    snapshot = engine.build_snapshot(context, now=_NOW)

    assert goal.id in {g.id for g in snapshot.relevant_goals}
    assert (
        "relationship_to_project"
        in snapshot.retrieval_metadata["goals"].items[goal.id].reasons
    )


def test_multiple_blockers_all_considered_and_sorted_by_severity():
    service, engine = make_engine()
    low = service.create_blocker("minor issue", severity=BlockerSeverity.LOW)
    critical = service.create_blocker("printer nozzle isn't heating", severity=BlockerSeverity.CRITICAL)
    context = all_categories_context()

    snapshot = engine.build_snapshot(context, now=_NOW)

    ids = [b.id for b in snapshot.active_blockers]
    assert ids.index(critical.id) < ids.index(low.id)


# -- priority relevance ---------------------------------------------------


def test_priority_relevance_orders_projects_when_otherwise_equal():
    service, engine = make_engine()
    high = service.create_project("High priority project", priority=Priority.HIGH)
    low = service.create_project("Low priority project", priority=Priority.LOW)
    context = all_categories_context()

    snapshot = engine.build_snapshot(context, now=_NOW)

    ids = [p.id for p in snapshot.active_projects]
    assert ids.index(high.id) < ids.index(low.id)


# -- hard filtering -----------------------------------------------------


def test_archived_project_normally_excluded():
    service, engine = make_engine()
    active = service.create_project("Active project")
    archived = service.create_project("Old Mars project")
    service.transition_project(archived.id, ProjectStatus.ARCHIVED)
    context = all_categories_context()

    snapshot = engine.build_snapshot(context, now=_NOW)

    ids = {p.id for p in snapshot.active_projects}
    assert active.id in ids
    assert archived.id not in ids


def test_terminal_goal_normally_excluded():
    service, engine = make_engine()
    project = service.create_project("p")
    active_goal = service.create_goal(project.id, "active goal")
    achieved_goal = service.create_goal(project.id, "achieved goal")
    service.transition_goal(achieved_goal.id, GoalStatus.ACHIEVED)
    context = all_categories_context()

    snapshot = engine.build_snapshot(context, now=_NOW)

    ids = {g.id for g in snapshot.relevant_goals}
    assert active_goal.id in ids
    assert achieved_goal.id not in ids


def test_resolved_blocker_normally_excluded():
    service, engine = make_engine()
    active = service.create_blocker("active blocker")
    resolved = service.create_blocker("resolved blocker")
    service.transition_blocker(resolved.id, BlockerStatus.RESOLVED)
    context = all_categories_context()

    snapshot = engine.build_snapshot(context, now=_NOW)

    ids = {b.id for b in snapshot.active_blockers}
    assert active.id in ids
    assert resolved.id not in ids


def test_past_event_normally_excluded():
    service, engine = make_engine()
    past = service.create_event(
        "past event", starts_at="2020-01-01T00:00:00+00:00", ends_at="2020-01-01T01:00:00+00:00"
    )
    future = service.create_event(
        "future event", starts_at="2027-01-01T00:00:00+00:00", ends_at="2027-01-01T01:00:00+00:00"
    )
    context = all_categories_context()

    snapshot = engine.build_snapshot(context, now=_NOW)

    ids = {e.id for e in snapshot.upcoming_events}
    assert future.id in ids
    assert past.id not in ids


def test_fulfilled_commitment_normally_excluded():
    service, engine = make_engine()
    open_commitment = service.create_commitment("open")
    fulfilled = service.create_commitment("fulfilled")
    from app.world_model.base import CommitmentStatus

    service.transition_commitment(fulfilled.id, CommitmentStatus.FULFILLED)
    context = all_categories_context()

    snapshot = engine.build_snapshot(context, now=_NOW)

    ids = {c.id for c in snapshot.upcoming_commitments}
    assert open_commitment.id in ids
    assert fulfilled.id not in ids


# -- explicit override of normal filtering -------------------------------


def test_explicit_request_overrides_archived_project_exclusion():
    """'Tell me about my old Mars project.' - archived, but explicitly
    named, so it must still be retrievable, and must still carry its
    real ARCHIVED status."""
    service, engine = make_engine()
    mars = service.create_project("Old Mars project")
    service.transition_project(mars.id, ProjectStatus.ARCHIVED)
    context = all_categories_context(mentioned_projects=["mars"])

    snapshot = engine.build_snapshot(context, now=_NOW)

    ids = {p.id for p in snapshot.active_projects}
    assert mars.id in ids
    returned = next(p for p in snapshot.active_projects if p.id == mars.id)
    assert returned.status is ProjectStatus.ARCHIVED  # still clearly marked
    assert "explicit_match" in snapshot.retrieval_metadata["projects"].items[mars.id].reasons


def test_explicit_request_overrides_resolved_blocker_exclusion():
    service, engine = make_engine()
    blocker = service.create_blocker("Printer nozzle isn't heating")
    service.transition_blocker(blocker.id, BlockerStatus.RESOLVED)
    context = all_categories_context(mentioned_projects=["printer nozzle isn't heating"])

    snapshot = engine.build_snapshot(context, now=_NOW)

    ids = {b.id for b in snapshot.active_blockers}
    assert blocker.id in ids
    returned = next(b for b in snapshot.active_blockers if b.id == blocker.id)
    assert returned.status is BlockerStatus.RESOLVED


def test_explicit_request_overrides_past_event_exclusion_via_overdue_time_scope():
    service, engine = make_engine()
    past = service.create_event(
        "old meeting", starts_at="2020-01-01T00:00:00+00:00", ends_at="2020-01-01T01:00:00+00:00"
    )
    context = all_categories_context(time_scope=TimeScope(kind=TimeScopeKind.OVERDUE))

    snapshot = engine.build_snapshot(context, now=_NOW)

    assert past.id in {e.id for e in snapshot.upcoming_events}


# -- budgets --------------------------------------------------------------


def test_considered_counts_candidates_before_hard_filtering_not_after():
    """`considered` must reflect every candidate that entered the
    pipeline BEFORE hard filtering - an archived project that gets
    hard-filtered OUT (not explicitly mentioned) must still count toward
    `considered`, even though it never reaches `selected`."""
    service, engine = make_engine()
    service.create_project("active one")
    service.create_project("active two")
    archived = service.create_project("archived one")
    service.transition_project(archived.id, ProjectStatus.ARCHIVED)
    context = all_categories_context()

    snapshot = engine.build_snapshot(context, now=_NOW)

    metadata = snapshot.retrieval_metadata["projects"]
    assert metadata.considered == 3  # all 3 projects, including the archived one
    assert metadata.selected == 2  # only the 2 active ones actually selected
    assert len(snapshot.active_projects) == 2


def test_considered_counts_before_filtering_for_goals_commitments_events_blockers():
    from app.world_model.base import CommitmentStatus

    service, engine = make_engine()
    project = service.create_project("p")

    active_goal = service.create_goal(project.id, "active goal")
    achieved_goal = service.create_goal(project.id, "achieved goal")
    service.transition_goal(achieved_goal.id, GoalStatus.ACHIEVED)

    open_commitment = service.create_commitment("open")
    fulfilled_commitment = service.create_commitment("fulfilled")
    service.transition_commitment(fulfilled_commitment.id, CommitmentStatus.FULFILLED)

    future_event = service.create_event(
        "future", starts_at="2027-01-01T00:00:00+00:00", ends_at="2027-01-01T01:00:00+00:00"
    )
    past_event = service.create_event(
        "past", starts_at="2020-01-01T00:00:00+00:00", ends_at="2020-01-01T01:00:00+00:00"
    )

    active_blocker = service.create_blocker("active")
    resolved_blocker = service.create_blocker("resolved")
    service.transition_blocker(resolved_blocker.id, BlockerStatus.RESOLVED)

    context = all_categories_context()
    snapshot = engine.build_snapshot(context, now=_NOW)

    assert snapshot.retrieval_metadata["goals"].considered == 2
    assert snapshot.retrieval_metadata["goals"].selected == 1
    assert snapshot.retrieval_metadata["commitments"].considered == 2
    assert snapshot.retrieval_metadata["commitments"].selected == 1
    assert snapshot.retrieval_metadata["events"].considered == 2
    assert snapshot.retrieval_metadata["events"].selected == 1
    assert snapshot.retrieval_metadata["blockers"].considered == 2
    assert snapshot.retrieval_metadata["blockers"].selected == 1


def test_context_budget_caps_the_number_of_selected_projects():
    service, engine = make_engine()
    for i in range(10):
        service.create_project(f"project {i}")
    context = all_categories_context(context_budget=ContextBudget(projects=3))

    snapshot = engine.build_snapshot(context, now=_NOW)

    assert len(snapshot.active_projects) == 3
    assert snapshot.retrieval_metadata["projects"].considered == 10
    assert snapshot.retrieval_metadata["projects"].selected == 3


def test_multiple_projects_all_considered_but_budget_limited():
    service, engine = make_engine()
    ids = [service.create_project(f"p{i}", priority=Priority.LOW).id for i in range(3)]
    context = all_categories_context(context_budget=ContextBudget(projects=2))

    snapshot = engine.build_snapshot(context, now=_NOW)

    assert len(snapshot.active_projects) == 2
    assert snapshot.retrieval_metadata["projects"].considered == 3


def test_multiple_goals_and_tasks_are_each_retrieved():
    service, engine = make_engine()
    project = service.create_project("p")
    goals = [service.create_goal(project.id, f"g{i}") for i in range(3)]
    for g in goals:
        service.link_task(TaskRefType.JOB, f"job-{g.id}", goal_id=g.id)
    context = all_categories_context()

    snapshot = engine.build_snapshot(context, now=_NOW)

    assert len(snapshot.relevant_goals) == 3
    assert len(snapshot.relevant_tasks) == 3


# -- deterministic tie-breaking -------------------------------------------


def test_deterministic_tie_break_is_stable_across_repeated_calls():
    """Equal-priority projects with no explicit/time/blocker signal tie
    on score; the engine must still return the exact same order every
    time (unit-level id-fallback behavior is covered directly by
    test_context_engine_scoring.py:test_tie_break_falls_back_to_stable_id_ordering)."""
    service, engine = make_engine()
    for i in range(5):
        service.create_project(f"project {i}", priority=Priority.MEDIUM)
    context = all_categories_context()

    first = [p.id for p in engine.build_snapshot(context, now=_NOW).active_projects]
    second = [p.id for p in engine.build_snapshot(context, now=_NOW).active_projects]

    assert first == second
    assert len(first) == 5


# -- provenance / reasons -------------------------------------------------


def test_retrieval_metadata_records_considered_and_selected_counts():
    service, engine = make_engine()
    service.create_project("a")
    service.create_project("b")
    context = all_categories_context(context_budget=ContextBudget(projects=1))

    snapshot = engine.build_snapshot(context, now=_NOW)

    metadata = snapshot.retrieval_metadata["projects"]
    assert metadata.considered == 2
    assert metadata.selected == 1
    assert len(metadata.items) == 1


def test_every_selected_item_has_a_relevance_score_and_reasons():
    service, engine = make_engine()
    project = service.create_project("Robotics arm")
    context = all_categories_context()

    snapshot = engine.build_snapshot(context, now=_NOW)

    provenance = snapshot.retrieval_metadata["projects"].items[project.id]
    assert provenance.relevance_score > 0
    assert len(provenance.reasons) > 0
    assert all(isinstance(r, str) for r in provenance.reasons)


# -- resources ----------------------------------------------------------


def test_resources_linked_to_explicit_project_are_boosted():
    service, engine = make_engine()
    project = service.create_project("Robotics Arm")
    linked = service.create_resource("Soldering iron", type=ResourceType.DEVICE)
    unlinked = service.create_resource("Random cable", type=ResourceType.DEVICE)
    service.add_resource_link(linked.id, LinkedEntityKind.PROJECT, project.id)
    context = all_categories_context(mentioned_projects=["robotics arm"])

    snapshot = engine.build_snapshot(context, now=_NOW)

    linked_score = snapshot.retrieval_metadata["resources"].items[linked.id].relevance_score
    unlinked_score = snapshot.retrieval_metadata["resources"].items[unlinked.id].relevance_score
    assert linked_score > unlinked_score


# -- context_window -------------------------------------------------------


def test_context_window_reflects_the_request_time_scope():
    service, engine = make_engine()
    context = all_categories_context(
        time_scope=TimeScope(
            kind=TimeScopeKind.TODAY,
            start="2026-03-04T00:00:00+00:00",
            end="2026-03-05T00:00:00+00:00",
        )
    )

    snapshot = engine.build_snapshot(context, now=_NOW)

    assert snapshot.context_window.start == "2026-03-04T00:00:00+00:00"
    assert snapshot.context_window.end == "2026-03-05T00:00:00+00:00"


# -- persistence across restart -------------------------------------------


def test_snapshot_is_consistent_across_a_world_model_restart(tmp_path):
    db_path = str(tmp_path / "world_model.db")
    service = WorldModelService(SQLiteWorldModelStore(db_path))
    engine = ContextEngine(service)
    project = service.create_project("Robotics arm")
    context = all_categories_context()
    before = engine.build_snapshot(context, now=_NOW)

    restarted_service = WorldModelService(SQLiteWorldModelStore(db_path))
    restarted_engine = ContextEngine(restarted_service)
    after = restarted_engine.build_snapshot(context, now=_NOW)

    assert [p.id for p in before.active_projects] == [p.id for p in after.active_projects]
    assert [p.id for p in after.active_projects] == [project.id]


# -- memory (separate, optional retrieval source) --------------------------


def test_memory_not_populated_without_a_memory_store_injected():
    service, engine = make_engine()
    context = RequestContext(
        raw_request="x", required_context=frozenset({RequiredContextCategory.MEMORY})
    )
    snapshot = engine.build_snapshot(context, now=_NOW)
    assert snapshot.relevant_memory == []


def test_memory_populated_when_requested_and_a_memory_store_is_injected():
    from app.memory.sqlite_store import SQLiteMemoryStore

    service = WorldModelService(SQLiteWorldModelStore(":memory:"))
    memory = SQLiteMemoryStore(":memory:")
    memory.save_fact("response_style", "concise")
    engine = ContextEngine(service, memory=memory)
    context = RequestContext(
        raw_request="x", required_context=frozenset({RequiredContextCategory.MEMORY})
    )

    snapshot = engine.build_snapshot(context, now=_NOW)

    assert snapshot.relevant_memory == ["response_style: concise"]
