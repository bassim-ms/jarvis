"""
Tests for WorldModelService (app/world_model/service.py) - the
deterministic operations layer: creation, updates, validated
transitions, relationship management (including cross-entity foreign-key
validation and goal-dependency cycle prevention), and the structured
query methods a future Context Engine will use.
"""

import pytest

from app.world_model.base import (
    BlockerSeverity,
    BlockerStatus,
    CommitmentStatus,
    EntitySource,
    GoalStatus,
    InvalidConfidenceError,
    InvalidTransitionError,
    LinkedEntityKind,
    Priority,
    ProjectStatus,
    ResourceStatus,
    ResourceType,
    TaskRefType,
)
from app.world_model.service import (
    GoalDependencyCycleError,
    InvalidLinkError,
    UnknownEntityError,
    WorldModelService,
)
from app.world_model.sqlite_store import SQLiteWorldModelStore


def make_service() -> WorldModelService:
    return WorldModelService(SQLiteWorldModelStore(":memory:"))


# -- Project ----------------------------------------------------------------


def test_create_project_defaults_to_active():
    service = make_service()
    project = service.create_project("Robotics arm", priority=Priority.HIGH)

    assert project.status == ProjectStatus.ACTIVE
    assert project.priority == Priority.HIGH
    assert service.get_project(project.id) == project


def test_update_project_only_changes_given_fields():
    service = make_service()
    project = service.create_project("Robotics arm", description="v1")

    updated = service.update_project(project.id, description="v2")

    assert updated.name == "Robotics arm"
    assert updated.description == "v2"


def test_update_project_unknown_id_raises():
    service = make_service()
    with pytest.raises(UnknownEntityError):
        service.update_project("nope", name="x")


def test_transition_project_valid():
    service = make_service()
    project = service.create_project("p")

    paused = service.transition_project(project.id, ProjectStatus.PAUSED)
    assert paused.status == ProjectStatus.PAUSED

    active = service.transition_project(project.id, ProjectStatus.ACTIVE)
    assert active.status == ProjectStatus.ACTIVE


def test_transition_project_invalid_raises_and_does_not_change_status():
    service = make_service()
    project = service.create_project("p")
    service.transition_project(project.id, ProjectStatus.COMPLETED)

    with pytest.raises(InvalidTransitionError):
        service.transition_project(project.id, ProjectStatus.ACTIVE)

    assert service.get_project(project.id).status == ProjectStatus.COMPLETED


def test_get_active_projects_excludes_paused():
    service = make_service()
    active = service.create_project("active")
    paused = service.create_project("paused")
    service.transition_project(paused.id, ProjectStatus.PAUSED)

    result = service.get_active_projects()

    assert [p.id for p in result] == [active.id]


# -- Goal ---------------------------------------------------------------


def test_create_goal_requires_existing_project():
    service = make_service()
    with pytest.raises(UnknownEntityError):
        service.create_goal("no-such-project", "title")


def test_create_goal_defaults_to_active_and_links_to_project():
    service = make_service()
    project = service.create_project("p")

    goal = service.create_goal(project.id, "Print the gripper")

    assert goal.status == GoalStatus.ACTIVE
    assert goal.project_id == project.id


def test_transition_goal_valid_and_invalid():
    service = make_service()
    project = service.create_project("p")
    goal = service.create_goal(project.id, "g")

    achieved = service.transition_goal(goal.id, GoalStatus.ACHIEVED)
    assert achieved.status == GoalStatus.ACHIEVED

    with pytest.raises(InvalidTransitionError):
        service.transition_goal(goal.id, GoalStatus.ACTIVE)


def test_get_active_goals_filters_by_project_and_status():
    service = make_service()
    project_a = service.create_project("a")
    project_b = service.create_project("b")
    goal_a = service.create_goal(project_a.id, "goal a")
    service.create_goal(project_b.id, "goal b")
    paused_a = service.create_goal(project_a.id, "paused a")
    service.transition_goal(paused_a.id, GoalStatus.PAUSED)

    result = service.get_active_goals(project_id=project_a.id)

    assert [g.id for g in result] == [goal_a.id]


# -- Task/Job linkage -------------------------------------------------------


def test_link_task_requires_project_or_goal():
    service = make_service()
    with pytest.raises(InvalidLinkError):
        service.link_task(TaskRefType.JOB, "job-1")


def test_link_task_to_project_and_goal():
    service = make_service()
    project = service.create_project("p")
    goal = service.create_goal(project.id, "g")

    link = service.link_task(TaskRefType.JOB, "job-1", project_id=project.id, goal_id=goal.id)

    assert link.ref_type == TaskRefType.JOB
    assert link.ref_id == "job-1"
    assert [l.id for l in service.get_tasks_for_goal(goal.id)] == [link.id]
    assert [l.id for l in service.get_tasks_for_project(project.id)] == [link.id]


def test_link_task_with_unknown_project_raises():
    service = make_service()
    with pytest.raises(UnknownEntityError):
        service.link_task(TaskRefType.TASK, "task-1", project_id="nope")


def test_link_task_does_not_validate_the_external_ref_id():
    """M15/M9 remain the sole authority for whether a Task/Job id
    actually exists - the World Model stores it opaquely."""
    service = make_service()
    project = service.create_project("p")

    link = service.link_task(TaskRefType.JOB, "totally-made-up-job-id", project_id=project.id)

    assert link.ref_id == "totally-made-up-job-id"


def test_unlink_task():
    service = make_service()
    project = service.create_project("p")
    link = service.link_task(TaskRefType.JOB, "job-1", project_id=project.id)

    assert service.unlink_task(link.id) is True
    assert service.get_tasks_for_project(project.id) == []
    assert service.unlink_task(link.id) is False


# -- Commitment -----------------------------------------------------------


def test_create_commitment_defaults_to_open():
    service = make_service()
    commitment = service.create_commitment("Submit assignment", due_at="2026-03-01T00:00:00+00:00")

    assert commitment.status == CommitmentStatus.OPEN
    assert commitment.source == EntitySource.USER


def test_create_commitment_with_unknown_project_raises():
    service = make_service()
    with pytest.raises(UnknownEntityError):
        service.create_commitment("x", project_id="nope")


def test_transition_commitment_valid_and_invalid():
    service = make_service()
    commitment = service.create_commitment("Submit assignment")

    fulfilled = service.transition_commitment(commitment.id, CommitmentStatus.FULFILLED)
    assert fulfilled.status == CommitmentStatus.FULFILLED

    with pytest.raises(InvalidTransitionError):
        service.transition_commitment(commitment.id, CommitmentStatus.OPEN)


def test_get_open_commitments():
    service = make_service()
    open_one = service.create_commitment("open")
    missed = service.create_commitment("missed")
    service.transition_commitment(missed.id, CommitmentStatus.MISSED)

    result = service.get_open_commitments()

    assert [c.id for c in result] == [open_one.id]


# -- Event ------------------------------------------------------------------


def test_create_and_update_event():
    service = make_service()
    event = service.create_event(
        "Robotics meeting", starts_at="2026-02-01T10:00:00+00:00", ends_at="2026-02-01T11:00:00+00:00"
    )

    updated = service.update_event(event.id, location="Lab B")

    assert updated.location == "Lab B"
    assert updated.title == "Robotics meeting"


def test_get_upcoming_events_filters_by_after():
    service = make_service()
    past = service.create_event("past", starts_at="2020-01-01T00:00:00+00:00", ends_at="2020-01-01T01:00:00+00:00")
    future = service.create_event("future", starts_at="2030-01-01T00:00:00+00:00", ends_at="2030-01-01T01:00:00+00:00")

    result = service.get_upcoming_events(after="2025-01-01T00:00:00+00:00")

    assert [e.id for e in result] == [future.id]
    assert past.id not in [e.id for e in result]


# -- Resource -------------------------------------------------------------


def test_create_and_update_resource():
    service = make_service()
    resource = service.create_resource("Arduino UNO Q", type=ResourceType.HARDWARE)

    updated = service.update_resource(resource.id, status=ResourceStatus.IN_USE)

    assert updated.status == ResourceStatus.IN_USE


def test_add_resource_link_and_get_resources_for_entity():
    service = make_service()
    resource = service.create_resource("3D printer", type=ResourceType.DEVICE)

    service.add_resource_link(resource.id, LinkedEntityKind.TASK, "task-1")
    result = service.get_resources_for_entity(LinkedEntityKind.TASK, "task-1")

    assert [r.id for r in result] == [resource.id]


def test_add_resource_link_rejects_goal_kind():
    service = make_service()
    resource = service.create_resource("printer", type=ResourceType.DEVICE)

    with pytest.raises(InvalidLinkError):
        service.add_resource_link(resource.id, LinkedEntityKind.GOAL, "goal-1")


def test_add_resource_link_unknown_resource_raises():
    service = make_service()
    with pytest.raises(UnknownEntityError):
        service.add_resource_link("nope", LinkedEntityKind.TASK, "task-1")


def test_add_resource_link_validates_project_existence():
    service = make_service()
    resource = service.create_resource("printer", type=ResourceType.DEVICE)
    with pytest.raises(UnknownEntityError):
        service.add_resource_link(resource.id, LinkedEntityKind.PROJECT, "no-such-project")


# -- Blocker ----------------------------------------------------------------


def test_create_blocker_defaults_to_active():
    service = make_service()
    blocker = service.create_blocker("Printer nozzle isn't heating", severity=BlockerSeverity.HIGH)

    assert blocker.status == BlockerStatus.ACTIVE
    assert blocker.resolved_at is None


def test_transition_blocker_to_resolved_sets_resolved_at():
    service = make_service()
    blocker = service.create_blocker("blocked")

    resolved = service.transition_blocker(blocker.id, BlockerStatus.RESOLVED)

    assert resolved.status == BlockerStatus.RESOLVED
    assert resolved.resolved_at is not None


def test_transition_blocker_to_dismissed_leaves_resolved_at_none():
    service = make_service()
    blocker = service.create_blocker("blocked")

    dismissed = service.transition_blocker(blocker.id, BlockerStatus.DISMISSED)

    assert dismissed.status == BlockerStatus.DISMISSED
    assert dismissed.resolved_at is None


def test_transition_blocker_invalid_raises():
    service = make_service()
    blocker = service.create_blocker("blocked")
    service.transition_blocker(blocker.id, BlockerStatus.RESOLVED)

    with pytest.raises(InvalidTransitionError):
        service.transition_blocker(blocker.id, BlockerStatus.ACTIVE)


def test_get_active_blockers():
    service = make_service()
    active = service.create_blocker("active blocker")
    resolved = service.create_blocker("resolved blocker")
    service.transition_blocker(resolved.id, BlockerStatus.RESOLVED)

    result = service.get_active_blockers()

    assert [b.id for b in result] == [active.id]


# -- Blocker relationships --------------------------------------------------


def test_add_blocker_link_to_project_goal_task_and_job():
    service = make_service()
    project = service.create_project("p")
    goal = service.create_goal(project.id, "g")
    blocker = service.create_blocker("Printer nozzle isn't heating")

    service.add_blocker_link(blocker.id, LinkedEntityKind.PROJECT, project.id)
    service.add_blocker_link(blocker.id, LinkedEntityKind.GOAL, goal.id)
    service.add_blocker_link(blocker.id, LinkedEntityKind.TASK, "task-1")
    service.add_blocker_link(blocker.id, LinkedEntityKind.JOB, "job-1")

    affected = service.get_entities_affected_by_blocker(blocker.id)
    assert len(affected) == 4
    kinds = {link.entity_kind for link in affected}
    assert kinds == {
        LinkedEntityKind.PROJECT,
        LinkedEntityKind.GOAL,
        LinkedEntityKind.TASK,
        LinkedEntityKind.JOB,
    }


def test_add_blocker_link_unknown_blocker_raises():
    service = make_service()
    with pytest.raises(UnknownEntityError):
        service.add_blocker_link("nope", LinkedEntityKind.TASK, "task-1")


def test_add_blocker_link_unknown_project_raises():
    service = make_service()
    blocker = service.create_blocker("blocked")
    with pytest.raises(UnknownEntityError):
        service.add_blocker_link(blocker.id, LinkedEntityKind.PROJECT, "no-such-project")


def test_get_blockers_for_entity():
    service = make_service()
    blocker = service.create_blocker("Printer nozzle isn't heating")
    service.add_blocker_link(blocker.id, LinkedEntityKind.TASK, "task-1")

    result = service.get_blockers_for_entity(LinkedEntityKind.TASK, "task-1")

    assert [b.id for b in result] == [blocker.id]


def test_remove_blocker_link():
    service = make_service()
    blocker = service.create_blocker("blocked")
    link = service.add_blocker_link(blocker.id, LinkedEntityKind.TASK, "task-1")

    assert service.remove_blocker_link(link.id) is True
    assert service.get_entities_affected_by_blocker(blocker.id) == []


# -- Preference -------------------------------------------------------------


def test_create_preference_defaults_to_full_confidence():
    service = make_service()
    preference = service.create_preference("response_style", "concise")

    assert preference.confidence == 1.0
    assert preference.source == EntitySource.USER


def test_create_preference_rejects_out_of_range_confidence():
    service = make_service()
    with pytest.raises(InvalidConfidenceError):
        service.create_preference("response_style", "concise", confidence=1.5)


def test_create_preference_records_inference_source_distinctly():
    service = make_service()
    preference = service.create_preference(
        "likely_timezone", "UTC+4", source=EntitySource.INFERENCE, confidence=0.4
    )

    assert preference.source == EntitySource.INFERENCE
    assert preference.confidence == 0.4
    # Nothing in this module upgrades it on its own.
    fetched = service.get_preference(preference.id)
    assert fetched.source == EntitySource.INFERENCE


def test_update_preference_rejects_out_of_range_confidence():
    service = make_service()
    preference = service.create_preference("k", "v")
    with pytest.raises(InvalidConfidenceError):
        service.update_preference(preference.id, confidence=-0.1)


def test_get_preferences_filters_by_key_and_category():
    service = make_service()
    style = service.create_preference("response_style", "concise", category="communication")
    service.create_preference("unit_system", "metric", category="units")

    result = service.get_preferences(key="response_style")

    assert [p.id for p in result] == [style.id]


# -- Goal dependencies --------------------------------------------------


def test_add_goal_dependency_valid():
    service = make_service()
    project = service.create_project("p")
    a = service.create_goal(project.id, "a")
    b = service.create_goal(project.id, "b")

    dependency = service.add_goal_dependency(a.id, b.id)

    assert [d.id for d in service.get_goal_dependencies(a.id)] == [dependency.id]
    assert [d.id for d in service.get_goal_dependents(b.id)] == [dependency.id]


def test_add_goal_dependency_rejects_self_dependency():
    service = make_service()
    project = service.create_project("p")
    a = service.create_goal(project.id, "a")

    with pytest.raises(GoalDependencyCycleError):
        service.add_goal_dependency(a.id, a.id)


def test_add_goal_dependency_rejects_direct_cycle():
    service = make_service()
    project = service.create_project("p")
    a = service.create_goal(project.id, "a")
    b = service.create_goal(project.id, "b")
    service.add_goal_dependency(a.id, b.id)  # a depends on b

    with pytest.raises(GoalDependencyCycleError):
        service.add_goal_dependency(b.id, a.id)  # would close a<->b cycle


def test_add_goal_dependency_rejects_transitive_cycle():
    service = make_service()
    project = service.create_project("p")
    a = service.create_goal(project.id, "a")
    b = service.create_goal(project.id, "b")
    c = service.create_goal(project.id, "c")
    service.add_goal_dependency(a.id, b.id)  # a depends on b
    service.add_goal_dependency(b.id, c.id)  # b depends on c

    with pytest.raises(GoalDependencyCycleError):
        service.add_goal_dependency(c.id, a.id)  # would close a->b->c->a


def test_add_goal_dependency_unknown_goal_raises():
    service = make_service()
    project = service.create_project("p")
    a = service.create_goal(project.id, "a")

    with pytest.raises(UnknownEntityError):
        service.add_goal_dependency(a.id, "no-such-goal")


def test_remove_goal_dependency():
    service = make_service()
    project = service.create_project("p")
    a = service.create_goal(project.id, "a")
    b = service.create_goal(project.id, "b")
    dependency = service.add_goal_dependency(a.id, b.id)

    assert service.remove_goal_dependency(dependency.id) is True
    assert service.get_goal_dependencies(a.id) == []


# -- get_project_context ------------------------------------------------


def test_get_project_context_bundles_everything():
    service = make_service()
    project = service.create_project("Robotics arm")
    goal = service.create_goal(project.id, "Print the gripper")
    commitment = service.create_commitment("obligation", project_id=project.id)
    event = service.create_event(
        "meeting",
        starts_at="2026-02-01T00:00:00+00:00",
        ends_at="2026-02-01T01:00:00+00:00",
        project_id=project.id,
    )
    task_link = service.link_task(TaskRefType.JOB, "job-1", project_id=project.id)
    blocker = service.create_blocker("blocked")
    service.add_blocker_link(blocker.id, LinkedEntityKind.PROJECT, project.id)
    resource = service.create_resource("printer", type=ResourceType.DEVICE)
    service.add_resource_link(resource.id, LinkedEntityKind.PROJECT, project.id)

    context = service.get_project_context(project.id)

    assert context.project.id == project.id
    assert [g.id for g in context.goals] == [goal.id]
    assert [c.id for c in context.commitments] == [commitment.id]
    assert [e.id for e in context.events] == [event.id]
    assert [l.id for l in context.task_links] == [task_link.id]
    assert len(context.blocker_links) == 1
    assert len(context.resource_links) == 1


def test_get_project_context_unknown_project_raises():
    service = make_service()
    with pytest.raises(UnknownEntityError):
        service.get_project_context("nope")
