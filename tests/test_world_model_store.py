"""
Tests for SQLiteWorldModelStore (app/world_model/sqlite_store.py):
create/get/list/save round-trip every entity and relationship through
real SQLite (never mocked) - the same convention every other store's own
tests use (see e.g. tests/test_orchestration_store.py) - plus a
restart-survival test using a real on-disk file (tmp_path), since an
in-memory (":memory:") database can't prove persistence across process
restarts on its own.
"""

from app.world_model.base import (
    Blocker,
    BlockerLink,
    BlockerSeverity,
    BlockerStatus,
    Commitment,
    CommitmentStatus,
    Event,
    EntitySource,
    Goal,
    GoalDependency,
    GoalStatus,
    LinkedEntityKind,
    Preference,
    Priority,
    Project,
    ProjectStatus,
    Resource,
    ResourceLink,
    ResourceStatus,
    ResourceType,
    TaskLink,
    TaskRefType,
)
from app.world_model.sqlite_store import SQLiteWorldModelStore, new_id


def make_store() -> SQLiteWorldModelStore:
    return SQLiteWorldModelStore(":memory:")


def make_project(store: SQLiteWorldModelStore, **overrides) -> Project:
    project = Project(
        id=new_id(),
        name="Robotics arm",
        description="Build a robotic arm",
        status=ProjectStatus.ACTIVE,
        priority=Priority.HIGH,
        metadata={"tag": "hardware"},
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    for key, value in overrides.items():
        setattr(project, key, value)
    store.create_project(project)
    return project


# -- Project ----------------------------------------------------------------


def test_create_and_get_project_round_trips():
    store = make_store()
    project = make_project(store)

    fetched = store.get_project(project.id)

    assert fetched == project


def test_get_project_returns_none_for_unknown_id():
    store = make_store()
    assert store.get_project("nope") is None


def test_list_projects_orders_most_recent_first_and_filters_by_status():
    store = make_store()
    first = make_project(store, name="first", created_at="2026-01-01T00:00:00+00:00")
    second = make_project(store, name="second", created_at="2026-01-02T00:00:00+00:00")
    second.status = ProjectStatus.PAUSED
    store.save_project(second)

    all_projects = store.list_projects()
    assert [p.id for p in all_projects] == [second.id, first.id]

    active_only = store.list_projects(status=ProjectStatus.ACTIVE)
    assert [p.id for p in active_only] == [first.id]


def test_save_project_persists_status_and_metadata_changes():
    store = make_store()
    project = make_project(store)

    project.status = ProjectStatus.PAUSED
    project.metadata = {"tag": "paused-for-parts"}
    project.updated_at = "2026-02-01T00:00:00+00:00"
    store.save_project(project)

    fetched = store.get_project(project.id)
    assert fetched.status == ProjectStatus.PAUSED
    assert fetched.metadata == {"tag": "paused-for-parts"}


# -- Goal ---------------------------------------------------------------


def test_create_get_list_and_save_goal():
    store = make_store()
    project = make_project(store)
    goal = Goal(
        id=new_id(),
        project_id=project.id,
        title="Print the gripper",
        description="",
        status=GoalStatus.ACTIVE,
        priority=Priority.MEDIUM,
        target_date=None,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    store.create_goal(goal)

    assert store.get_goal(goal.id) == goal
    assert [g.id for g in store.list_goals(project_id=project.id)] == [goal.id]
    assert store.list_goals(project_id="other-project") == []

    goal.status = GoalStatus.ACHIEVED
    goal.updated_at = "2026-01-02T00:00:00+00:00"
    store.save_goal(goal)
    assert store.get_goal(goal.id).status == GoalStatus.ACHIEVED
    assert [g.id for g in store.list_goals(status=GoalStatus.ACHIEVED)] == [goal.id]


# -- Commitment -------------------------------------------------------------


def test_create_get_list_and_save_commitment():
    store = make_store()
    project = make_project(store)
    commitment = Commitment(
        id=new_id(),
        title="Submit physics assignment",
        description="",
        status=CommitmentStatus.OPEN,
        priority=Priority.HIGH,
        due_at="2026-03-01T00:00:00+00:00",
        source=EntitySource.USER,
        project_id=project.id,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    store.create_commitment(commitment)

    assert store.get_commitment(commitment.id) == commitment
    assert [c.id for c in store.list_commitments(project_id=project.id)] == [commitment.id]

    commitment.status = CommitmentStatus.FULFILLED
    store.save_commitment(commitment)
    assert store.get_commitment(commitment.id).status == CommitmentStatus.FULFILLED
    assert [
        c.id for c in store.list_commitments(status=CommitmentStatus.FULFILLED)
    ] == [commitment.id]


def test_commitment_project_id_is_optional():
    store = make_store()
    commitment = Commitment(
        id=new_id(),
        title="Standalone obligation",
        description="",
        status=CommitmentStatus.OPEN,
        priority=Priority.LOW,
        due_at=None,
        source=EntitySource.USER,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    store.create_commitment(commitment)
    assert store.get_commitment(commitment.id).project_id is None


# -- Event ------------------------------------------------------------------


def test_create_get_list_and_save_event():
    store = make_store()
    project = make_project(store)
    event = Event(
        id=new_id(),
        title="Robotics club meeting",
        description="",
        starts_at="2026-02-01T10:00:00+00:00",
        ends_at="2026-02-01T11:00:00+00:00",
        location="Lab",
        source=EntitySource.CALENDAR,
        external_id="gcal-123",
        project_id=project.id,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    store.create_event(event)

    assert store.get_event(event.id) == event
    assert [e.id for e in store.list_events(project_id=project.id)] == [event.id]

    event.location = "Workshop"
    store.save_event(event)
    assert store.get_event(event.id).location == "Workshop"


def test_list_events_orders_soonest_first():
    store = make_store()
    later = Event(
        id=new_id(),
        title="later",
        description="",
        starts_at="2026-05-01T00:00:00+00:00",
        ends_at="2026-05-01T01:00:00+00:00",
        location="",
        source=EntitySource.USER,
        external_id=None,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    sooner = Event(
        id=new_id(),
        title="sooner",
        description="",
        starts_at="2026-02-01T00:00:00+00:00",
        ends_at="2026-02-01T01:00:00+00:00",
        location="",
        source=EntitySource.USER,
        external_id=None,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    store.create_event(later)
    store.create_event(sooner)

    assert [e.id for e in store.list_events()] == [sooner.id, later.id]


# -- Resource -----------------------------------------------------------


def test_create_get_list_and_save_resource():
    store = make_store()
    resource = Resource(
        id=new_id(),
        name="Arduino UNO Q",
        type=ResourceType.HARDWARE,
        status=ResourceStatus.AVAILABLE,
        description="Microcontroller board",
        metadata={"quantity": 2},
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    store.create_resource(resource)

    assert store.get_resource(resource.id) == resource
    assert [
        r.id for r in store.list_resources(type=ResourceType.HARDWARE)
    ] == [resource.id]

    resource.status = ResourceStatus.IN_USE
    store.save_resource(resource)
    assert store.get_resource(resource.id).status == ResourceStatus.IN_USE
    assert [
        r.id for r in store.list_resources(status=ResourceStatus.IN_USE)
    ] == [resource.id]


# -- Blocker --------------------------------------------------------------


def test_create_get_list_and_save_blocker():
    store = make_store()
    blocker = Blocker(
        id=new_id(),
        title="Printer nozzle isn't heating",
        description="",
        status=BlockerStatus.ACTIVE,
        severity=BlockerSeverity.HIGH,
        detected_at="2026-01-01T00:00:00+00:00",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    store.create_blocker(blocker)

    assert store.get_blocker(blocker.id) == blocker
    assert [b.id for b in store.list_blockers(status=BlockerStatus.ACTIVE)] == [blocker.id]

    blocker.status = BlockerStatus.RESOLVED
    blocker.resolved_at = "2026-01-02T00:00:00+00:00"
    store.save_blocker(blocker)
    fetched = store.get_blocker(blocker.id)
    assert fetched.status == BlockerStatus.RESOLVED
    assert fetched.resolved_at == "2026-01-02T00:00:00+00:00"


# -- Preference -------------------------------------------------------------


def test_create_get_list_and_save_preference():
    store = make_store()
    preference = Preference(
        id=new_id(),
        key="response_style",
        value="concise",
        category="communication",
        source=EntitySource.USER,
        confidence=1.0,
        updated_at="2026-01-01T00:00:00+00:00",
    )
    store.create_preference(preference)

    assert store.get_preference(preference.id) == preference
    assert [
        p.id for p in store.list_preferences(key="response_style")
    ] == [preference.id]
    assert [
        p.id for p in store.list_preferences(category="communication")
    ] == [preference.id]

    preference.value = "detailed"
    store.save_preference(preference)
    assert store.get_preference(preference.id).value == "detailed"


# -- TaskLink ---------------------------------------------------------------


def test_create_list_and_delete_task_link():
    store = make_store()
    project = make_project(store)
    link = TaskLink(
        id=new_id(),
        ref_type=TaskRefType.JOB,
        ref_id="some-job-id",
        project_id=project.id,
        goal_id=None,
        created_at="2026-01-01T00:00:00+00:00",
    )
    store.create_task_link(link)

    assert store.get_task_link(link.id) == link
    assert [l.id for l in store.list_task_links(project_id=project.id)] == [link.id]

    assert store.delete_task_link(link.id) is True
    assert store.get_task_link(link.id) is None
    assert store.delete_task_link(link.id) is False


# -- GoalDependency -------------------------------------------------------


def test_create_list_and_delete_goal_dependency():
    store = make_store()
    dependency = GoalDependency(
        id=new_id(),
        goal_id="goal-a",
        depends_on_goal_id="goal-b",
        created_at="2026-01-01T00:00:00+00:00",
    )
    store.create_goal_dependency(dependency)

    assert [d.id for d in store.list_goal_dependencies("goal-a")] == [dependency.id]
    assert [d.id for d in store.list_goal_dependents("goal-b")] == [dependency.id]
    assert store.list_goal_dependencies("goal-b") == []

    assert store.delete_goal_dependency(dependency.id) is True
    assert store.list_goal_dependencies("goal-a") == []


# -- BlockerLink --------------------------------------------------------


def test_create_list_and_delete_blocker_link():
    store = make_store()
    blocker = Blocker(
        id=new_id(),
        title="blocked",
        description="",
        status=BlockerStatus.ACTIVE,
        severity=BlockerSeverity.LOW,
        detected_at="2026-01-01T00:00:00+00:00",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    store.create_blocker(blocker)
    link = BlockerLink(
        id=new_id(),
        blocker_id=blocker.id,
        entity_kind=LinkedEntityKind.PROJECT,
        entity_id="project-x",
        created_at="2026-01-01T00:00:00+00:00",
    )
    store.create_blocker_link(link)

    assert [l.id for l in store.list_blocker_links_for_blocker(blocker.id)] == [link.id]
    assert [
        l.id
        for l in store.list_blocker_links_for_entity(LinkedEntityKind.PROJECT, "project-x")
    ] == [link.id]

    assert store.delete_blocker_link(link.id) is True
    assert store.list_blocker_links_for_blocker(blocker.id) == []


# -- ResourceLink -----------------------------------------------------------


def test_create_list_and_delete_resource_link():
    store = make_store()
    resource = Resource(
        id=new_id(),
        name="3D printer",
        type=ResourceType.DEVICE,
        status=ResourceStatus.AVAILABLE,
        description="",
        metadata={},
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    store.create_resource(resource)
    link = ResourceLink(
        id=new_id(),
        resource_id=resource.id,
        entity_kind=LinkedEntityKind.TASK,
        entity_id="task-1",
        created_at="2026-01-01T00:00:00+00:00",
    )
    store.create_resource_link(link)

    assert [l.id for l in store.list_resource_links_for_resource(resource.id)] == [link.id]
    assert [
        l.id
        for l in store.list_resource_links_for_entity(LinkedEntityKind.TASK, "task-1")
    ] == [link.id]

    assert store.delete_resource_link(link.id) is True
    assert store.list_resource_links_for_resource(resource.id) == []


# -- Restart / persistence survival -----------------------------------------


def test_all_entities_and_relationships_survive_a_restart(tmp_path):
    db_path = str(tmp_path / "world_model.db")
    store = SQLiteWorldModelStore(db_path)

    project = make_project(store)
    goal = Goal(
        id=new_id(),
        project_id=project.id,
        title="goal",
        description="",
        status=GoalStatus.ACTIVE,
        priority=Priority.MEDIUM,
        target_date=None,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    store.create_goal(goal)
    dependency = GoalDependency(
        id=new_id(), goal_id=goal.id, depends_on_goal_id=goal.id, created_at="2026-01-01T00:00:00+00:00"
    )
    store.create_goal_dependency(dependency)
    task_link = TaskLink(
        id=new_id(),
        ref_type=TaskRefType.TASK,
        ref_id="task-1",
        project_id=project.id,
        goal_id=goal.id,
        created_at="2026-01-01T00:00:00+00:00",
    )
    store.create_task_link(task_link)

    # A brand-new store instance pointed at the same file - simulates the
    # process restarting, the same "second store, same path" restart
    # convention tests/test_reminders_scheduler.py already uses.
    restarted = SQLiteWorldModelStore(db_path)

    assert restarted.get_project(project.id) == project
    assert restarted.get_goal(goal.id) == goal
    assert [d.id for d in restarted.list_goal_dependencies(goal.id)] == [dependency.id]
    assert [l.id for l in restarted.list_task_links(project_id=project.id)] == [task_link.id]
