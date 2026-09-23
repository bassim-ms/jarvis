"""
WorldModelService (M16.1): the deterministic operations layer on top of
WorldModelStore (app/world_model/base.py) - creation, updates, validated
state transitions, relationship management, and the structured query
methods a future Context Engine (M16.2+) will use.

WorldModelStore itself is a fairly "dumb" persistence layer (see its own
docstring) - this is where the actual domain rules live: transition
validation (app/world_model/base.py's validate_*_transition functions),
confidence-range validation, and cross-entity referential integrity for
every id that lives inside this same database (Project/Goal/Commitment/
Event/Resource/Blocker/Preference all get checked; an external M9
Task/M15 Job id is stored opaquely and never checked here - see
TaskLink's own docstring for why).

Nothing in this module calls an AI provider or makes any decision about
what to DO with world-model state (no scheduling, no notification, no
automatic status changes) - every operation here runs to completion from
its own explicit arguments alone. A future Context Engine may call these
methods after reasoning about what changed; this module never reasons on
its own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.world_model.base import (
    Blocker,
    BlockerLink,
    BlockerSeverity,
    BlockerStatus,
    Commitment,
    CommitmentStatus,
    EntitySource,
    Event,
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
    VALID_BLOCKER_LINK_KINDS,
    VALID_RESOURCE_LINK_KINDS,
    DEFAULT_CONFIDENCE,
    WorldModelStore,
    validate_blocker_transition,
    validate_commitment_transition,
    validate_confidence,
    validate_goal_transition,
    validate_project_transition,
)
from app.world_model.sqlite_store import new_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class UnknownEntityError(ValueError):
    """Raised when an operation references a Project/Goal/Commitment/
    Event/Resource/Blocker/Preference id that doesn't exist in this
    WorldModelStore. Never raised for an external Task/Job ref_id - see
    TaskLink's own docstring for why those aren't validated here."""


class InvalidLinkError(ValueError):
    """Raised for a relationship request that's structurally invalid -
    an entity_kind not valid for that relationship type (see
    VALID_BLOCKER_LINK_KINDS/VALID_RESOURCE_LINK_KINDS), or a TaskLink
    naming neither a project nor a goal."""


class GoalDependencyCycleError(ValueError):
    """Raised by `add_goal_dependency` when the requested dependency
    would create a cycle (directly or transitively) - a goal can never
    end up depending on itself through any chain of dependencies."""


@dataclass
class ProjectContext:
    """Everything WorldModelService currently knows about one Project,
    bundled in a single query - the "structured current state" a future
    Context Engine (or a human/dashboard) can read in one call, via
    `get_project_context`. Deliberately narrower than ContextSnapshot
    (app/world_model/base.py): this is scoped to exactly one project,
    with no notion of "relevance" - every goal/commitment/event/task-link
    that belongs to the project is included, full stop."""

    project: Project
    goals: list[Goal] = field(default_factory=list)
    commitments: list[Commitment] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    task_links: list[TaskLink] = field(default_factory=list)
    blocker_links: list[BlockerLink] = field(default_factory=list)
    resource_links: list[ResourceLink] = field(default_factory=list)


class WorldModelService:
    """The single entry point for every deterministic World Model
    operation - construct once against a WorldModelStore (see
    app/world_model/factory.py) and reuse, the same "service wraps a
    store" shape ReminderScheduler wraps a ReminderStore or JobEngine
    wraps a JobStore, just without a background thread of its own (M16.1
    has none - see this module's own docstring)."""

    def __init__(self, store: WorldModelStore):
        self.store = store

    # -- Project ----------------------------------------------------------

    def create_project(
        self,
        name: str,
        description: str = "",
        priority: Priority = Priority.MEDIUM,
        metadata: dict[str, Any] | None = None,
    ) -> Project:
        now = _now()
        project = Project(
            id=new_id(),
            name=name,
            description=description,
            status=ProjectStatus.ACTIVE,
            priority=priority,
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        self.store.create_project(project)
        return project

    def _require_project(self, project_id: str) -> Project:
        project = self.store.get_project(project_id)
        if project is None:
            raise UnknownEntityError(f"No such project: '{project_id}'")
        return project

    def update_project(
        self,
        project_id: str,
        name: str | None = None,
        description: str | None = None,
        priority: Priority | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Project:
        """Update only the given (non-None) fields - same convention as
        CalendarProvider.update_event (app/calendar/base.py). Never
        touches `status` - see `transition_project` for that."""
        project = self._require_project(project_id)
        if name is not None:
            project.name = name
        if description is not None:
            project.description = description
        if priority is not None:
            project.priority = priority
        if metadata is not None:
            project.metadata = dict(metadata)
        project.updated_at = _now()
        self.store.save_project(project)
        return project

    def transition_project(self, project_id: str, target_status: ProjectStatus) -> Project:
        project = self._require_project(project_id)
        validate_project_transition(project.status, target_status)
        project.status = target_status
        project.updated_at = _now()
        self.store.save_project(project)
        return project

    def get_project(self, project_id: str) -> Project | None:
        return self.store.get_project(project_id)

    def get_active_projects(self) -> list[Project]:
        return self.store.list_projects(status=ProjectStatus.ACTIVE)

    # -- Goal ---------------------------------------------------------------

    def create_goal(
        self,
        project_id: str,
        title: str,
        description: str = "",
        priority: Priority = Priority.MEDIUM,
        target_date: str | None = None,
    ) -> Goal:
        self._require_project(project_id)
        now = _now()
        goal = Goal(
            id=new_id(),
            project_id=project_id,
            title=title,
            description=description,
            status=GoalStatus.ACTIVE,
            priority=priority,
            target_date=target_date,
            created_at=now,
            updated_at=now,
        )
        self.store.create_goal(goal)
        return goal

    def _require_goal(self, goal_id: str) -> Goal:
        goal = self.store.get_goal(goal_id)
        if goal is None:
            raise UnknownEntityError(f"No such goal: '{goal_id}'")
        return goal

    def update_goal(
        self,
        goal_id: str,
        title: str | None = None,
        description: str | None = None,
        priority: Priority | None = None,
        target_date: str | None = None,
    ) -> Goal:
        goal = self._require_goal(goal_id)
        if title is not None:
            goal.title = title
        if description is not None:
            goal.description = description
        if priority is not None:
            goal.priority = priority
        if target_date is not None:
            goal.target_date = target_date
        goal.updated_at = _now()
        self.store.save_goal(goal)
        return goal

    def transition_goal(self, goal_id: str, target_status: GoalStatus) -> Goal:
        goal = self._require_goal(goal_id)
        validate_goal_transition(goal.status, target_status)
        goal.status = target_status
        goal.updated_at = _now()
        self.store.save_goal(goal)
        return goal

    def get_goal(self, goal_id: str) -> Goal | None:
        return self.store.get_goal(goal_id)

    def get_active_goals(self, project_id: str | None = None) -> list[Goal]:
        return self.store.list_goals(project_id=project_id, status=GoalStatus.ACTIVE)

    # -- Task/Job linkage -------------------------------------------------

    def link_task(
        self,
        ref_type: TaskRefType,
        ref_id: str,
        project_id: str | None = None,
        goal_id: str | None = None,
    ) -> TaskLink:
        """Associate an existing M9 Task or M15 Job (named by `ref_type`/
        `ref_id`, never validated against its own store - see TaskLink's
        docstring) with a project and/or goal. At least one of
        `project_id`/`goal_id` is required."""
        if project_id is None and goal_id is None:
            raise InvalidLinkError(
                "link_task requires at least one of project_id/goal_id."
            )
        if project_id is not None:
            self._require_project(project_id)
        if goal_id is not None:
            self._require_goal(goal_id)
        link = TaskLink(
            id=new_id(),
            ref_type=ref_type,
            ref_id=ref_id,
            project_id=project_id,
            goal_id=goal_id,
            created_at=_now(),
        )
        self.store.create_task_link(link)
        return link

    def unlink_task(self, link_id: str) -> bool:
        return self.store.delete_task_link(link_id)

    def get_tasks_for_goal(self, goal_id: str) -> list[TaskLink]:
        return self.store.list_task_links(goal_id=goal_id)

    def get_tasks_for_project(self, project_id: str) -> list[TaskLink]:
        return self.store.list_task_links(project_id=project_id)

    # -- Commitment -----------------------------------------------------

    def create_commitment(
        self,
        title: str,
        description: str = "",
        priority: Priority = Priority.MEDIUM,
        due_at: str | None = None,
        source: EntitySource = EntitySource.USER,
        project_id: str | None = None,
    ) -> Commitment:
        if project_id is not None:
            self._require_project(project_id)
        now = _now()
        commitment = Commitment(
            id=new_id(),
            title=title,
            description=description,
            status=CommitmentStatus.OPEN,
            priority=priority,
            due_at=due_at,
            source=source,
            project_id=project_id,
            created_at=now,
            updated_at=now,
        )
        self.store.create_commitment(commitment)
        return commitment

    def _require_commitment(self, commitment_id: str) -> Commitment:
        commitment = self.store.get_commitment(commitment_id)
        if commitment is None:
            raise UnknownEntityError(f"No such commitment: '{commitment_id}'")
        return commitment

    def update_commitment(
        self,
        commitment_id: str,
        title: str | None = None,
        description: str | None = None,
        priority: Priority | None = None,
        due_at: str | None = None,
    ) -> Commitment:
        commitment = self._require_commitment(commitment_id)
        if title is not None:
            commitment.title = title
        if description is not None:
            commitment.description = description
        if priority is not None:
            commitment.priority = priority
        if due_at is not None:
            commitment.due_at = due_at
        commitment.updated_at = _now()
        self.store.save_commitment(commitment)
        return commitment

    def transition_commitment(
        self, commitment_id: str, target_status: CommitmentStatus
    ) -> Commitment:
        commitment = self._require_commitment(commitment_id)
        validate_commitment_transition(commitment.status, target_status)
        commitment.status = target_status
        commitment.updated_at = _now()
        self.store.save_commitment(commitment)
        return commitment

    def get_commitment(self, commitment_id: str) -> Commitment | None:
        return self.store.get_commitment(commitment_id)

    def get_open_commitments(self) -> list[Commitment]:
        return self.store.list_commitments(status=CommitmentStatus.OPEN)

    # -- Event ------------------------------------------------------------

    def create_event(
        self,
        title: str,
        starts_at: str,
        ends_at: str,
        description: str = "",
        location: str = "",
        source: EntitySource = EntitySource.USER,
        external_id: str | None = None,
        project_id: str | None = None,
    ) -> Event:
        if project_id is not None:
            self._require_project(project_id)
        now = _now()
        event = Event(
            id=new_id(),
            title=title,
            description=description,
            starts_at=starts_at,
            ends_at=ends_at,
            location=location,
            source=source,
            external_id=external_id,
            project_id=project_id,
            created_at=now,
            updated_at=now,
        )
        self.store.create_event(event)
        return event

    def _require_event(self, event_id: str) -> Event:
        event = self.store.get_event(event_id)
        if event is None:
            raise UnknownEntityError(f"No such event: '{event_id}'")
        return event

    def update_event(
        self,
        event_id: str,
        title: str | None = None,
        description: str | None = None,
        starts_at: str | None = None,
        ends_at: str | None = None,
        location: str | None = None,
        external_id: str | None = None,
    ) -> Event:
        event = self._require_event(event_id)
        if title is not None:
            event.title = title
        if description is not None:
            event.description = description
        if starts_at is not None:
            event.starts_at = starts_at
        if ends_at is not None:
            event.ends_at = ends_at
        if location is not None:
            event.location = location
        if external_id is not None:
            event.external_id = external_id
        event.updated_at = _now()
        self.store.save_event(event)
        return event

    def get_event(self, event_id: str) -> Event | None:
        return self.store.get_event(event_id)

    def get_upcoming_events(self, after: str | None = None) -> list[Event]:
        """Events starting at or after `after` (default: now), soonest
        first. A plain string comparison against ISO 8601 UTC timestamps
        - the same convention every other timestamp comparison in this
        codebase relies on (they sort correctly as strings)."""
        cutoff = after if after is not None else _now()
        return [e for e in self.store.list_events() if e.starts_at >= cutoff]

    # -- Resource -------------------------------------------------------

    def create_resource(
        self,
        name: str,
        type: ResourceType,
        status: ResourceStatus = ResourceStatus.AVAILABLE,
        description: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Resource:
        now = _now()
        resource = Resource(
            id=new_id(),
            name=name,
            type=type,
            status=status,
            description=description,
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        self.store.create_resource(resource)
        return resource

    def _require_resource(self, resource_id: str) -> Resource:
        resource = self.store.get_resource(resource_id)
        if resource is None:
            raise UnknownEntityError(f"No such resource: '{resource_id}'")
        return resource

    def update_resource(
        self,
        resource_id: str,
        name: str | None = None,
        status: ResourceStatus | None = None,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Resource:
        resource = self._require_resource(resource_id)
        if name is not None:
            resource.name = name
        if status is not None:
            resource.status = status
        if description is not None:
            resource.description = description
        if metadata is not None:
            resource.metadata = dict(metadata)
        resource.updated_at = _now()
        self.store.save_resource(resource)
        return resource

    def get_resource(self, resource_id: str) -> Resource | None:
        return self.store.get_resource(resource_id)

    def get_resources_for_entity(
        self, entity_kind: LinkedEntityKind, entity_id: str
    ) -> list[Resource]:
        links = self.store.list_resource_links_for_entity(entity_kind, entity_id)
        resources = [self.store.get_resource(link.resource_id) for link in links]
        return [r for r in resources if r is not None]

    # -- Blocker ----------------------------------------------------------

    def create_blocker(
        self,
        title: str,
        description: str = "",
        severity: BlockerSeverity = BlockerSeverity.MEDIUM,
        detected_at: str | None = None,
    ) -> Blocker:
        now = _now()
        blocker = Blocker(
            id=new_id(),
            title=title,
            description=description,
            status=BlockerStatus.ACTIVE,
            severity=severity,
            detected_at=detected_at or now,
            resolved_at=None,
            created_at=now,
            updated_at=now,
        )
        self.store.create_blocker(blocker)
        return blocker

    def _require_blocker(self, blocker_id: str) -> Blocker:
        blocker = self.store.get_blocker(blocker_id)
        if blocker is None:
            raise UnknownEntityError(f"No such blocker: '{blocker_id}'")
        return blocker

    def update_blocker(
        self,
        blocker_id: str,
        title: str | None = None,
        description: str | None = None,
        severity: BlockerSeverity | None = None,
    ) -> Blocker:
        blocker = self._require_blocker(blocker_id)
        if title is not None:
            blocker.title = title
        if description is not None:
            blocker.description = description
        if severity is not None:
            blocker.severity = severity
        blocker.updated_at = _now()
        self.store.save_blocker(blocker)
        return blocker

    def transition_blocker(self, blocker_id: str, target_status: BlockerStatus) -> Blocker:
        blocker = self._require_blocker(blocker_id)
        validate_blocker_transition(blocker.status, target_status)
        blocker.status = target_status
        now = _now()
        # `resolved_at` only means something once a blocker is actually
        # RESOLVED (fixed) - a DISMISSED blocker was never fixed, so it
        # stays None, matching the field's own "resolved, not just
        # closed" meaning (see Blocker's docstring).
        if target_status == BlockerStatus.RESOLVED:
            blocker.resolved_at = now
        blocker.updated_at = now
        self.store.save_blocker(blocker)
        return blocker

    def get_blocker(self, blocker_id: str) -> Blocker | None:
        return self.store.get_blocker(blocker_id)

    def get_active_blockers(self) -> list[Blocker]:
        return self.store.list_blockers(status=BlockerStatus.ACTIVE)

    # -- Preference -----------------------------------------------------

    def create_preference(
        self,
        key: str,
        value: str,
        category: str = "general",
        source: EntitySource = EntitySource.USER,
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> Preference:
        validate_confidence(confidence)
        preference = Preference(
            id=new_id(),
            key=key,
            value=value,
            category=category,
            source=source,
            confidence=confidence,
            updated_at=_now(),
        )
        self.store.create_preference(preference)
        return preference

    def _require_preference(self, preference_id: str) -> Preference:
        preference = self.store.get_preference(preference_id)
        if preference is None:
            raise UnknownEntityError(f"No such preference: '{preference_id}'")
        return preference

    def update_preference(
        self,
        preference_id: str,
        value: str | None = None,
        category: str | None = None,
        source: EntitySource | None = None,
        confidence: float | None = None,
    ) -> Preference:
        preference = self._require_preference(preference_id)
        if value is not None:
            preference.value = value
        if category is not None:
            preference.category = category
        if source is not None:
            preference.source = source
        if confidence is not None:
            validate_confidence(confidence)
            preference.confidence = confidence
        preference.updated_at = _now()
        self.store.save_preference(preference)
        return preference

    def get_preference(self, preference_id: str) -> Preference | None:
        return self.store.get_preference(preference_id)

    def get_preferences(
        self, key: str | None = None, category: str | None = None
    ) -> list[Preference]:
        return self.store.list_preferences(key=key, category=category)

    # -- Blocker relationships --------------------------------------------

    def add_blocker_link(
        self, blocker_id: str, entity_kind: LinkedEntityKind, entity_id: str
    ) -> BlockerLink:
        self._require_blocker(blocker_id)
        if entity_kind not in VALID_BLOCKER_LINK_KINDS:
            raise InvalidLinkError(
                f"A blocker cannot link to entity_kind '{entity_kind.value}'. "
                f"Must be one of: {sorted(k.value for k in VALID_BLOCKER_LINK_KINDS)}"
            )
        # Only PROJECT/GOAL live in this store and can be cheaply
        # existence-checked; TASK/JOB refs are opaque (see TaskLink's
        # docstring for the same cross-database reasoning).
        if entity_kind == LinkedEntityKind.PROJECT:
            self._require_project(entity_id)
        elif entity_kind == LinkedEntityKind.GOAL:
            self._require_goal(entity_id)
        link = BlockerLink(
            id=new_id(),
            blocker_id=blocker_id,
            entity_kind=entity_kind,
            entity_id=entity_id,
            created_at=_now(),
        )
        self.store.create_blocker_link(link)
        return link

    def remove_blocker_link(self, link_id: str) -> bool:
        return self.store.delete_blocker_link(link_id)

    def get_entities_affected_by_blocker(self, blocker_id: str) -> list[BlockerLink]:
        self._require_blocker(blocker_id)
        return self.store.list_blocker_links_for_blocker(blocker_id)

    def get_blockers_for_entity(
        self, entity_kind: LinkedEntityKind, entity_id: str
    ) -> list[Blocker]:
        links = self.store.list_blocker_links_for_entity(entity_kind, entity_id)
        blockers = [self.store.get_blocker(link.blocker_id) for link in links]
        return [b for b in blockers if b is not None]

    # -- Resource relationships -------------------------------------------

    def add_resource_link(
        self, resource_id: str, entity_kind: LinkedEntityKind, entity_id: str
    ) -> ResourceLink:
        self._require_resource(resource_id)
        if entity_kind not in VALID_RESOURCE_LINK_KINDS:
            raise InvalidLinkError(
                f"A resource cannot link to entity_kind '{entity_kind.value}'. "
                f"Must be one of: {sorted(k.value for k in VALID_RESOURCE_LINK_KINDS)}"
            )
        if entity_kind == LinkedEntityKind.PROJECT:
            self._require_project(entity_id)
        link = ResourceLink(
            id=new_id(),
            resource_id=resource_id,
            entity_kind=entity_kind,
            entity_id=entity_id,
            created_at=_now(),
        )
        self.store.create_resource_link(link)
        return link

    def remove_resource_link(self, link_id: str) -> bool:
        return self.store.delete_resource_link(link_id)

    # -- Goal dependencies --------------------------------------------------

    def add_goal_dependency(self, goal_id: str, depends_on_goal_id: str) -> GoalDependency:
        if goal_id == depends_on_goal_id:
            raise GoalDependencyCycleError("A goal cannot depend on itself.")
        self._require_goal(goal_id)
        self._require_goal(depends_on_goal_id)
        if self._goal_depends_on(depends_on_goal_id, goal_id, set()):
            raise GoalDependencyCycleError(
                f"Adding this dependency would create a cycle: '{depends_on_goal_id}' "
                f"already (transitively) depends on '{goal_id}'."
            )
        dependency = GoalDependency(
            id=new_id(),
            goal_id=goal_id,
            depends_on_goal_id=depends_on_goal_id,
            created_at=_now(),
        )
        self.store.create_goal_dependency(dependency)
        return dependency

    def _goal_depends_on(self, goal_id: str, target_id: str, seen: set[str]) -> bool:
        """True if `goal_id` (transitively) depends on `target_id` -
        used by `add_goal_dependency` to reject a dependency that would
        close a cycle. `seen` guards against revisiting a goal already
        walked in this same check (defensive; the same cycle-prevention
        this function exists for also keeps the underlying data
        acyclic)."""
        if goal_id in seen:
            return False
        seen.add(goal_id)
        for dependency in self.store.list_goal_dependencies(goal_id):
            if dependency.depends_on_goal_id == target_id:
                return True
            if self._goal_depends_on(dependency.depends_on_goal_id, target_id, seen):
                return True
        return False

    def remove_goal_dependency(self, dependency_id: str) -> bool:
        return self.store.delete_goal_dependency(dependency_id)

    def get_goal_dependencies(self, goal_id: str) -> list[GoalDependency]:
        """Goals `goal_id` itself depends on."""
        return self.store.list_goal_dependencies(goal_id)

    def get_goal_dependents(self, goal_id: str) -> list[GoalDependency]:
        """Goals that depend on `goal_id`."""
        return self.store.list_goal_dependents(goal_id)

    # -- Composite queries --------------------------------------------------

    def get_project_context(self, project_id: str) -> ProjectContext:
        project = self._require_project(project_id)
        goals = self.store.list_goals(project_id=project_id)
        blocker_links: list[BlockerLink] = list(
            self.store.list_blocker_links_for_entity(LinkedEntityKind.PROJECT, project_id)
        )
        resource_links: list[ResourceLink] = list(
            self.store.list_resource_links_for_entity(LinkedEntityKind.PROJECT, project_id)
        )
        for goal in goals:
            blocker_links.extend(
                self.store.list_blocker_links_for_entity(LinkedEntityKind.GOAL, goal.id)
            )
        return ProjectContext(
            project=project,
            goals=goals,
            commitments=self.store.list_commitments(project_id=project_id),
            events=self.store.list_events(project_id=project_id),
            task_links=self.store.list_task_links(project_id=project_id),
            blocker_links=blocker_links,
            resource_links=resource_links,
        )
