"""
SQLite-backed implementation of WorldModelStore, using Python's stdlib
sqlite3 - the exact same pattern as SQLiteJobStore/SQLiteTaskStore: one
connection held for the store's lifetime, guarded by a lock since
multiple threads (a future Context Engine, tests, tools) can call into
it concurrently, backed by a plain local file so every entity survives a
Jarvis restart with no special recovery code.

One file, many tables - the same "one store, several typed tables" shape
SQLiteJobStore already uses for jobs/plan/events (there, folded into one
JSON column per Job; here, spread across real tables since these entities
are independently queried/filtered far more often than a Job's own
nested plan is). Fixed-shape fields get their own typed columns;
`metadata` (Project/Resource) is the only JSON column, for the same
reason app/orchestration/sqlite_store.py keeps `context`/`events` as
JSON - its shape genuinely varies per caller.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from app.world_model.base import (
    Blocker,
    BlockerLink,
    BlockerStatus,
    BlockerSeverity,
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
    WorldModelStore,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    """Shared id generator for every World Model entity/relationship -
    matches app.orchestration.base.Job's own `str(uuid.uuid4())` id
    shape."""
    return str(uuid.uuid4())


#: The World Model's default db file path - deliberately NOT a
#: `settings.*` field (M16.1's own correction: "the World Model should
#: continue receiving db_path explicitly through its store/factory
#: APIs," never a config mechanism). Named here, once, as the single
#: source of truth for that literal string, so `app/main.py` (the one
#: real caller that needs a concrete path to activate the World Model)
#: imports and passes this explicitly rather than duplicating the
#: string as a second hardcoded literal - see app/main.py's own wiring.
DEFAULT_WORLD_MODEL_DB_PATH = "jarvis_world_model.db"


class SQLiteWorldModelStore(WorldModelStore):
    """WorldModelStore backed by a local SQLite database file."""

    def __init__(self, db_path: str = DEFAULT_WORLD_MODEL_DB_PATH):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        # "Forget" must really forget: have SQLite overwrite deleted content with zeros instead of leaving it in
        # the file's free pages.
        self._conn.execute("PRAGMA secure_delete = ON")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS goals (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    target_date TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS commitments (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    due_at TEXT,
                    source TEXT NOT NULL,
                    project_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    starts_at TEXT NOT NULL,
                    ends_at TEXT NOT NULL,
                    location TEXT NOT NULL,
                    source TEXT NOT NULL,
                    external_id TEXT,
                    project_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS resources (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    description TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS blockers (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    detected_at TEXT NOT NULL,
                    resolved_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS preferences (
                    id TEXT PRIMARY KEY,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    category TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS task_links (
                    id TEXT PRIMARY KEY,
                    ref_type TEXT NOT NULL,
                    ref_id TEXT NOT NULL,
                    project_id TEXT,
                    goal_id TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS goal_dependencies (
                    id TEXT PRIMARY KEY,
                    goal_id TEXT NOT NULL,
                    depends_on_goal_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS blocker_links (
                    id TEXT PRIMARY KEY,
                    blocker_id TEXT NOT NULL,
                    entity_kind TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS resource_links (
                    id TEXT PRIMARY KEY,
                    resource_id TEXT NOT NULL,
                    entity_kind TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    # -- Project --------------------------------------------------------

    def create_project(self, project: Project) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO projects
                    (id, name, description, status, priority, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project.id,
                    project.name,
                    project.description,
                    project.status.value,
                    project.priority.value,
                    json.dumps(project.metadata),
                    project.created_at,
                    project.updated_at,
                ),
            )
            self._conn.commit()

    def get_project(self, project_id: str) -> Project | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM projects WHERE id = ?", (project_id,)
            ).fetchone()
        return self._row_to_project(row) if row else None

    def list_projects(self, status: ProjectStatus | None = None) -> list[Project]:
        with self._lock:
            if status is None:
                rows = self._conn.execute(
                    "SELECT * FROM projects ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM projects WHERE status = ? ORDER BY created_at DESC",
                    (status.value,),
                ).fetchall()
        return [self._row_to_project(row) for row in rows]

    def save_project(self, project: Project) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE projects SET
                    name = ?, description = ?, status = ?, priority = ?,
                    metadata = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    project.name,
                    project.description,
                    project.status.value,
                    project.priority.value,
                    json.dumps(project.metadata),
                    project.updated_at,
                    project.id,
                ),
            )
            self._conn.commit()

    @staticmethod
    def _row_to_project(row: sqlite3.Row) -> Project:
        return Project(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            status=ProjectStatus(row["status"]),
            priority=Priority(row["priority"]),
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # -- Goal ---------------------------------------------------------------

    def create_goal(self, goal: Goal) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO goals
                    (id, project_id, title, description, status, priority,
                     target_date, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    goal.id,
                    goal.project_id,
                    goal.title,
                    goal.description,
                    goal.status.value,
                    goal.priority.value,
                    goal.target_date,
                    goal.created_at,
                    goal.updated_at,
                ),
            )
            self._conn.commit()

    def get_goal(self, goal_id: str) -> Goal | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM goals WHERE id = ?", (goal_id,)
            ).fetchone()
        return self._row_to_goal(row) if row else None

    def list_goals(
        self, project_id: str | None = None, status: GoalStatus | None = None
    ) -> list[Goal]:
        clauses: list[str] = []
        params: list[Any] = []
        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(project_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM goals {where} ORDER BY created_at DESC", params
            ).fetchall()
        return [self._row_to_goal(row) for row in rows]

    def save_goal(self, goal: Goal) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE goals SET
                    title = ?, description = ?, status = ?, priority = ?,
                    target_date = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    goal.title,
                    goal.description,
                    goal.status.value,
                    goal.priority.value,
                    goal.target_date,
                    goal.updated_at,
                    goal.id,
                ),
            )
            self._conn.commit()

    @staticmethod
    def _row_to_goal(row: sqlite3.Row) -> Goal:
        return Goal(
            id=row["id"],
            project_id=row["project_id"],
            title=row["title"],
            description=row["description"],
            status=GoalStatus(row["status"]),
            priority=Priority(row["priority"]),
            target_date=row["target_date"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # -- Commitment -----------------------------------------------------

    def create_commitment(self, commitment: Commitment) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO commitments
                    (id, title, description, status, priority, due_at, source,
                     project_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    commitment.id,
                    commitment.title,
                    commitment.description,
                    commitment.status.value,
                    commitment.priority.value,
                    commitment.due_at,
                    commitment.source.value,
                    commitment.project_id,
                    commitment.created_at,
                    commitment.updated_at,
                ),
            )
            self._conn.commit()

    def get_commitment(self, commitment_id: str) -> Commitment | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM commitments WHERE id = ?", (commitment_id,)
            ).fetchone()
        return self._row_to_commitment(row) if row else None

    def list_commitments(
        self, project_id: str | None = None, status: CommitmentStatus | None = None
    ) -> list[Commitment]:
        clauses: list[str] = []
        params: list[Any] = []
        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(project_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM commitments {where} ORDER BY created_at DESC", params
            ).fetchall()
        return [self._row_to_commitment(row) for row in rows]

    def save_commitment(self, commitment: Commitment) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE commitments SET
                    title = ?, description = ?, status = ?, priority = ?,
                    due_at = ?, source = ?, project_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    commitment.title,
                    commitment.description,
                    commitment.status.value,
                    commitment.priority.value,
                    commitment.due_at,
                    commitment.source.value,
                    commitment.project_id,
                    commitment.updated_at,
                    commitment.id,
                ),
            )
            self._conn.commit()

    @staticmethod
    def _row_to_commitment(row: sqlite3.Row) -> Commitment:
        return Commitment(
            id=row["id"],
            title=row["title"],
            description=row["description"],
            status=CommitmentStatus(row["status"]),
            priority=Priority(row["priority"]),
            due_at=row["due_at"],
            source=EntitySource(row["source"]),
            project_id=row["project_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # -- Event ------------------------------------------------------------

    def create_event(self, event: Event) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO events
                    (id, title, description, starts_at, ends_at, location,
                     source, external_id, project_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.title,
                    event.description,
                    event.starts_at,
                    event.ends_at,
                    event.location,
                    event.source.value,
                    event.external_id,
                    event.project_id,
                    event.created_at,
                    event.updated_at,
                ),
            )
            self._conn.commit()

    def get_event(self, event_id: str) -> Event | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM events WHERE id = ?", (event_id,)
            ).fetchone()
        return self._row_to_event(row) if row else None

    def list_events(self, project_id: str | None = None) -> list[Event]:
        with self._lock:
            if project_id is None:
                rows = self._conn.execute(
                    "SELECT * FROM events ORDER BY starts_at ASC"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM events WHERE project_id = ? ORDER BY starts_at ASC",
                    (project_id,),
                ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def save_event(self, event: Event) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE events SET
                    title = ?, description = ?, starts_at = ?, ends_at = ?,
                    location = ?, source = ?, external_id = ?, project_id = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    event.title,
                    event.description,
                    event.starts_at,
                    event.ends_at,
                    event.location,
                    event.source.value,
                    event.external_id,
                    event.project_id,
                    event.updated_at,
                    event.id,
                ),
            )
            self._conn.commit()

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Event:
        return Event(
            id=row["id"],
            title=row["title"],
            description=row["description"],
            starts_at=row["starts_at"],
            ends_at=row["ends_at"],
            location=row["location"],
            source=EntitySource(row["source"]),
            external_id=row["external_id"],
            project_id=row["project_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # -- Resource -------------------------------------------------------

    def create_resource(self, resource: Resource) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO resources
                    (id, name, type, status, description, metadata, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resource.id,
                    resource.name,
                    resource.type.value,
                    resource.status.value,
                    resource.description,
                    json.dumps(resource.metadata),
                    resource.created_at,
                    resource.updated_at,
                ),
            )
            self._conn.commit()

    def get_resource(self, resource_id: str) -> Resource | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM resources WHERE id = ?", (resource_id,)
            ).fetchone()
        return self._row_to_resource(row) if row else None

    def list_resources(
        self, type: ResourceType | None = None, status: ResourceStatus | None = None
    ) -> list[Resource]:
        clauses: list[str] = []
        params: list[Any] = []
        if type is not None:
            clauses.append("type = ?")
            params.append(type.value)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM resources {where} ORDER BY created_at DESC", params
            ).fetchall()
        return [self._row_to_resource(row) for row in rows]

    def save_resource(self, resource: Resource) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE resources SET
                    name = ?, type = ?, status = ?, description = ?,
                    metadata = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    resource.name,
                    resource.type.value,
                    resource.status.value,
                    resource.description,
                    json.dumps(resource.metadata),
                    resource.updated_at,
                    resource.id,
                ),
            )
            self._conn.commit()

    @staticmethod
    def _row_to_resource(row: sqlite3.Row) -> Resource:
        return Resource(
            id=row["id"],
            name=row["name"],
            type=ResourceType(row["type"]),
            status=ResourceStatus(row["status"]),
            description=row["description"],
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # -- Blocker ----------------------------------------------------------

    def create_blocker(self, blocker: Blocker) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO blockers
                    (id, title, description, status, severity, detected_at,
                     resolved_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    blocker.id,
                    blocker.title,
                    blocker.description,
                    blocker.status.value,
                    blocker.severity.value,
                    blocker.detected_at,
                    blocker.resolved_at,
                    blocker.created_at,
                    blocker.updated_at,
                ),
            )
            self._conn.commit()

    def get_blocker(self, blocker_id: str) -> Blocker | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM blockers WHERE id = ?", (blocker_id,)
            ).fetchone()
        return self._row_to_blocker(row) if row else None

    def list_blockers(self, status: BlockerStatus | None = None) -> list[Blocker]:
        with self._lock:
            if status is None:
                rows = self._conn.execute(
                    "SELECT * FROM blockers ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM blockers WHERE status = ? ORDER BY created_at DESC",
                    (status.value,),
                ).fetchall()
        return [self._row_to_blocker(row) for row in rows]

    def save_blocker(self, blocker: Blocker) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE blockers SET
                    title = ?, description = ?, status = ?, severity = ?,
                    detected_at = ?, resolved_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    blocker.title,
                    blocker.description,
                    blocker.status.value,
                    blocker.severity.value,
                    blocker.detected_at,
                    blocker.resolved_at,
                    blocker.updated_at,
                    blocker.id,
                ),
            )
            self._conn.commit()

    @staticmethod
    def _row_to_blocker(row: sqlite3.Row) -> Blocker:
        return Blocker(
            id=row["id"],
            title=row["title"],
            description=row["description"],
            status=BlockerStatus(row["status"]),
            severity=BlockerSeverity(row["severity"]),
            detected_at=row["detected_at"],
            resolved_at=row["resolved_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # -- Preference -----------------------------------------------------

    def create_preference(self, preference: Preference) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO preferences
                    (id, key, value, category, source, confidence, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    preference.id,
                    preference.key,
                    preference.value,
                    preference.category,
                    preference.source.value,
                    preference.confidence,
                    preference.updated_at,
                ),
            )
            self._conn.commit()

    def get_preference(self, preference_id: str) -> Preference | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM preferences WHERE id = ?", (preference_id,)
            ).fetchone()
        return self._row_to_preference(row) if row else None

    def list_preferences(
        self, key: str | None = None, category: str | None = None
    ) -> list[Preference]:
        clauses: list[str] = []
        params: list[Any] = []
        if key is not None:
            clauses.append("key = ?")
            params.append(key)
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM preferences {where} ORDER BY updated_at DESC", params
            ).fetchall()
        return [self._row_to_preference(row) for row in rows]

    def delete_preference(self, preference_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM preferences WHERE id = ?", (preference_id,))
            self._conn.commit()
        return cursor.rowcount > 0

    def save_preference(self, preference: Preference) -> None:
        with self._lock:
            self._conn.execute(
                """
                UPDATE preferences SET
                    key = ?, value = ?, category = ?, source = ?,
                    confidence = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    preference.key,
                    preference.value,
                    preference.category,
                    preference.source.value,
                    preference.confidence,
                    preference.updated_at,
                    preference.id,
                ),
            )
            self._conn.commit()

    @staticmethod
    def _row_to_preference(row: sqlite3.Row) -> Preference:
        return Preference(
            id=row["id"],
            key=row["key"],
            value=row["value"],
            category=row["category"],
            source=EntitySource(row["source"]),
            confidence=row["confidence"],
            updated_at=row["updated_at"],
        )

    # -- TaskLink -------------------------------------------------------

    def create_task_link(self, link: TaskLink) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO task_links
                    (id, ref_type, ref_id, project_id, goal_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    link.id,
                    link.ref_type.value,
                    link.ref_id,
                    link.project_id,
                    link.goal_id,
                    link.created_at,
                ),
            )
            self._conn.commit()

    def get_task_link(self, link_id: str) -> TaskLink | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM task_links WHERE id = ?", (link_id,)
            ).fetchone()
        return self._row_to_task_link(row) if row else None

    def list_task_links(
        self, project_id: str | None = None, goal_id: str | None = None
    ) -> list[TaskLink]:
        clauses: list[str] = []
        params: list[Any] = []
        if project_id is not None:
            clauses.append("project_id = ?")
            params.append(project_id)
        if goal_id is not None:
            clauses.append("goal_id = ?")
            params.append(goal_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM task_links {where} ORDER BY created_at DESC", params
            ).fetchall()
        return [self._row_to_task_link(row) for row in rows]

    def delete_task_link(self, link_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM task_links WHERE id = ?", (link_id,)
            )
            self._conn.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_task_link(row: sqlite3.Row) -> TaskLink:
        return TaskLink(
            id=row["id"],
            ref_type=TaskRefType(row["ref_type"]),
            ref_id=row["ref_id"],
            project_id=row["project_id"],
            goal_id=row["goal_id"],
            created_at=row["created_at"],
        )

    # -- GoalDependency -----------------------------------------------------

    def create_goal_dependency(self, dependency: GoalDependency) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO goal_dependencies
                    (id, goal_id, depends_on_goal_id, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    dependency.id,
                    dependency.goal_id,
                    dependency.depends_on_goal_id,
                    dependency.created_at,
                ),
            )
            self._conn.commit()

    def list_goal_dependencies(self, goal_id: str) -> list[GoalDependency]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM goal_dependencies WHERE goal_id = ? ORDER BY created_at DESC",
                (goal_id,),
            ).fetchall()
        return [self._row_to_goal_dependency(row) for row in rows]

    def list_goal_dependents(self, goal_id: str) -> list[GoalDependency]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM goal_dependencies WHERE depends_on_goal_id = ? "
                "ORDER BY created_at DESC",
                (goal_id,),
            ).fetchall()
        return [self._row_to_goal_dependency(row) for row in rows]

    def delete_goal_dependency(self, dependency_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM goal_dependencies WHERE id = ?", (dependency_id,)
            )
            self._conn.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_goal_dependency(row: sqlite3.Row) -> GoalDependency:
        return GoalDependency(
            id=row["id"],
            goal_id=row["goal_id"],
            depends_on_goal_id=row["depends_on_goal_id"],
            created_at=row["created_at"],
        )

    # -- BlockerLink ------------------------------------------------------

    def create_blocker_link(self, link: BlockerLink) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO blocker_links
                    (id, blocker_id, entity_kind, entity_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    link.id,
                    link.blocker_id,
                    link.entity_kind.value,
                    link.entity_id,
                    link.created_at,
                ),
            )
            self._conn.commit()

    def list_blocker_links_for_blocker(self, blocker_id: str) -> list[BlockerLink]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM blocker_links WHERE blocker_id = ? ORDER BY created_at DESC",
                (blocker_id,),
            ).fetchall()
        return [self._row_to_blocker_link(row) for row in rows]

    def list_blocker_links_for_entity(
        self, entity_kind: LinkedEntityKind, entity_id: str
    ) -> list[BlockerLink]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM blocker_links WHERE entity_kind = ? AND entity_id = ? "
                "ORDER BY created_at DESC",
                (entity_kind.value, entity_id),
            ).fetchall()
        return [self._row_to_blocker_link(row) for row in rows]

    def delete_blocker_link(self, link_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM blocker_links WHERE id = ?", (link_id,)
            )
            self._conn.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_blocker_link(row: sqlite3.Row) -> BlockerLink:
        return BlockerLink(
            id=row["id"],
            blocker_id=row["blocker_id"],
            entity_kind=LinkedEntityKind(row["entity_kind"]),
            entity_id=row["entity_id"],
            created_at=row["created_at"],
        )

    # -- ResourceLink -----------------------------------------------------

    def create_resource_link(self, link: ResourceLink) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO resource_links
                    (id, resource_id, entity_kind, entity_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    link.id,
                    link.resource_id,
                    link.entity_kind.value,
                    link.entity_id,
                    link.created_at,
                ),
            )
            self._conn.commit()

    def list_resource_links_for_resource(self, resource_id: str) -> list[ResourceLink]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM resource_links WHERE resource_id = ? ORDER BY created_at DESC",
                (resource_id,),
            ).fetchall()
        return [self._row_to_resource_link(row) for row in rows]

    def list_resource_links_for_entity(
        self, entity_kind: LinkedEntityKind, entity_id: str
    ) -> list[ResourceLink]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM resource_links WHERE entity_kind = ? AND entity_id = ? "
                "ORDER BY created_at DESC",
                (entity_kind.value, entity_id),
            ).fetchall()
        return [self._row_to_resource_link(row) for row in rows]

    def delete_resource_link(self, link_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM resource_links WHERE id = ?", (link_id,)
            )
            self._conn.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_resource_link(row: sqlite3.Row) -> ResourceLink:
        return ResourceLink(
            id=row["id"],
            resource_id=row["resource_id"],
            entity_kind=LinkedEntityKind(row["entity_kind"]),
            entity_id=row["entity_id"],
            created_at=row["created_at"],
        )
