"""
SQLite-backed implementation of JobStore, using Python's stdlib sqlite3 -
the exact same pattern as SQLiteAutomationStore/SQLiteReminderStore/
SQLiteTaskStore: one connection held for the store's lifetime, guarded by
a lock since FastAPI/JobEngine can call into a store from more than one
thread, backed by a plain local file so a Job's plan and progress survive
a Rhasspy restart with no special recovery code (see
app/orchestration/engine.py:JobEngine._recover_interrupted_steps for the
one piece of state - a step caught mid-execution - that does need
explicit, deliberate handling on top of that).

`goal`/`status`/`priority`/`autonomy`/timestamps get their own typed
columns; `plan` (a list of PlanSteps, each with its own retry policy and
result), `context`, `pending_step_ids`, and `events` are stored as JSON -
the same "typed columns for fixed-shape data, JSON only where the shape
genuinely varies" split app/automation/sqlite_store.py already uses for
`conditions`/`action`.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from app.orchestration.base import (
    AutonomyLevel,
    Job,
    JobEvent,
    JobPriority,
    JobStatus,
    JobStore,
    Plan,
    PlanStep,
    RetryPolicy,
    StaleJobError,
    StepStatus,
    parse_autonomous_permissions,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _step_to_dict(step: PlanStep) -> dict[str, Any]:
    return {
        "id": step.id,
        "tool_name": step.tool_name,
        "arguments": step.arguments,
        "depends_on": step.depends_on,
        "permission": step.permission,
        "retry_policy": {
            "max_attempts": step.retry_policy.max_attempts,
            "delay_seconds": step.retry_policy.delay_seconds,
        },
        "status": step.status.value,
        "attempts": step.attempts,
        "confirmed": step.confirmed,
        "result": step.result,
        "error": step.error,
        "expectation": step.expectation,
        "evaluation": step.evaluation,
    }


def _step_from_dict(data: dict[str, Any]) -> PlanStep:
    retry_data = data.get("retry_policy") or {}
    return PlanStep(
        id=data["id"],
        tool_name=data["tool_name"],
        arguments=data.get("arguments") or {},
        depends_on=list(data.get("depends_on") or []),
        permission=data.get("permission"),
        retry_policy=RetryPolicy(
            max_attempts=retry_data.get("max_attempts", 1),
            delay_seconds=retry_data.get("delay_seconds", 0.0),
        ),
        status=StepStatus(data.get("status", StepStatus.PENDING.value)),
        attempts=data.get("attempts", 0),
        confirmed=data.get("confirmed", False),
        result=data.get("result"),
        error=data.get("error"),
        expectation=data.get("expectation"),
        evaluation=data.get("evaluation"),
    )


def _plan_to_json(plan: Plan | None) -> str | None:
    if plan is None:
        return None
    return json.dumps({"steps": [_step_to_dict(s) for s in plan.steps]})


def _plan_from_json(raw: str | None) -> Plan | None:
    if raw is None:
        return None
    data = json.loads(raw)
    return Plan(steps=[_step_from_dict(s) for s in data.get("steps", [])])


def _event_to_dict(event: JobEvent) -> dict[str, Any]:
    return {
        "timestamp": event.timestamp,
        "kind": event.kind,
        "detail": event.detail,
        "step_id": event.step_id,
    }


def _event_from_dict(data: dict[str, Any]) -> JobEvent:
    return JobEvent(
        timestamp=data["timestamp"],
        kind=data["kind"],
        detail=data.get("detail", ""),
        step_id=data.get("step_id"),
    )


class SQLiteJobStore(JobStore):
    """JobStore backed by a local SQLite database file."""

    def __init__(self, db_path: str = "jarvis_orchestration.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    @property
    def connection(self) -> sqlite3.Connection:
        """The store's one connection - shared with co-located stores in the
        same database file (SQLiteInteractionStore) so a later stage can
        change a Job and an interaction in a single transaction."""
        return self._conn

    @property
    def lock(self) -> threading.Lock:
        """The lock guarding `connection`; any store using the connection
        must hold it for the duration of each operation."""
        return self._lock

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    autonomy TEXT NOT NULL,
                    plan TEXT,
                    context TEXT NOT NULL,
                    replan_count INTEGER NOT NULL,
                    total_step_count INTEGER NOT NULL,
                    pending_step_ids TEXT NOT NULL,
                    confirmation_message TEXT,
                    final_result TEXT,
                    error TEXT,
                    events TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    conversation_id TEXT,
                    autonomous_permissions TEXT NOT NULL DEFAULT '[]',
                    version INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            self._conn.commit()
            self._migrate_add_column("conversation_id", "TEXT")
            self._migrate_add_column("autonomous_permissions", "TEXT NOT NULL DEFAULT '[]'")
            self._migrate_add_column("version", "INTEGER NOT NULL DEFAULT 0")

    def _migrate_add_column(self, name: str, ddl_type: str) -> None:
        """Add a column to an already-existing `jobs` table from before it
        existed - `CREATE TABLE IF NOT EXISTS` above only creates the full
        schema for a brand-new database file; an on-disk file from before
        `conversation_id`/`autonomous_permissions` existed needs this to
        pick them up. No other store in this project has needed a
        migration yet (each shipped with its final schema from its own
        first milestone) - this is deliberately the minimal form: try to
        add the column, silently accept "duplicate column name" (it's
        already there), let anything else surface normally."""
        try:
            self._conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {ddl_type}")
            self._conn.commit()
        except sqlite3.OperationalError as exc:
            if "duplicate column name" not in str(exc).lower():
                raise

    def create(
        self,
        goal: str,
        priority: JobPriority = JobPriority.MEDIUM,
        autonomy: AutonomyLevel = AutonomyLevel.ASSISTED,
        context: dict[str, Any] | None = None,
        conversation_id: str | None = None,
        autonomous_permissions: frozenset[str] | None = None,
    ) -> Job:
        job = self.build_job(
            goal, priority, autonomy, context, conversation_id, autonomous_permissions
        )
        with self._lock:
            self.insert_job_locked(self._conn, job)
            self._conn.commit()
        return job

    # -- lock-free construction/insert, for creating a Job together with another
    # row in ONE transaction (see SQLiteInteractionStore.create_job_awaiting_input)

    @staticmethod
    def build_job(
        goal: str,
        priority: JobPriority = JobPriority.MEDIUM,
        autonomy: AutonomyLevel = AutonomyLevel.ASSISTED,
        context: dict[str, Any] | None = None,
        conversation_id: str | None = None,
        autonomous_permissions: frozenset[str] | None = None,
        status: JobStatus = JobStatus.PENDING,
    ) -> Job:
        """A new in-memory Job (nothing persisted). `create` always passes the
        default PENDING; only an operation that inserts the Job in the same
        transaction as the row that makes it wait may pass another status."""
        now = _now()
        return Job(
            id=str(uuid.uuid4()),
            goal=goal,
            status=status,
            priority=priority,
            autonomy=autonomy,
            created_at=now,
            updated_at=now,
            context=dict(context or {}),
            conversation_id=conversation_id,
            autonomous_permissions=parse_autonomous_permissions(
                list(autonomous_permissions) if autonomous_permissions is not None else None
            ),
        )

    @staticmethod
    def insert_job_locked(conn: sqlite3.Connection, job: Job) -> None:
        """INSERT a new Job row WITHOUT committing. The caller must hold the
        store's lock and commit on success."""
        conn.execute(
            """
            INSERT INTO jobs (
                id, goal, status, priority, autonomy, plan, context,
                replan_count, total_step_count, pending_step_ids,
                confirmation_message, final_result, error, events,
                created_at, updated_at, conversation_id, autonomous_permissions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.id,
                job.goal,
                job.status.value,
                job.priority.value,
                job.autonomy.value,
                _plan_to_json(job.plan),
                json.dumps(job.context),
                job.replan_count,
                job.total_step_count,
                json.dumps(job.pending_step_ids),
                job.confirmation_message,
                job.final_result,
                job.error,
                json.dumps([_event_to_dict(e) for e in job.events]),
                job.created_at,
                job.updated_at,
                job.conversation_id,
                json.dumps(sorted(job.autonomous_permissions)),
            ),
        )

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return self._row_to_job(row) if row else None

    def list(self, status: JobStatus | None = None) -> list[Job]:
        with self._lock:
            if status is None:
                rows = self._conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC",
                    (status.value,),
                ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def save(self, job: Job) -> None:
        """Compare-and-set: one atomic UPDATE guarded by `version = ?`.
        Zero rows changed means the row is gone or someone saved it since
        the caller read it - nothing was written, StaleJobError is raised
        and `job.version` is left exactly as it was."""
        with self._lock:
            self.write_job_locked(self._conn, job)
            self._conn.commit()
            job.version += 1

    # -- lock-free building blocks -------------------------------------------------------
    #
    # For an operation that must change a Job AND another table in ONE
    # transaction (see SQLiteInteractionStore.transition_with_job). The plain
    # `threading.Lock` is not re-entrant, so such an operation takes
    # `self.lock` ONCE, itself, and calls these directly on `self.connection`
    # instead of going through `get`/`save` (which would try to take the lock
    # again and deadlock). Neither helper takes the lock, commits, or bumps
    # `job.version` - the caller does those, after the whole transaction has
    # succeeded.

    @staticmethod
    def load_job_locked(conn: sqlite3.Connection, job_id: str) -> Job | None:
        """Read a Job on `conn`. The caller must hold the store's lock."""
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return SQLiteJobStore._row_to_job(row) if row else None

    @staticmethod
    def write_job_locked(conn: sqlite3.Connection, job: Job) -> None:
        """The guarded (`version = ?`) Job UPDATE, WITHOUT committing. If it
        matches no row the whole open transaction is rolled back and
        StaleJobError raised, so anything written earlier in the same
        transaction is undone too. The caller must hold the store's lock,
        commit on success, and advance `job.version` after the commit."""
        cursor = conn.execute(
            """
            UPDATE jobs SET
                status = ?, priority = ?, autonomy = ?, plan = ?,
                context = ?, replan_count = ?, total_step_count = ?,
                pending_step_ids = ?, confirmation_message = ?,
                final_result = ?, error = ?, events = ?, updated_at = ?,
                version = ?
            WHERE id = ? AND version = ?
            """,
            (
                job.status.value,
                job.priority.value,
                job.autonomy.value,
                _plan_to_json(job.plan),
                json.dumps(job.context),
                job.replan_count,
                job.total_step_count,
                json.dumps(job.pending_step_ids),
                job.confirmation_message,
                job.final_result,
                job.error,
                json.dumps([_event_to_dict(e) for e in job.events]),
                job.updated_at,
                job.version + 1,
                job.id,
                job.version,
            ),
        )
        if cursor.rowcount == 0:
            conn.rollback()
            current = conn.execute("SELECT version FROM jobs WHERE id = ?", (job.id,)).fetchone()
            if current is None:
                raise StaleJobError(f"Job '{job.id}' no longer exists; nothing was saved.")
            raise StaleJobError(
                f"Job '{job.id}' changed since it was read (read version "
                f"{job.version}, current version {current['version']}); nothing was saved."
            )

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        return Job(
            id=row["id"],
            goal=row["goal"],
            status=JobStatus(row["status"]),
            priority=JobPriority(row["priority"]),
            autonomy=AutonomyLevel(row["autonomy"]),
            plan=_plan_from_json(row["plan"]),
            context=json.loads(row["context"]),
            replan_count=row["replan_count"],
            total_step_count=row["total_step_count"],
            pending_step_ids=json.loads(row["pending_step_ids"]),
            confirmation_message=row["confirmation_message"],
            final_result=row["final_result"],
            error=row["error"],
            events=[_event_from_dict(e) for e in json.loads(row["events"])],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            conversation_id=row["conversation_id"],
            autonomous_permissions=frozenset(
                json.loads(row["autonomous_permissions"] or "[]")
            ),
            version=row["version"],
        )
