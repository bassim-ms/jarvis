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
                    autonomous_permissions TEXT NOT NULL DEFAULT '[]'
                );
                """
            )
            self._conn.commit()
            self._migrate_add_column("conversation_id", "TEXT")
            self._migrate_add_column("autonomous_permissions", "TEXT NOT NULL DEFAULT '[]'")

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
        now = _now()
        job = Job(
            id=str(uuid.uuid4()),
            goal=goal,
            status=JobStatus.PENDING,
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
        with self._lock:
            self._conn.execute(
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
                    json.dumps([]),
                    job.created_at,
                    job.updated_at,
                    job.conversation_id,
                    json.dumps(sorted(job.autonomous_permissions)),
                ),
            )
            self._conn.commit()
        return job

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
        with self._lock:
            self._conn.execute(
                """
                UPDATE jobs SET
                    status = ?, priority = ?, autonomy = ?, plan = ?,
                    context = ?, replan_count = ?, total_step_count = ?,
                    pending_step_ids = ?, confirmation_message = ?,
                    final_result = ?, error = ?, events = ?, updated_at = ?
                WHERE id = ?
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
                    job.id,
                ),
            )
            self._conn.commit()

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
        )
