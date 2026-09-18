"""
SQLite-backed implementation of TaskStore, using Python's stdlib sqlite3 -
the exact same pattern as SQLiteReminderStore/SQLiteMemoryStore: one
connection held for the store's lifetime, guarded by a lock since
FastAPI can call into a store from more than one thread, backed by a
plain local file so tasks survive a Jarvis restart with no special
recovery code.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone

from app.tasks.base import (
    TERMINAL_TASK_STATUSES,
    Task,
    TaskPriority,
    TaskStatus,
    TaskStore,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteTaskStore(TaskStore):
    """TaskStore backed by a local SQLite database file."""

    def __init__(self, db_path: str = "jarvis_tasks.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    due_date TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    @staticmethod
    def _row_to_task(row: sqlite3.Row) -> Task:
        return Task(
            id=row["id"],
            text=row["text"],
            status=TaskStatus(row["status"]),
            priority=TaskPriority(row["priority"]),
            due_date=row["due_date"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def create(
        self,
        text: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        due_date: str | None = None,
    ) -> Task:
        task_id = uuid.uuid4().hex
        now = _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO tasks
                    (id, text, status, priority, due_date, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    text,
                    TaskStatus.TODO.value,
                    priority.value,
                    due_date,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        return Task(
            id=task_id,
            text=text,
            status=TaskStatus.TODO,
            priority=priority,
            due_date=due_date,
            created_at=now,
            updated_at=now,
        )

    def get(self, task_id: str) -> Task | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return self._row_to_task(row) if row is not None else None

    def list(self, status: TaskStatus | None = None) -> list[Task]:
        with self._lock:
            if status is None:
                rows = self._conn.execute(
                    "SELECT * FROM tasks ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC",
                    (status.value,),
                ).fetchall()
        return [self._row_to_task(row) for row in rows]

    def _set_terminal_status(self, task_id: str, status: TaskStatus) -> bool:
        terminal_values = tuple(s.value for s in TERMINAL_TASK_STATUSES)
        placeholders = ", ".join("?" for _ in terminal_values)
        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE tasks SET status = ?, updated_at = ? "
                f"WHERE id = ? AND status NOT IN ({placeholders})",
                (status.value, _now(), task_id, *terminal_values),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def complete(self, task_id: str) -> bool:
        return self._set_terminal_status(task_id, TaskStatus.DONE)

    def cancel(self, task_id: str) -> bool:
        return self._set_terminal_status(task_id, TaskStatus.CANCELLED)
