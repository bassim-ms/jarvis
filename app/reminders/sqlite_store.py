"""
SQLite-backed implementation of ReminderStore, using Python's stdlib
sqlite3 - same pattern as SQLiteMemoryStore (app/memory/sqlite_store.py)
and SQLiteSearchCache (app/search/sqlite_cache.py): one connection held for
the store's lifetime (needed so ":memory:" databases work, and so
file-based ones persist), guarded by a lock since FastAPI/the reminder
scheduler can call into a store from more than one thread.

Backed by a plain local file, so reminders - and whether they've already
fired - survive a Jarvis restart with no special recovery code: a new
SQLiteReminderStore pointed at the same db_path just sees the same rows
`list_due` already knows how to query.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone

from app.reminders.base import (
    Recurrence,
    RecurrenceUnit,
    Reminder,
    ReminderExecution,
    ReminderStatus,
    ReminderStore,
    TERMINAL_STATUSES,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteReminderStore(ReminderStore):
    """ReminderStore backed by a local SQLite database file."""

    def __init__(self, db_path: str = "jarvis_reminders.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    due_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    recurrence_unit TEXT,
                    recurrence_interval INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS reminder_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reminder_id TEXT NOT NULL,
                    triggered_at TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    detail TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    @staticmethod
    def _row_to_reminder(row: sqlite3.Row) -> Reminder:
        recurrence = None
        if row["recurrence_unit"] is not None:
            recurrence = Recurrence(
                unit=RecurrenceUnit(row["recurrence_unit"]),
                interval=row["recurrence_interval"],
            )
        return Reminder(
            id=row["id"],
            text=row["text"],
            due_at=row["due_at"],
            status=ReminderStatus(row["status"]),
            recurrence=recurrence,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def create(
        self, text: str, due_at: str, recurrence: Recurrence | None = None
    ) -> Reminder:
        reminder_id = uuid.uuid4().hex
        now = _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO reminders (
                    id, text, due_at, status, recurrence_unit,
                    recurrence_interval, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    reminder_id,
                    text,
                    due_at,
                    ReminderStatus.PENDING.value,
                    recurrence.unit.value if recurrence else None,
                    recurrence.interval if recurrence else None,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        return Reminder(
            id=reminder_id,
            text=text,
            due_at=due_at,
            status=ReminderStatus.PENDING,
            recurrence=recurrence,
            created_at=now,
            updated_at=now,
        )

    def get(self, reminder_id: str) -> Reminder | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM reminders WHERE id = ?", (reminder_id,)
            ).fetchone()
        return self._row_to_reminder(row) if row is not None else None

    def list(self, status: ReminderStatus | None = None) -> list[Reminder]:
        with self._lock:
            if status is None:
                rows = self._conn.execute(
                    "SELECT * FROM reminders ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM reminders WHERE status = ? "
                    "ORDER BY created_at DESC",
                    (status.value,),
                ).fetchall()
        return [self._row_to_reminder(row) for row in rows]

    def list_due(self, now: str) -> list[Reminder]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM reminders WHERE status = ? AND due_at <= ? "
                "ORDER BY due_at ASC",
                (ReminderStatus.PENDING.value, now),
            ).fetchall()
        return [self._row_to_reminder(row) for row in rows]

    def record_trigger(
        self,
        reminder_id: str,
        triggered_at: str,
        success: bool,
        detail: str = "",
        next_due_at: str | None = None,
    ) -> None:
        with self._lock:
            if success:
                if next_due_at is not None:
                    self._conn.execute(
                        "UPDATE reminders SET due_at = ?, status = ?, "
                        "updated_at = ? WHERE id = ?",
                        (
                            next_due_at,
                            ReminderStatus.PENDING.value,
                            triggered_at,
                            reminder_id,
                        ),
                    )
                else:
                    self._conn.execute(
                        "UPDATE reminders SET status = ?, updated_at = ? "
                        "WHERE id = ?",
                        (ReminderStatus.TRIGGERED.value, triggered_at, reminder_id),
                    )
            self._conn.execute(
                "INSERT INTO reminder_executions "
                "(reminder_id, triggered_at, success, detail) VALUES (?, ?, ?, ?)",
                (reminder_id, triggered_at, int(success), detail),
            )
            self._conn.commit()

    def get_executions(self, reminder_id: str) -> list[ReminderExecution]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM reminder_executions WHERE reminder_id = ? "
                "ORDER BY id ASC",
                (reminder_id,),
            ).fetchall()
        return [
            ReminderExecution(
                reminder_id=row["reminder_id"],
                triggered_at=row["triggered_at"],
                success=bool(row["success"]),
                detail=row["detail"],
            )
            for row in rows
        ]

    def _set_terminal_status(self, reminder_id: str, status: ReminderStatus) -> bool:
        terminal_values = tuple(s.value for s in TERMINAL_STATUSES)
        placeholders = ", ".join("?" for _ in terminal_values)
        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE reminders SET status = ?, updated_at = ? "
                f"WHERE id = ? AND status NOT IN ({placeholders})",
                (status.value, _now(), reminder_id, *terminal_values),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def complete(self, reminder_id: str) -> bool:
        return self._set_terminal_status(reminder_id, ReminderStatus.COMPLETED)

    def cancel(self, reminder_id: str) -> bool:
        return self._set_terminal_status(reminder_id, ReminderStatus.CANCELLED)
