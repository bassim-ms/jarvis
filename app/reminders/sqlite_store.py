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

Reminder V2 adds, ADDITIVELY, a nullable `schedule` column (the Schedule as
JSON, which carries the reminder's own time zone) and a nullable
`occurrence` column on the execution log. A database created before V2 is
upgraded in place by adding those two columns; every existing row keeps
loading exactly as before (schedule NULL = the original behaviour).
"""

from __future__ import annotations

import json
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
from app.scheduling.schedule import Schedule, ScheduleError


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
            self._add_column_if_missing("reminders", "schedule", "TEXT")
            self._add_column_if_missing("reminder_executions", "occurrence", "TEXT")
            # One occurrence of one reminder is recorded as delivered at most once, whatever calls this store.
            self._conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS ux_reminder_occurrence_delivered "
                "ON reminder_executions (reminder_id, occurrence) WHERE success = 1 AND occurrence IS NOT NULL"
            )
            self._conn.commit()

    def _add_column_if_missing(self, table: str, column: str, declaration: str) -> None:
        existing = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    @staticmethod
    def _row_to_reminder(row: sqlite3.Row) -> Reminder:
        recurrence = None
        if row["recurrence_unit"] is not None:
            recurrence = Recurrence(
                unit=RecurrenceUnit(row["recurrence_unit"]),
                interval=row["recurrence_interval"],
            )
        schedule, schedule_invalid = None, False
        if row["schedule"] is not None:
            try:
                schedule = Schedule.from_dict(json.loads(row["schedule"]))
            except (ValueError, ScheduleError):  # a damaged stored schedule: loadable, listed, never fired
                schedule_invalid = True
        return Reminder(
            id=row["id"],
            text=row["text"],
            due_at=row["due_at"],
            status=ReminderStatus(row["status"]),
            recurrence=recurrence,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            schedule=schedule,
            schedule_invalid=schedule_invalid,
        )

    def create(
        self,
        text: str,
        due_at: str,
        recurrence: Recurrence | None = None,
        schedule: Schedule | None = None,
    ) -> Reminder:
        if schedule is not None and recurrence is not None:
            raise ValueError("a reminder has either a schedule or a legacy recurrence, never both")
        reminder_id = uuid.uuid4().hex
        now = _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO reminders (
                    id, text, due_at, status, recurrence_unit,
                    recurrence_interval, created_at, updated_at, schedule
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    json.dumps(schedule.to_dict(), separators=(",", ":")) if schedule is not None else None,
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
            schedule=schedule,
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
        occurrence: str | None = None,
        expected_due_at: str | None = None,
    ) -> bool:
        with self._lock:
            try:
                if success:
                    guard, guard_args = "", ()
                    if expected_due_at is not None:
                        # compare-and-set: only the poller that still sees this occurrence may advance it
                        guard = " AND due_at = ? AND status = ?"
                        guard_args = (expected_due_at, ReminderStatus.PENDING.value)
                    if next_due_at is not None:
                        cursor = self._conn.execute(
                            "UPDATE reminders SET due_at = ?, status = ?, updated_at = ? WHERE id = ?" + guard,
                            (next_due_at, ReminderStatus.PENDING.value, triggered_at, reminder_id, *guard_args),
                        )
                    else:
                        cursor = self._conn.execute(
                            "UPDATE reminders SET status = ?, updated_at = ? WHERE id = ?" + guard,
                            (ReminderStatus.TRIGGERED.value, triggered_at, reminder_id, *guard_args),
                        )
                    if expected_due_at is not None and cursor.rowcount == 0:
                        self._conn.rollback()
                        return False
                self._conn.execute(
                    "INSERT INTO reminder_executions "
                    "(reminder_id, triggered_at, success, detail, occurrence) VALUES (?, ?, ?, ?, ?)",
                    (reminder_id, triggered_at, int(success), detail, occurrence),
                )
                self._conn.commit()
            except sqlite3.IntegrityError:  # this occurrence was already recorded as delivered
                self._conn.rollback()
                return False
        return True

    def pause(self, reminder_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE reminders SET status = ?, updated_at = ? WHERE id = ? AND status = ?",
                (ReminderStatus.PAUSED.value, _now(), reminder_id, ReminderStatus.PENDING.value),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def resume(self, reminder_id: str, next_due_at: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE reminders SET status = ?, due_at = ?, updated_at = ? WHERE id = ? AND status = ?",
                (ReminderStatus.PENDING.value, next_due_at, _now(), reminder_id, ReminderStatus.PAUSED.value),
            )
            self._conn.commit()
        return cursor.rowcount > 0

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
