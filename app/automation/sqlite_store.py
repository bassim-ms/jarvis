"""
SQLite-backed implementation of AutomationStore, using Python's stdlib
sqlite3 - the exact same pattern as SQLiteReminderStore
(app/reminders/sqlite_store.py) and SQLiteMemoryStore
(app/memory/sqlite_store.py): one connection held for the store's
lifetime, guarded by a lock since FastAPI/the automation engine can call
into a store from more than one thread, backed by a plain local file so
automations - and their execution history - survive a Rhasspy restart.

`trigger.recurrence`/`trigger.reminder_id` get their own typed columns
(fixed shape, exactly like Reminder's own recurrence columns); `action`
(a ToolCall's tool_name + parameters) and `conditions` (a variable-length,
variable-shape list) are stored as JSON, the same "typed columns for
fixed-shape data, JSON only where the shape genuinely varies" split this
module inherits from its own reminders precedent.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

from app.automation.base import (
    Automation,
    AutomationExecution,
    AutomationStatus,
    AutomationStore,
    Condition,
    ExecutionOutcome,
    TERMINAL_AUTOMATION_STATUSES,
    Trigger,
    TriggerType,
    condition_from_dict,
)
from app.reminders.base import Recurrence, RecurrenceUnit
from app.tools.decision import ToolCall


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteAutomationStore(AutomationStore):
    """AutomationStore backed by a local SQLite database file."""

    def __init__(self, db_path: str = "jarvis_automations.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS automations (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    trigger_recurrence_unit TEXT,
                    trigger_recurrence_interval INTEGER,
                    trigger_reminder_id TEXT,
                    conditions TEXT NOT NULL,
                    action_tool_name TEXT NOT NULL,
                    action_params TEXT NOT NULL,
                    next_run_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS automation_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    automation_id TEXT NOT NULL,
                    executed_at TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    detail TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    @staticmethod
    def _row_to_automation(row: sqlite3.Row) -> Automation:
        recurrence = None
        if row["trigger_recurrence_unit"] is not None:
            recurrence = Recurrence(
                unit=RecurrenceUnit(row["trigger_recurrence_unit"]),
                interval=row["trigger_recurrence_interval"],
            )
        trigger = Trigger(
            type=TriggerType(row["trigger_type"]),
            recurrence=recurrence,
            reminder_id=row["trigger_reminder_id"],
        )
        conditions: list[Condition] = [
            condition_from_dict(d) for d in json.loads(row["conditions"])
        ]
        action = ToolCall(
            tool_name=row["action_tool_name"],
            parameters=json.loads(row["action_params"]),
        )
        return Automation(
            id=row["id"],
            name=row["name"],
            status=AutomationStatus(row["status"]),
            trigger=trigger,
            conditions=conditions,
            action=action,
            next_run_at=row["next_run_at"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def create(
        self,
        name: str,
        trigger: Trigger,
        conditions: list[Condition],
        action: ToolCall,
        next_run_at: str | None = None,
    ) -> Automation:
        automation_id = uuid.uuid4().hex
        now = _now()
        conditions_json = json.dumps([c.to_dict() for c in conditions])
        action_params_json = json.dumps(action.parameters)

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO automations (
                    id, name, status, trigger_type, trigger_recurrence_unit,
                    trigger_recurrence_interval, trigger_reminder_id,
                    conditions, action_tool_name, action_params,
                    next_run_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    automation_id,
                    name,
                    AutomationStatus.ENABLED.value,
                    trigger.type.value,
                    trigger.recurrence.unit.value if trigger.recurrence else None,
                    trigger.recurrence.interval if trigger.recurrence else None,
                    trigger.reminder_id,
                    conditions_json,
                    action.tool_name,
                    action_params_json,
                    next_run_at,
                    now,
                    now,
                ),
            )
            self._conn.commit()

        return Automation(
            id=automation_id,
            name=name,
            status=AutomationStatus.ENABLED,
            trigger=trigger,
            conditions=conditions,
            action=action,
            next_run_at=next_run_at,
            created_at=now,
            updated_at=now,
        )

    def get(self, automation_id: str) -> Automation | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM automations WHERE id = ?", (automation_id,)
            ).fetchone()
        return self._row_to_automation(row) if row is not None else None

    def list(self, status: AutomationStatus | None = None) -> list[Automation]:
        with self._lock:
            if status is None:
                rows = self._conn.execute(
                    "SELECT * FROM automations ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM automations WHERE status = ? "
                    "ORDER BY created_at DESC",
                    (status.value,),
                ).fetchall()
        return [self._row_to_automation(row) for row in rows]

    def record_execution(
        self,
        automation_id: str,
        executed_at: str,
        outcome: ExecutionOutcome,
        detail: str = "",
        next_run_at: str | None = None,
        disable: bool = False,
    ) -> None:
        with self._lock:
            if disable:
                self._conn.execute(
                    "UPDATE automations SET status = ?, updated_at = ? "
                    "WHERE id = ? AND status != ?",
                    (
                        AutomationStatus.DISABLED.value,
                        executed_at,
                        automation_id,
                        AutomationStatus.CANCELLED.value,
                    ),
                )
            elif next_run_at is not None:
                self._conn.execute(
                    "UPDATE automations SET next_run_at = ?, updated_at = ? "
                    "WHERE id = ?",
                    (next_run_at, executed_at, automation_id),
                )
            self._conn.execute(
                "INSERT INTO automation_executions "
                "(automation_id, executed_at, outcome, detail) VALUES (?, ?, ?, ?)",
                (automation_id, executed_at, outcome.value, detail),
            )
            self._conn.commit()

    def get_executions(self, automation_id: str) -> list[AutomationExecution]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM automation_executions WHERE automation_id = ? "
                "ORDER BY id ASC",
                (automation_id,),
            ).fetchall()
        return [
            AutomationExecution(
                automation_id=row["automation_id"],
                executed_at=row["executed_at"],
                outcome=ExecutionOutcome(row["outcome"]),
                detail=row["detail"],
            )
            for row in rows
        ]

    def _set_status(
        self, automation_id: str, status: AutomationStatus, *, guard_terminal: bool
    ) -> bool:
        with self._lock:
            if guard_terminal:
                terminal_values = tuple(s.value for s in TERMINAL_AUTOMATION_STATUSES)
                placeholders = ", ".join("?" for _ in terminal_values)
                cursor = self._conn.execute(
                    f"UPDATE automations SET status = ?, updated_at = ? "
                    f"WHERE id = ? AND status NOT IN ({placeholders})",
                    (status.value, _now(), automation_id, *terminal_values),
                )
            else:
                cursor = self._conn.execute(
                    "UPDATE automations SET status = ?, updated_at = ? WHERE id = ?",
                    (status.value, _now(), automation_id),
                )
            self._conn.commit()
        return cursor.rowcount > 0

    def enable(self, automation_id: str) -> bool:
        return self._set_status(
            automation_id, AutomationStatus.ENABLED, guard_terminal=True
        )

    def disable(self, automation_id: str) -> bool:
        return self._set_status(
            automation_id, AutomationStatus.DISABLED, guard_terminal=True
        )

    def cancel(self, automation_id: str) -> bool:
        return self._set_status(
            automation_id, AutomationStatus.CANCELLED, guard_terminal=True
        )
