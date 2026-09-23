"""
Task abstraction (M9): a local, independent todo-item model - "Rhasspy
Tasks" - deliberately separate from Reminder (app/reminders/base.py). A
Reminder's whole purpose is a scheduled notification (M7); a Task has no
scheduler entry, no trigger, and no notification of its own - just a
plain create/list/complete/cancel lifecycle, the same SQLite-backed
"interface + swappable backend" shape as every other store in this
codebase.

If a user wants "remind me about this task," that's naturally two linked
objects (a Task plus a Reminder), not one model doing both jobs - keeping
the two independent avoids Task inheriting Reminder's scheduling/
notification complexity it doesn't need.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class TaskStatus(Enum):
    """Lifecycle state of a Task.

    TODO - not yet done.
    DONE - the user marked it complete.
    CANCELLED - the user explicitly cancelled it (abandoned, not done).
    """

    TODO = "todo"
    DONE = "done"
    CANCELLED = "cancelled"


#: Terminal states a task can never leave.
TERMINAL_TASK_STATUSES = frozenset({TaskStatus.DONE, TaskStatus.CANCELLED})


class TaskPriority(Enum):
    """A simple, fixed three-level priority - display/filtering only,
    never affects scheduling (Tasks have none)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Task:
    """A single persisted task.

    `due_date` is display-only - an optional ISO 8601 date/timestamp
    string a user or a future UI can show/sort by; unlike
    Reminder.due_at, nothing ever polls it or fires a notification from
    it - Tasks have no background process of their own.
    """

    id: str
    text: str
    status: TaskStatus
    priority: TaskPriority
    due_date: str | None
    created_at: str
    updated_at: str


class TaskStore(ABC):
    """Persistent storage for tasks."""

    @abstractmethod
    def create(
        self,
        text: str,
        priority: TaskPriority = TaskPriority.MEDIUM,
        due_date: str | None = None,
    ) -> Task:
        """Create and persist a new TODO task."""

    @abstractmethod
    def get(self, task_id: str) -> Task | None:
        """Look up a task by ID, or None if it doesn't exist."""

    @abstractmethod
    def list(self, status: TaskStatus | None = None) -> list[Task]:
        """Return tasks, optionally filtered to one status, most
        recently created first."""

    @abstractmethod
    def complete(self, task_id: str) -> bool:
        """Mark a task DONE. Returns False if the ID doesn't exist or is
        already in a terminal state (DONE/CANCELLED)."""

    @abstractmethod
    def cancel(self, task_id: str) -> bool:
        """Mark a task CANCELLED. Same return-value contract as
        `complete`."""
