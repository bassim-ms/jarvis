"""
Reminder abstraction (M7): persistent tasks/reminders with a due time,
optional recurrence, and a lifecycle a background scheduler can safely
recover after a restart.

The reminder Tools (app/reminders/tools.py) and ReminderScheduler
(app/reminders/scheduler.py) depend only on this interface, never on
SQLite directly - same "Core/Registry never knows the storage engine"
split as MemoryStore (app/memory/base.py) and SearchCache
(app/search/cache.py).

A Reminder models a one-time or recurring "notify the user about this at/
after this time" entry - the only kind of "task" this milestone adds.
Recurrence is deliberately just a fixed interval (every N minutes/hours/
days/weeks) - no cron expressions, no calendar-aware scheduling (last day
of month, business days, timezones beyond UTC) - the same "accepted,
documented trade-off" philosophy as SearchRouter's keyword list or
DeterministicSearchPlanner's fixed suffixes.

Scheduling here is deliberately narrow, and that narrowness is the whole
security boundary M7 adds: a Reminder only ever carries free-text plus a
due time/recurrence - never a tool name or parameters - so persisted
scheduled state can never be used to run something other than a
notification (see ReminderScheduler), no matter what's stored in it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class ReminderStatus(Enum):
    """Lifecycle state of a Reminder.

    PENDING - not yet due, or a recurring reminder that has fired before
      and is waiting for its next occurrence.
    TRIGGERED - a one-time reminder that has already fired once. Kept
      distinct from COMPLETED so "it fired" and "the user acknowledged
      it" are recorded separately. A recurring reminder never enters this
      state; it stays PENDING across occurrences instead (see
      ReminderStore.record_trigger).
    COMPLETED - the user explicitly marked it done; for a recurring
      reminder this stops all future occurrences, not just one.
    CANCELLED - the user explicitly cancelled it, before or after it first
      fired.
    """

    PENDING = "pending"
    TRIGGERED = "triggered"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


#: Terminal states a reminder can never leave - excluded from every
#: "is this due" query and from `complete`/`cancel` once reached.
TERMINAL_STATUSES = frozenset({ReminderStatus.COMPLETED, ReminderStatus.CANCELLED})


class RecurrenceUnit(Enum):
    """Fixed, calendar-naive interval unit a recurring Reminder repeats on."""

    MINUTES = "minutes"
    HOURS = "hours"
    DAYS = "days"
    WEEKS = "weeks"


@dataclass
class Recurrence:
    """How often a recurring Reminder repeats: every `interval` `unit`s.
    `None` on a Reminder itself (not this class) means one-time."""

    unit: RecurrenceUnit
    interval: int = 1


@dataclass
class Reminder:
    """A single persisted task/reminder.

    `due_at` is always the *next* time this reminder should fire - for a
    one-time reminder, the only time; for a recurring one, advanced by
    `recurrence` each time it fires (see ReminderStore.record_trigger).
    `due_at`/`created_at`/`updated_at` are ISO 8601 UTC strings, the same
    convention SQLiteMemoryStore/SQLiteSearchCache already use.
    """

    id: str
    text: str
    due_at: str
    status: ReminderStatus
    recurrence: Recurrence | None
    created_at: str
    updated_at: str


@dataclass
class ReminderExecution:
    """One record of a Reminder actually coming due and being handled -
    the lifecycle/execution log a restart's recovery behavior can be
    verified against, distinct from the Reminder's own current state.
    `success` reflects whether the notification itself was delivered;
    `detail` carries a failure message when it wasn't."""

    reminder_id: str
    triggered_at: str
    success: bool
    detail: str


class ReminderStore(ABC):
    """Persistent storage for reminders and their execution history."""

    @abstractmethod
    def create(
        self, text: str, due_at: str, recurrence: Recurrence | None = None
    ) -> Reminder:
        """Create and persist a new PENDING reminder, returning it."""

    @abstractmethod
    def get(self, reminder_id: str) -> Reminder | None:
        """Look up a reminder by ID, or None if it doesn't exist."""

    @abstractmethod
    def list(self, status: ReminderStatus | None = None) -> list[Reminder]:
        """Return reminders, optionally filtered to one status, most
        recently created first."""

    @abstractmethod
    def list_due(self, now: str) -> list[Reminder]:
        """Return every PENDING reminder whose `due_at` is <= `now` - the
        scheduler's only query. Persistence-backed, so it naturally
        recovers anything that became due while Jarvis wasn't running; no
        separate "missed reminder" recovery logic is needed anywhere."""

    @abstractmethod
    def record_trigger(
        self,
        reminder_id: str,
        triggered_at: str,
        success: bool,
        detail: str = "",
        next_due_at: str | None = None,
    ) -> None:
        """Record that `reminder_id` came due and was handled. Always
        appends a ReminderExecution row, regardless of `success`.

        On a *successful* notification (`success=True`): `next_due_at`
        given advances `due_at` to it and leaves the reminder PENDING (a
        recurring occurrence); omitted, the reminder's status becomes
        TRIGGERED so `list_due` never returns it again (a one-time
        reminder's only occurrence).

        On a *failed* notification (`success=False`): the reminder's
        status/due_at are left untouched (still PENDING, still due), so
        the next scheduler poll retries it - a delivery failure is never
        silently treated as "handled".
        """

    @abstractmethod
    def get_executions(self, reminder_id: str) -> list[ReminderExecution]:
        """Return every recorded execution for `reminder_id`, oldest
        first."""

    @abstractmethod
    def complete(self, reminder_id: str) -> bool:
        """Mark a reminder COMPLETED, stopping any future occurrences.
        Returns False if the ID doesn't exist or is already in a terminal
        state (COMPLETED/CANCELLED)."""

    @abstractmethod
    def cancel(self, reminder_id: str) -> bool:
        """Mark a reminder CANCELLED. Same return-value contract as
        `complete`."""
