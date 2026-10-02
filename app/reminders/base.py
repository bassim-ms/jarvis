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
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation only: app.scheduling.schedule itself imports Recurrence from this module
    from app.scheduling.schedule import Schedule


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
    CANCELLED - the user explicitly cancelled (deleted) it, before or after
      it first fired.
    PAUSED - the user paused it. Not terminal: `ReminderStore.resume` makes
      it PENDING again. A paused reminder is never due (`list_due` only
      returns PENDING ones), and nothing that comes due while it is paused
      is ever replayed afterwards - resuming jumps to the next FUTURE
      occurrence.
    """

    PENDING = "pending"
    TRIGGERED = "triggered"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


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


class RecurrenceError(ValueError):
    """Raised by `parse_recurrence` for an unknown unit or a non-positive
    interval - never silently coerced or dropped."""


_VALID_RECURRENCE_UNIT_VALUES = {member.value for member in RecurrenceUnit}


def parse_recurrence(unit: str | None, interval: float | int = 1) -> Recurrence | None:
    """Validate and construct a Recurrence from raw tool parameters, or
    return None if `unit` is None (one-time - the caller never supplied a
    recurrence at all). Shared by CreateReminderTool
    (app/reminders/tools.py) and CreateAutomationTool
    (app/automation/tools.py) so both validate a recurring schedule
    identically, rather than each re-implementing the same checks.

    Raises RecurrenceError (never silently coerces) for an unrecognized
    `unit`, or for an `interval` that truncates to <= 0 - a non-positive
    interval would never advance a due time into the future (see
    `advance_by_recurrence` below), so whatever it's attached to would
    become due again on every single poll, forever.
    """
    if unit is None:
        return None
    if unit not in _VALID_RECURRENCE_UNIT_VALUES:
        raise RecurrenceError(
            f"Unknown recurrence_unit: '{unit}'. Must be one of: "
            f"{sorted(_VALID_RECURRENCE_UNIT_VALUES)}"
        )
    parsed_interval = int(interval)
    if parsed_interval <= 0:
        raise RecurrenceError("recurrence_interval must be a positive integer.")
    return Recurrence(unit=RecurrenceUnit(unit), interval=parsed_interval)


_RECURRENCE_DELTA_KWARGS = {
    RecurrenceUnit.MINUTES: "minutes",
    RecurrenceUnit.HOURS: "hours",
    RecurrenceUnit.DAYS: "days",
    RecurrenceUnit.WEEKS: "weeks",
}


def advance_by_recurrence(timestamp: str, recurrence: Recurrence) -> str:
    """Advance an ISO 8601 UTC `timestamp` by one recurrence interval,
    relative to `timestamp` itself - not "now" - so a delay in
    *processing* a recurring occurrence (Jarvis was offline, or a poll
    was late) doesn't also shift every future occurrence later; it just
    catches up one tick per poll until the timestamp is back in the
    future. Shared by ReminderScheduler (app/reminders/scheduler.py) and
    AutomationEngine (app/automation/engine.py) - both a recurring
    Reminder's `due_at` and a recurring TIME-triggered Automation's
    `next_run_at` advance exactly the same way.

    Can raise OverflowError for an absurd interval (e.g. billions of
    weeks) - callers isolate that themselves (see ReminderScheduler.tick/
    AutomationEngine.tick's per-item failure isolation) rather than this
    function silently clamping it.
    """
    kwarg = _RECURRENCE_DELTA_KWARGS[recurrence.unit]
    delta = timedelta(**{kwarg: recurrence.interval})
    return (datetime.fromisoformat(timestamp) + delta).isoformat()


@dataclass
class Reminder:
    """A single persisted task/reminder.

    `due_at` is always the *next* time this reminder should fire - for a
    one-time reminder, the only time; for a recurring one, advanced by
    `recurrence` each time it fires (see ReminderStore.record_trigger).
    `due_at`/`created_at`/`updated_at` are ISO 8601 UTC strings, the same
    convention SQLiteMemoryStore/SQLiteSearchCache already use.

    Reminder V2: `schedule` (app/scheduling/schedule.py) is the optional,
    richer description of WHEN - local wall-clock time in the reminder's
    own, persisted time zone (a one-time date and time, every day, every
    weekday, chosen weekdays, or an elapsed interval). When it is present
    it is the source of truth for every occurrence after the first and
    `recurrence` stays None; `due_at` remains the next occurrence as a UTC
    instant, exactly as before. A reminder created before V2 has no
    schedule and behaves as it always did. `schedule_invalid` is True when
    a stored schedule could not be read back: such a reminder is listed
    but never fired, rather than guessed at.
    """

    id: str
    text: str
    due_at: str
    status: ReminderStatus
    recurrence: Recurrence | None
    created_at: str
    updated_at: str
    schedule: "Schedule | None" = None
    schedule_invalid: bool = False

    @property
    def timezone(self) -> str | None:
        """The IANA zone this reminder's schedule was created in (None for a pre-V2 reminder)."""
        return self.schedule.tz if self.schedule is not None else None


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
        self,
        text: str,
        due_at: str,
        recurrence: Recurrence | None = None,
        schedule: "Schedule | None" = None,
    ) -> Reminder:
        """Create and persist a new PENDING reminder, returning it. `schedule` (Reminder V2) is persisted with
        the reminder, including the time zone it was created in; `due_at` is its first occurrence."""

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
        occurrence: str | None = None,
        expected_due_at: str | None = None,
    ) -> bool:
        """Record that `reminder_id` came due and was handled. Appends a
        ReminderExecution row regardless of `success` (unless one of the
        duplicate guards below refuses it).

        Duplicate prevention (Reminder V2). `expected_due_at` makes the
        update a compare-and-set: it applies only while the reminder's
        `due_at` still IS that value, so two pollers that both saw the same
        occurrence can never both advance (and so never both announce) it.
        `occurrence` (app/scheduling/runs.py `occurrence_key`) names the
        occurrence being handled; the store refuses to record the SAME
        occurrence as successfully delivered twice. In either case nothing
        is changed and False is returned. Returns True when recorded.

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
    def pause(self, reminder_id: str) -> bool:
        """PENDING -> PAUSED. False if the reminder doesn't exist or isn't PENDING."""

    @abstractmethod
    def resume(self, reminder_id: str, next_due_at: str) -> bool:
        """PAUSED -> PENDING, with `due_at` moved to `next_due_at` (the caller computes the next FUTURE
        occurrence, so nothing missed while paused is replayed). False if it isn't PAUSED."""

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
