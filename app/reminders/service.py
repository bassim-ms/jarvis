"""
ReminderService (Reminder V2): the ONE place a structured reminder request becomes a stored reminder, and the one place a
reminder is paused, resumed or deleted. The reminder tools (create / pause / resume / cancel) call it, so a Planner step, an
explicit tool command and the chat door all get identical behaviour. It builds on app/scheduling/ - it contains no second
scheduling implementation - and on the existing ReminderStore; it adds no scheduler, no thread and no notification path.

What it decides, deterministically:

  * Which time zone a NEW reminder is created in: JARVIS_TIMEZONE, else the system's zone when it can be determined reliably,
    else a clear failure (nothing is created that might fire at the wrong time). The zone is stored with the reminder and
    never changes afterwards.
  * "7 PM tomorrow", "every weekday at 6 PM", "every Monday at 5 PM", "on 5 October at 9 AM", "every 2 hours" -> a Schedule and
    its first occurrence, with Schedule's own DST rules.
  * A bare "at 7" (morning or evening?): with nothing else pinning the day it means the next time the clock reads 7 and the
    reply says exactly which; with an explicit day or a repetition it is refused until AM/PM is given. Never silently guessed
    where a wrong guess would be material.
  * Limits: a bounded number of active reminders, bounded text, a horizon of about two years, and (for NEW repeating
    reminders only) the minimum gap between runs.
  * Resuming moves to the next FUTURE occurrence; whatever came due while paused is never replayed.

Errors are `ReminderRequestError` with a plain, user-facing message. Nothing here runs a tool, makes a Job, calls a Planner
or a model, or writes memory or the world model: a reminder is data that becomes a notification, nothing more.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from app.notifications.text import clean_text
from app.reminders.base import Reminder, ReminderStatus, ReminderStore
from app.reminders.language import ClockSpec, MAX_TEXT_CHARS
from app.reminders.occurrences import effective_schedule, parse_instant
from app.reminders.timefmt import local_phrase
from app.scheduling.builder import (  # noqa: F401 - ask_meridiem / needs_meridiem stay importable from here
    MAX_HORIZON,
    REMINDER_WORDING,
    ScheduleSpec,
    ScheduleSpecError,
    ask_meridiem,
    build_schedule,
    needs_meridiem,
)
from app.scheduling.schedule import DEFAULT_MIN_INTERVAL, Schedule, ScheduleError, next_due
from app.scheduling.timezones import UTC, TimezoneUnavailableError, ZoneChoice, resolve_timezone

MAX_ACTIVE_REMINDERS = 100


class ReminderRequestError(ValueError):
    """A reminder request that cannot be honoured, with a plain message safe to say to the user."""


@dataclass(frozen=True)
class ReminderRequest:
    """What to remind about and WHEN, as fields (never prose). `clock` is a time of day; `day` is "today" | "tonight" |
    "tomorrow" | a weekday name ("friday", "next friday") | "YYYY-MM-DD" | "MM-DD" | "+N" (N days ahead); `repeat` is
    "daily" | "weekdays" | "weekly" (with `weekdays` as ISO numbers); `every` is an elapsed interval ("hours", 2), first due
    after `first_in` (default: one interval from now)."""

    text: str
    clock: ClockSpec | None = None
    day: str | None = None
    repeat: str | None = None
    weekdays: tuple[int, ...] = ()
    every: tuple[str, int] | None = None
    first_in: timedelta | None = None

    @property
    def spec(self) -> ScheduleSpec:
        """The WHEN part of the request, as the shared schedule builder (app/scheduling/builder.py) reads it."""
        return ScheduleSpec(self.clock, self.day, self.repeat, self.weekdays, self.every, self.first_in)


@dataclass(frozen=True)
class Created:
    reminder: Reminder
    schedule: Schedule
    due: datetime  # the first occurrence, aware UTC
    rolled: bool = False  # a bare "at 7pm" that had passed today was set for tomorrow
    duplicate: bool = False  # an identical active reminder already existed, so none was added
    warning: str = ""  # JARVIS_TIMEZONE was set but unusable and the system zone was used instead


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ReminderService:
    def __init__(
        self,
        store: ReminderStore,
        *,
        clock: Callable[[], datetime] | None = None,
        zone_resolver: Callable[[], ZoneChoice] | None = None,
        min_interval: timedelta | None = None,
        max_active: int = MAX_ACTIVE_REMINDERS,
    ):
        self.store = store
        self._clock = clock or _utc_now
        self._zone_resolver = zone_resolver
        self._min_interval = min_interval
        self.max_active = max_active

    # -- the environment ------------------------------------------------------------------------------------------------------

    def now(self) -> datetime:
        return self._clock().astimezone(UTC)

    def zone(self) -> ZoneChoice:
        """The zone a NEW reminder is created in (see the module docstring), or a clear ReminderRequestError."""
        try:
            if self._zone_resolver is not None:
                return self._zone_resolver()
            from app.config import settings
            from app.system_timezone import detect_system_timezone

            return resolve_timezone(settings.timezone, detect_system_timezone())
        except TimezoneUnavailableError as exc:
            raise ReminderRequestError(str(exc)) from None

    def min_interval(self) -> timedelta:
        if self._min_interval is not None:
            return self._min_interval
        try:
            from app.config import settings

            return timedelta(minutes=max(1, int(settings.schedule_min_interval_minutes)))
        except Exception:  # noqa: BLE001 - an unreadable setting means the documented default
            return DEFAULT_MIN_INTERVAL

    def active(self) -> list[Reminder]:
        """Reminders that are still in play (pending or paused), soonest first."""
        found = self.store.list(ReminderStatus.PENDING) + self.store.list(ReminderStatus.PAUSED)
        return sorted(found, key=lambda r: (parse_instant(r.due_at) if _readable(r.due_at) else datetime.max.replace(tzinfo=UTC)))

    # -- creating ---------------------------------------------------------------------------------------------------------------

    def create(self, request: ReminderRequest) -> Created:
        text = clean_text(request.text, 100_000)
        if not text:
            raise ReminderRequestError("Tell me what to remind you about.")
        if len(text) > MAX_TEXT_CHARS:
            raise ReminderRequestError(f"That reminder is too long - please keep it under {MAX_TEXT_CHARS} characters.")
        try:
            choice = self.zone()
        except ReminderRequestError:
            if request.every is None:
                raise
            # An elapsed interval ("every 2 hours") does not depend on a time zone, and never needed one: UTC is exact.
            choice = ZoneChoice("UTC", "system")
        now = self.now()
        schedule, due, rolled = self._schedule_for(request, choice.name, now)
        if now + MAX_HORIZON < due:
            raise ReminderRequestError("That is more than two years away - please pick a nearer date.")

        active = self.active()
        key = schedule.to_dict()
        for existing in active:
            if existing.schedule is not None and existing.text.casefold() == text.casefold() and existing.schedule.to_dict() == key:
                return Created(existing, schedule, parse_instant(existing.due_at), rolled, duplicate=True, warning=choice.warning)
        if len(active) >= self.max_active:
            raise ReminderRequestError(
                f"You already have {len(active)} reminders - please delete some before adding more."
            )
        reminder = self.store.create(text=text, due_at=due.isoformat(), schedule=schedule)
        return Created(reminder, schedule, due, rolled, warning=choice.warning)

    def _schedule_for(self, request: ReminderRequest, tz: str, now: datetime) -> tuple[Schedule, datetime, bool]:
        try:
            return build_schedule(request.spec, tz, now, min_interval=self.min_interval(), wording=REMINDER_WORDING)
        except ScheduleSpecError as exc:
            raise ReminderRequestError(str(exc)) from None
        except TimezoneUnavailableError as exc:
            raise ReminderRequestError(str(exc)) from None
        except ScheduleError as exc:
            raise ReminderRequestError(f"I couldn't set that reminder: {exc}.") from None

    # -- pausing, resuming, deleting --------------------------------------------------------------------------------------------

    def pause(self, reminder_id: str) -> Reminder:
        reminder = self._require(reminder_id)
        if reminder.status is ReminderStatus.PAUSED:
            raise ReminderRequestError(f"'{_shown(reminder)}' is already paused.")
        if reminder.status is not ReminderStatus.PENDING or not self.store.pause(reminder_id):
            raise ReminderRequestError(f"'{_shown(reminder)}' isn't an active reminder, so there is nothing to pause.")
        return reminder

    def resume(self, reminder_id: str) -> tuple[Reminder, datetime]:
        """Make a paused reminder live again at its next FUTURE occurrence. Returns the reminder and that occurrence."""
        reminder = self._require(reminder_id)
        if reminder.status is not ReminderStatus.PAUSED:
            raise ReminderRequestError(f"'{_shown(reminder)}' isn't paused.")
        if reminder.schedule_invalid:
            raise ReminderRequestError(f"I can't read the schedule of '{_shown(reminder)}', so I can't resume it. Please delete it and set it again.")
        now = self.now()
        schedule = effective_schedule(reminder)
        if schedule is not None:
            upcoming = next_due(schedule, now)
        else:
            stored = parse_instant(reminder.due_at)
            upcoming = stored if stored > now else None
        if upcoming is None:
            raise ReminderRequestError(
                f"The time for '{_shown(reminder)}' has already passed, so there is nothing left to resume. "
                "Delete it, or set a new one."
            )
        if not self.store.resume(reminder_id, upcoming.isoformat()):
            raise ReminderRequestError(f"'{_shown(reminder)}' isn't paused.")
        return reminder, upcoming

    def delete(self, reminder_id: str) -> Reminder:
        """Delete (cancel) one reminder. Pending or paused only; a finished one has nothing left to delete."""
        reminder = self._require(reminder_id)
        if reminder.status not in (ReminderStatus.PENDING, ReminderStatus.PAUSED) or not self.store.cancel(reminder_id):
            raise ReminderRequestError(f"'{_shown(reminder)}' isn't an active reminder.")
        return reminder

    def _require(self, reminder_id: str) -> Reminder:
        reminder = self.store.get(reminder_id) if isinstance(reminder_id, str) and reminder_id else None
        if reminder is None:
            raise ReminderRequestError("I couldn't find that reminder.")
        return reminder

    # -- describing -------------------------------------------------------------------------------------------------------------

    def schedule_words(self, reminder: Reminder) -> str:
        """When a reminder happens, in plain words ("every day at 7:00 PM (Asia/Dubai)")."""
        if reminder.schedule is not None:
            return reminder.schedule.describe()
        if reminder.recurrence is not None:
            n, unit = reminder.recurrence.interval, reminder.recurrence.unit.value
            return f"every {n} {unit[:-1] if n == 1 else unit}"
        return f"at {self._phrase(parse_instant(reminder.due_at), None)}"

    def instant_words(self, instant: datetime) -> str:
        """An instant as local words in the zone new reminders use ("Thursday 1 October, 8:10 PM"), or in UTC if that is unknown."""
        return self._phrase(instant, None)

    def next_words(self, reminder: Reminder) -> str:
        return self._phrase(parse_instant(reminder.due_at), reminder.timezone)

    def zone_name_or_none(self) -> str | None:
        """The zone new reminders use, or None when it cannot be determined (never raises)."""
        try:
            return self.zone().name
        except ReminderRequestError:
            return None

    def _phrase(self, instant: datetime, tz: str | None) -> str:
        if tz is None:  # a pre-V2 reminder has no zone of its own: show it in the zone new ones use, if that is known
            tz = self.zone_name_or_none()
        return local_phrase(instant, tz)


def _readable(value: str) -> bool:
    try:
        parse_instant(value)
        return True
    except (ValueError, TypeError):
        return False


def _shown(reminder: Reminder) -> str:
    return clean_text(reminder.text, 60)
