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
from datetime import datetime, time, timedelta, timezone

from app.notifications.text import clean_text
from app.reminders.base import Recurrence, RecurrenceUnit, Reminder, ReminderStatus, ReminderStore
from app.reminders.language import ClockSpec, MAX_TEXT_CHARS, WEEKDAY_NUMBERS
from app.reminders.occurrences import effective_schedule, parse_instant
from app.reminders.timefmt import clock_text, local_phrase
from app.scheduling.schedule import (
    DEFAULT_MIN_INTERVAL,
    Schedule,
    ScheduleError,
    interval_delta,
    next_due,
    resolve_once,
)
from app.scheduling.timezones import UTC, TimezoneUnavailableError, ZoneChoice, load_zone, local_to_utc, resolve_timezone

MAX_ACTIVE_REMINDERS = 100
MAX_HORIZON = timedelta(days=800)
MAX_DAY_OFFSET = 365
_UNITS = {unit.value: unit for unit in RecurrenceUnit}


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


def needs_meridiem(clock: ClockSpec | None, day: str | None, repeat: str | None, every: tuple[str, int] | None = None) -> bool:
    """Whether the request can only be honoured once the user says AM or PM: a bare 12-hour time together with an explicit day
    or a repetition. (A bare time alone is read as the next upcoming one - see the module docstring.)"""
    return clock is not None and clock.ambiguous and (day is not None or repeat is not None)


def ask_meridiem(clock: ClockSpec) -> str:
    first, second = clock.candidates[0], clock.candidates[1]
    return (
        f"Do you mean {clock_text(first)} or {clock_text(second)}? Say AM or PM, or give the time as 24-hour, like "
        f"{second.hour:02d}:{second.minute:02d}."
    )


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
            if request.every is not None:
                return self._interval(request, tz, now)
            if request.repeat is not None:
                return self._repeating(request, tz, now)
            return self._one_time(request, tz, now)
        except TimezoneUnavailableError as exc:
            raise ReminderRequestError(str(exc)) from None
        except ScheduleError as exc:
            raise ReminderRequestError(f"I couldn't set that reminder: {exc}.") from None

    def _interval(self, request: ReminderRequest, tz: str, now: datetime) -> tuple[Schedule, datetime, bool]:
        if request.clock is not None or request.day is not None or request.repeat is not None:
            raise ReminderRequestError("A reminder that repeats every so often can't also have a fixed time or day.")
        unit, count = request.every
        if unit not in _UNITS:
            raise ReminderRequestError(f"I can repeat a reminder every N minutes, hours, days or weeks, not '{unit}'.")
        recurrence = Recurrence(_UNITS[unit], int(count))
        anchor = now.replace(microsecond=0) + (request.first_in if request.first_in is not None else interval_delta(recurrence))
        schedule = Schedule.every(tz, recurrence, anchor, min_interval=self.min_interval())
        return schedule, anchor, False

    def _repeating(self, request: ReminderRequest, tz: str, now: datetime) -> tuple[Schedule, datetime, bool]:
        if request.day is not None:
            raise ReminderRequestError("A repeating reminder can't also be for one particular day.")
        if request.clock is None:
            raise ReminderRequestError("What time should it repeat at? For example: every day at 7 PM.")
        if request.clock.ambiguous:
            raise ReminderRequestError(ask_meridiem(request.clock))
        at = request.clock.exact
        if request.repeat == "daily":
            schedule = Schedule.daily(tz, at)
        elif request.repeat == "weekdays":
            schedule = Schedule.weekdays(tz, at)
        elif request.repeat == "weekly":
            days = tuple(sorted({d for d in request.weekdays if isinstance(d, int) and 1 <= d <= 7}))
            if not days:
                raise ReminderRequestError("Which day of the week should it repeat on? For example: every Monday at 5 PM.")
            schedule = Schedule.weekly(tz, at, days)
        else:
            raise ReminderRequestError("I can repeat a reminder daily, on weekdays, or on chosen days of the week.")
        return schedule, next_due(schedule, now), False

    def _one_time(self, request: ReminderRequest, tz: str, now: datetime) -> tuple[Schedule, datetime, bool]:
        if request.clock is None:
            raise ReminderRequestError("What time should I remind you? For example: tomorrow at 7 PM.")
        if request.day is None:
            # No day given: the next time the clock reads that (morning or evening, whichever comes first), never in the past.
            options = [resolve_once(candidate, now, tz) for candidate in request.clock.candidates]
            schedule, rolled = min(options, key=lambda option: next_due(option[0], now))
            return schedule, next_due(schedule, now), rolled
        if request.clock.ambiguous:
            raise ReminderRequestError(ask_meridiem(request.clock))
        at = request.clock.exact
        zone = load_zone(tz)
        local_today = now.astimezone(zone).date()
        day = self._resolve_day(request.day, at, now, local_today, zone)
        schedule = Schedule.once(tz, datetime.combine(day, at))
        due = next_due(schedule, now)
        if due is None:
            when = "today" if request.day in ("today", "tonight") else "that date"
            raise ReminderRequestError(f"{clock_text(at)} has already passed {when}. Tell me a time that is still ahead.")
        return schedule, due, False

    @staticmethod
    def _resolve_day(day: str, at: time, now: datetime, today, zone):
        value = day.strip().lower()
        if value in ("today", "tonight"):
            return today
        if value == "tomorrow":
            return today + timedelta(days=1)
        if value.startswith("+") and value[1:].isdigit():
            offset = int(value[1:])
            if not 1 <= offset <= MAX_DAY_OFFSET:
                raise ReminderRequestError("I can set reminders up to a year ahead this way - give me a date instead.")
            return today + timedelta(days=offset)
        strictly_after = value.startswith("next ")
        name = value[5:] if strictly_after else value
        if name in WEEKDAY_NUMBERS:
            target = WEEKDAY_NUMBERS[name]
            for ahead in range(0 if not strictly_after else 1, 8):
                candidate = today + timedelta(days=ahead)
                if candidate.isoweekday() == target and local_to_utc(datetime.combine(candidate, at), zone) > now:
                    return candidate
            raise ReminderRequestError("I couldn't work out which day that is.")
        parts = value.split("-")
        try:
            if len(parts) == 3:
                return datetime(int(parts[0]), int(parts[1]), int(parts[2])).date()
            if len(parts) == 2:
                month, dom = int(parts[0]), int(parts[1])
                for year in (today.year, today.year + 1):
                    try:
                        candidate = datetime(year, month, dom).date()
                    except ValueError:
                        continue
                    if local_to_utc(datetime.combine(candidate, at), zone) > now:
                        return candidate
                raise ValueError
        except ValueError:
            pass
        raise ReminderRequestError(f"I couldn't understand the day '{day}'.")

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
