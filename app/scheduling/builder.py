"""
Building a NEW Schedule from a structured request - the rules, shared by everything that creates a schedule.

`build_schedule(spec, tz, now, ...)` turns FIELDS (a time of day, a day, a repetition, an elapsed interval - never prose) into a
`Schedule` and its first occurrence. It is the logic Reminder V2 introduced, moved here unchanged so that a Scheduled Job is created
by exactly the same rules as a reminder (same DST handling, same "never guess a morning/evening", same day words) instead of a second
copy of them:

  * "7 PM tomorrow", "every weekday at 6 PM", "every Monday at 5 PM", "on 5 October at 9 AM", "every 2 hours" -> a Schedule and its
    first occurrence, with Schedule's own DST rules.
  * A bare "at 7" (morning or evening?): with nothing else pinning the day it means the next time the clock reads 7, and the caller
    is told it rolled; with an explicit day or a repetition it is refused until AM/PM is given.
  * Day words: today / tonight / tomorrow / a weekday ("friday", "next friday" = strictly the next one) / YYYY-MM-DD / MM-DD / +N.

PURE: no clock (`now` is an argument), no I/O, no storage, no model. The only thing that differs between callers is the NOUN in a
refusal message ("reminder", later "scheduled job"), which is the `Wording` argument; the default reproduces Reminder V2's messages
exactly. Limits that belong to a caller (how many exist, how long the text may be, the overall horizon it enforces) stay with the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta

from app.reminders.base import Recurrence, RecurrenceUnit
from app.reminders.language import ClockSpec, WEEKDAY_NUMBERS
from app.reminders.timefmt import clock_text
from app.scheduling.schedule import Schedule, interval_delta, next_due, resolve_once
from app.scheduling.timezones import load_zone, local_to_utc

#: How far ahead a one-time schedule may be set (about two years). A limit on WHEN, not on how long something stays active.
MAX_HORIZON = timedelta(days=800)
#: The farthest "+N" days ahead the day words accept.
MAX_DAY_OFFSET = 365
_UNITS = {unit.value: unit for unit in RecurrenceUnit}


class ScheduleSpecError(ValueError):
    """A schedule request that cannot be honoured, with a plain message safe to say to the user."""


@dataclass(frozen=True)
class Wording:
    """The only caller-specific words in a refusal: what is being scheduled, and the question asked when no time was given."""

    subject: str = "reminder"
    when_question: str = "What time should I remind you? For example: tomorrow at 7 PM."


REMINDER_WORDING = Wording()


@dataclass(frozen=True)
class ScheduleSpec:
    """WHEN, as fields. `clock` is a time of day; `day` is "today" | "tonight" | "tomorrow" | a weekday name ("friday",
    "next friday") | "YYYY-MM-DD" | "MM-DD" | "+N" (N days ahead); `repeat` is "daily" | "weekdays" | "weekly" (with `weekdays` as
    ISO numbers); `every` is an elapsed interval ("hours", 2), first due after `first_in` (default: one interval from now)."""

    clock: ClockSpec | None = None
    day: str | None = None
    repeat: str | None = None
    weekdays: tuple[int, ...] = ()
    every: tuple[str, int] | None = None
    first_in: timedelta | None = None


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


def build_schedule(
    spec: ScheduleSpec,
    tz: str,
    now: datetime,
    *,
    min_interval: timedelta,
    wording: Wording = REMINDER_WORDING,
) -> tuple[Schedule, datetime, bool]:
    """The NEW Schedule for `spec` in zone `tz`, its first occurrence (aware UTC) and whether a bare time rolled to tomorrow.
    Raises ScheduleSpecError (user-safe message) or the scheduling layer's own ScheduleError / TimezoneUnavailableError, which the
    caller words in its own way. A repeating interval shorter than `min_interval` is refused here, for NEW schedules only."""
    if spec.every is not None:
        return _interval(spec, tz, now, min_interval, wording)
    if spec.repeat is not None:
        return _repeating(spec, tz, now, wording)
    return _one_time(spec, tz, now, wording)


def _interval(spec: ScheduleSpec, tz: str, now: datetime, min_interval: timedelta, wording: Wording) -> tuple[Schedule, datetime, bool]:
    if spec.clock is not None or spec.day is not None or spec.repeat is not None:
        raise ScheduleSpecError(f"A {wording.subject} that repeats every so often can't also have a fixed time or day.")
    unit, count = spec.every
    if unit not in _UNITS:
        raise ScheduleSpecError(f"I can repeat a {wording.subject} every N minutes, hours, days or weeks, not '{unit}'.")
    recurrence = Recurrence(_UNITS[unit], int(count))
    anchor = now.replace(microsecond=0) + (spec.first_in if spec.first_in is not None else interval_delta(recurrence))
    schedule = Schedule.every(tz, recurrence, anchor, min_interval=min_interval)
    return schedule, anchor, False


def _repeating(spec: ScheduleSpec, tz: str, now: datetime, wording: Wording) -> tuple[Schedule, datetime, bool]:
    if spec.day is not None:
        raise ScheduleSpecError(f"A repeating {wording.subject} can't also be for one particular day.")
    if spec.clock is None:
        raise ScheduleSpecError("What time should it repeat at? For example: every day at 7 PM.")
    if spec.clock.ambiguous:
        raise ScheduleSpecError(ask_meridiem(spec.clock))
    at = spec.clock.exact
    if spec.repeat == "daily":
        schedule = Schedule.daily(tz, at)
    elif spec.repeat == "weekdays":
        schedule = Schedule.weekdays(tz, at)
    elif spec.repeat == "weekly":
        days = tuple(sorted({d for d in spec.weekdays if isinstance(d, int) and 1 <= d <= 7}))
        if not days:
            raise ScheduleSpecError("Which day of the week should it repeat on? For example: every Monday at 5 PM.")
        schedule = Schedule.weekly(tz, at, days)
    else:
        raise ScheduleSpecError(f"I can repeat a {wording.subject} daily, on weekdays, or on chosen days of the week.")
    return schedule, next_due(schedule, now), False


def _one_time(spec: ScheduleSpec, tz: str, now: datetime, wording: Wording) -> tuple[Schedule, datetime, bool]:
    if spec.clock is None:
        raise ScheduleSpecError(wording.when_question)
    if spec.day is None:
        # No day given: the next time the clock reads that (morning or evening, whichever comes first), never in the past.
        options = [resolve_once(candidate, now, tz) for candidate in spec.clock.candidates]
        schedule, rolled = min(options, key=lambda option: next_due(option[0], now))
        return schedule, next_due(schedule, now), rolled
    if spec.clock.ambiguous:
        raise ScheduleSpecError(ask_meridiem(spec.clock))
    at = spec.clock.exact
    zone = load_zone(tz)
    local_today = now.astimezone(zone).date()
    day = _resolve_day(spec.day, at, now, local_today, zone, wording)
    schedule = Schedule.once(tz, datetime.combine(day, at))
    due = next_due(schedule, now)
    if due is None:
        when = "today" if spec.day in ("today", "tonight") else "that date"
        raise ScheduleSpecError(f"{clock_text(at)} has already passed {when}. Tell me a time that is still ahead.")
    return schedule, due, False


def _resolve_day(day: str, at: time, now: datetime, today, zone, wording: Wording):
    value = day.strip().lower()
    if value in ("today", "tonight"):
        return today
    if value == "tomorrow":
        return today + timedelta(days=1)
    if value.startswith("+") and value[1:].isdigit():
        offset = int(value[1:])
        if not 1 <= offset <= MAX_DAY_OFFSET:
            raise ScheduleSpecError(f"I can set {wording.subject}s up to a year ahead this way - give me a date instead.")
        return today + timedelta(days=offset)
    strictly_after = value.startswith("next ")
    name = value[5:] if strictly_after else value
    if name in WEEKDAY_NUMBERS:
        target = WEEKDAY_NUMBERS[name]
        for ahead in range(0 if not strictly_after else 1, 8):
            candidate = today + timedelta(days=ahead)
            if candidate.isoweekday() == target and local_to_utc(datetime.combine(candidate, at), zone) > now:
                return candidate
        raise ScheduleSpecError("I couldn't work out which day that is.")
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
    raise ScheduleSpecError(f"I couldn't understand the day '{day}'.")
