"""
Schedule: WHEN something should happen, as local wall-clock time, and the pure calculation of its occurrences.

    ONCE      one local date and time                       "at 7 PM tomorrow"
    DAILY     every day at a local time                     "every day at 7 PM"
    WEEKDAYS  Monday to Friday at a local time              "every weekday at 7 PM"
    WEEKLY    chosen weekdays at a local time               "every Monday at 6 PM"
    INTERVAL  every N minutes/hours/days/weeks (elapsed)    "every 30 minutes"   (reuses reminders' Recurrence)

Local time is the source of truth: a schedule stores the zone it was created in plus the wall-clock time, and every UTC instant
is derived from those - so "7 PM" stays 7 PM across daylight-saving changes. INTERVAL is the one exception, by design: it is
elapsed time ("every 2 hours"), counted from a UTC anchor, exactly as an existing recurring reminder is.

PURE: no clock, no I/O. `next_due(schedule, after)` returns the first occurrence STRICTLY after `after`; `previous_due` the last
one at or before it. Both take their time as an argument, and neither ever replays missed occurrences one by one.

The 15-minute minimum for repeating schedules is enforced when a NEW schedule is built (`Schedule.every`), never when a stored
one is loaded (`from_dict`) - what was valid when it was stored stays valid.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Any

from app.reminders.base import Recurrence, RecurrenceUnit
from app.scheduling.timezones import UTC, load_zone, local_to_utc, valid_zone_name


class ScheduleError(ValueError):
    """A schedule is malformed, or a time given to a calculation is not timezone-aware. Never silently repaired."""


class ScheduleKind(Enum):
    ONCE = "once"
    DAILY = "daily"
    WEEKDAYS = "weekdays"
    WEEKLY = "weekly"
    INTERVAL = "interval"


#: What a NEW repeating schedule must be at least (settings.schedule_min_interval_minutes overrides it at creation).
DEFAULT_MIN_INTERVAL = timedelta(minutes=15)
#: The structural limits every Schedule obeys, however it was made.
FLOOR_INTERVAL = timedelta(minutes=1)
MAX_INTERVAL = timedelta(days=400)

_UNIT_DELTA = {
    RecurrenceUnit.MINUTES: timedelta(minutes=1),
    RecurrenceUnit.HOURS: timedelta(hours=1),
    RecurrenceUnit.DAYS: timedelta(days=1),
    RecurrenceUnit.WEEKS: timedelta(weeks=1),
}
WEEKDAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",
)
_CLOCK = re.compile(r"([01]\d|2[0-3]):([0-5]\d)")
_SEARCH_DAYS = 9  # a weekly schedule's occurrences are at most 7 days apart; one spare day either side


def interval_delta(recurrence: Recurrence) -> timedelta:
    return _UNIT_DELTA[recurrence.unit] * recurrence.interval


def _is_whole_minute(value: datetime | time) -> bool:
    return value.second == 0 and value.microsecond == 0


@dataclass(frozen=True)
class Schedule:
    kind: ScheduleKind
    tz: str  # an IANA zone name
    at: time | None = None  # local time of day, whole minutes (DAILY / WEEKDAYS / WEEKLY)
    days: tuple[int, ...] = ()  # ISO weekdays 1-7, Monday = 1 (WEEKLY)
    once_local: datetime | None = None  # a naive local date and time (ONCE)
    interval: Recurrence | None = None  # (INTERVAL)
    anchor: datetime | None = None  # the first occurrence, aware UTC (INTERVAL)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ScheduleKind):
            raise ScheduleError(f"unknown schedule kind: {self.kind!r}")
        if not valid_zone_name(self.tz):
            raise ScheduleError(f"a schedule needs a valid time zone name, got {self.tz!r}")
        object.__setattr__(self, "days", tuple(sorted(set(self.days))) if isinstance(self.days, (tuple, list, set)) else self.days)
        kind = self.kind
        if kind in (ScheduleKind.DAILY, ScheduleKind.WEEKDAYS, ScheduleKind.WEEKLY):
            if not isinstance(self.at, time) or self.at.tzinfo is not None or not _is_whole_minute(self.at):
                raise ScheduleError("`at` must be a local time of day in whole minutes")
            if self.once_local is not None or self.interval is not None or self.anchor is not None:
                raise ScheduleError(f"a {kind.value} schedule takes only a time of day{' and days' if kind is ScheduleKind.WEEKLY else ''}")
            if kind is ScheduleKind.WEEKLY:
                if not self.days or any(not isinstance(d, int) or isinstance(d, bool) or not 1 <= d <= 7 for d in self.days):
                    raise ScheduleError("a weekly schedule needs one or more weekdays, 1 (Monday) to 7 (Sunday)")
            elif self.days:
                raise ScheduleError(f"a {kind.value} schedule takes no days")
        elif kind is ScheduleKind.ONCE:
            if not isinstance(self.once_local, datetime) or self.once_local.tzinfo is not None or not _is_whole_minute(self.once_local):
                raise ScheduleError("a one-time schedule needs a local date and time in whole minutes")
            if self.at is not None or self.days or self.interval is not None or self.anchor is not None:
                raise ScheduleError("a one-time schedule takes only a local date and time")
        else:  # INTERVAL
            if not isinstance(self.interval, Recurrence) or not isinstance(self.interval.unit, RecurrenceUnit):
                raise ScheduleError("an interval schedule needs a Recurrence")
            every = self.interval.interval
            if not isinstance(every, int) or isinstance(every, bool) or every <= 0:
                raise ScheduleError("an interval must be a positive whole number")
            if not FLOOR_INTERVAL <= interval_delta(self.interval) <= MAX_INTERVAL:
                raise ScheduleError("an interval must be between 1 minute and 400 days")
            if not isinstance(self.anchor, datetime) or self.anchor.tzinfo is None:
                raise ScheduleError("an interval schedule needs a timezone-aware anchor time")
            if self.at is not None or self.days or self.once_local is not None:
                raise ScheduleError("an interval schedule takes only an interval and an anchor")
            object.__setattr__(self, "anchor", self.anchor.astimezone(UTC).replace(microsecond=0))

    def __hash__(self) -> int:  # Recurrence is a mutable dataclass, so hash its two values instead
        interval = (self.interval.unit, self.interval.interval) if self.interval is not None else None
        return hash((self.kind, self.tz, self.at, self.days, self.once_local, interval, self.anchor))

    # -- building a NEW schedule (the zone is checked now: creation fails clearly, never later) ---------------------------------

    @classmethod
    def daily(cls, tz: str, at: time) -> "Schedule":
        load_zone(tz)
        return cls(ScheduleKind.DAILY, tz, at=at)

    @classmethod
    def weekdays(cls, tz: str, at: time) -> "Schedule":
        load_zone(tz)
        return cls(ScheduleKind.WEEKDAYS, tz, at=at)

    @classmethod
    def weekly(cls, tz: str, at: time, days: tuple[int, ...] | list[int]) -> "Schedule":
        load_zone(tz)
        return cls(ScheduleKind.WEEKLY, tz, at=at, days=tuple(days))

    @classmethod
    def once(cls, tz: str, local: datetime) -> "Schedule":
        load_zone(tz)
        return cls(ScheduleKind.ONCE, tz, once_local=local)

    @classmethod
    def every(
        cls, tz: str, recurrence: Recurrence, anchor: datetime, *, min_interval: timedelta = DEFAULT_MIN_INTERVAL
    ) -> "Schedule":
        """A NEW repeating schedule: never closer together than `min_interval` (15 minutes by default)."""
        load_zone(tz)
        schedule = cls(ScheduleKind.INTERVAL, tz, interval=recurrence, anchor=anchor)
        if interval_delta(recurrence) < min_interval:
            raise ScheduleError(f"a repeating schedule must be at least {int(min_interval.total_seconds() // 60)} minutes apart")
        return schedule

    # -- persistence: strict, small, never guessed -----------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind.value, "tz": self.tz}
        if self.at is not None:
            data["at"] = self.at.strftime("%H:%M")
        if self.days:
            data["days"] = list(self.days)
        if self.once_local is not None:
            data["once"] = self.once_local.strftime("%Y-%m-%dT%H:%M")
        if self.interval is not None:
            data["interval"] = {"unit": self.interval.unit.value, "every": self.interval.interval}
            data["anchor"] = self.anchor.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "Schedule":
        """The inverse of `to_dict`. Strict: anything unexpected raises ScheduleError. It checks structure, NOT policy: a
        stored schedule is never rejected for being below today's minimum interval."""
        if not isinstance(data, dict):
            raise ScheduleError("a stored schedule must be an object")
        unknown = set(data) - {"kind", "tz", "at", "days", "once", "interval", "anchor"}
        if unknown:
            raise ScheduleError(f"unknown schedule field(s): {sorted(unknown)}")
        try:
            kind = ScheduleKind(data.get("kind"))
        except ValueError:
            raise ScheduleError(f"unknown schedule kind: {data.get('kind')!r}") from None
        at = _parse_clock(data["at"]) if "at" in data else None
        days = data.get("days", [])
        if not isinstance(days, list) or any(not isinstance(d, int) or isinstance(d, bool) for d in days):
            raise ScheduleError("`days` must be a list of weekday numbers")
        once = _parse_local(data["once"]) if "once" in data else None
        interval = anchor = None
        if "interval" in data:
            raw = data["interval"]
            if not isinstance(raw, dict) or set(raw) != {"unit", "every"}:
                raise ScheduleError("`interval` must have exactly `unit` and `every`")
            try:
                interval = Recurrence(unit=RecurrenceUnit(raw["unit"]), interval=raw["every"])
            except ValueError:
                raise ScheduleError(f"unknown interval unit: {raw['unit']!r}") from None
            anchor = _parse_anchor(data.get("anchor"))
        elif "anchor" in data:
            raise ScheduleError("`anchor` belongs only to an interval schedule")
        return cls(kind=kind, tz=data.get("tz"), at=at, days=tuple(days), once_local=once, interval=interval, anchor=anchor)

    def describe(self) -> str:
        """Plain words for confirming a schedule back to the user - deterministic, no clock, no locale."""
        zone = f"({self.tz})"
        if self.kind is ScheduleKind.DAILY:
            return f"every day at {_clock(self.at)} {zone}"
        if self.kind is ScheduleKind.WEEKDAYS:
            return f"every weekday at {_clock(self.at)} {zone}"
        if self.kind is ScheduleKind.WEEKLY:
            names = [WEEKDAY_NAMES[d - 1] for d in self.days]
            listed = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
            return f"every {listed} at {_clock(self.at)} {zone}"
        if self.kind is ScheduleKind.ONCE:
            d = self.once_local
            return f"on {WEEKDAY_NAMES[d.weekday()]} {d.day} {_MONTH_NAMES[d.month - 1]} {d.year} at {_clock(d.time())} {zone}"
        n = self.interval.interval
        unit = self.interval.unit.value[:-1] if n == 1 else self.interval.unit.value
        return f"every {n} {unit}"


def _clock(value: time) -> str:
    return f"{value.hour % 12 or 12}:{value.minute:02d} {'AM' if value.hour < 12 else 'PM'}"


def _parse_clock(value: Any) -> time:
    match = _CLOCK.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ScheduleError(f"`at` must look like HH:MM, got {value!r}")
    return time(int(match.group(1)), int(match.group(2)))


def _parse_local(value: Any) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M")
    except (TypeError, ValueError):
        raise ScheduleError(f"`once` must look like YYYY-MM-DDTHH:MM, got {value!r}") from None


def _parse_anchor(value: Any) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise ScheduleError(f"`anchor` must be an ISO time with a UTC offset, got {value!r}") from None
    if parsed.tzinfo is None:
        raise ScheduleError("`anchor` must include a UTC offset")
    return parsed


def _as_utc(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ScheduleError(f"`{name}` must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _on_day(schedule: Schedule, day) -> bool:
    if schedule.kind is ScheduleKind.DAILY:
        return True
    if schedule.kind is ScheduleKind.WEEKDAYS:
        return day.isoweekday() <= 5
    return day.isoweekday() in schedule.days


# -- the calculation ----------------------------------------------------------------------------------------------------------------


def next_due(schedule: Schedule, after: datetime) -> datetime | None:
    """The first occurrence STRICTLY after `after`, as an aware UTC datetime, or None if there is none (a one-time schedule whose
    time has passed). Always moves forward: feeding the result back in never returns the same instant. Missed occurrences are
    never counted or replayed - this jumps straight to the next one."""
    after = _as_utc(after, "after")
    if schedule.kind is ScheduleKind.ONCE:
        instant = local_to_utc(schedule.once_local, load_zone(schedule.tz))
        return instant if instant > after else None
    if schedule.kind is ScheduleKind.INTERVAL:
        step = interval_delta(schedule.interval)
        if schedule.anchor > after:
            return schedule.anchor
        return schedule.anchor + step * ((after - schedule.anchor) // step + 1)
    zone = load_zone(schedule.tz)
    first_day = after.astimezone(zone).date()
    for offset in range(_SEARCH_DAYS):
        day = first_day + timedelta(days=offset)
        if _on_day(schedule, day):
            candidate = local_to_utc(datetime.combine(day, schedule.at), zone)
            if candidate > after:
                return candidate
    raise ScheduleError("no next occurrence could be found")  # unreachable for a valid schedule


def previous_due(schedule: Schedule, at: datetime) -> datetime | None:
    """The last occurrence at or before `at` (aware UTC), or None if there has been none yet."""
    at = _as_utc(at, "at")
    if schedule.kind is ScheduleKind.ONCE:
        instant = local_to_utc(schedule.once_local, load_zone(schedule.tz))
        return instant if instant <= at else None
    if schedule.kind is ScheduleKind.INTERVAL:
        if schedule.anchor > at:
            return None
        step = interval_delta(schedule.interval)
        return schedule.anchor + step * ((at - schedule.anchor) // step)
    zone = load_zone(schedule.tz)
    last_day = at.astimezone(zone).date()
    for offset in range(_SEARCH_DAYS):
        day = last_day - timedelta(days=offset)
        if _on_day(schedule, day):
            candidate = local_to_utc(datetime.combine(day, schedule.at), zone)
            if candidate <= at:
                return candidate
    return None


def resolve_once(local_time: time, now: datetime, tz: str) -> tuple[Schedule, bool]:
    """A one-time schedule for "at 7 PM": today if that is still ahead of `now`, otherwise tomorrow. Returns the schedule and
    whether it rolled to tomorrow (so the reply can say so). Exactly now counts as already passed."""
    zone = load_zone(tz)
    now = _as_utc(now, "now")
    today = now.astimezone(zone).date()
    for rolled, day in ((False, today), (True, today + timedelta(days=1))):
        wall = datetime.combine(day, local_time)
        if local_to_utc(wall, zone) > now:
            return Schedule.once(tz, wall), rolled
    raise ScheduleError("no future time could be found")  # unreachable
