"""
Plain-words times for reminders - pure (no clock): "7:00 PM", "Thursday 1 October, 7:00 PM". Deterministic and locale-free, the
same style Schedule.describe() uses, so the words said when a reminder is set, listed and announced all agree.
"""

from __future__ import annotations

from datetime import datetime, time

from app.scheduling.schedule import WEEKDAY_NAMES
from app.scheduling.timezones import UTC, TimezoneUnavailableError, load_zone

_MONTHS = (
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",
)


def clock_text(value: time) -> str:
    return f"{value.hour % 12 or 12}:{value.minute:02d} {'AM' if value.hour < 12 else 'PM'}"


def local_phrase(instant: datetime, tz: str | None) -> str:
    """An aware instant as local wall-clock words in `tz` ("Thursday 1 October, 7:00 PM"). With no usable zone it is stated
    in UTC and says so - never silently in some other zone."""
    zone, suffix = UTC, " UTC"
    if tz:
        try:
            zone, suffix = load_zone(tz), ""
        except TimezoneUnavailableError:
            pass
    local = instant.astimezone(zone)
    return f"{WEEKDAY_NAMES[local.weekday()]} {local.day} {_MONTHS[local.month - 1]}, {clock_text(local.time())}{suffix}"
