"""
Time zones for schedules: loading a zone, choosing which zone a NEW schedule is created in, and turning a local wall-clock time into
a UTC instant - deterministically, including across daylight-saving changes.

PURE: nothing here reads the clock, the environment, a file of ours, or the network. Zone rules come read-only from the `tzdata`
package through `zoneinfo`; every function takes everything else as an argument. (Finding out what the SYSTEM's zone is requires
reading the machine, so that lives outside this package, in app/system_timezone.py, and its answer is passed in.)

Choosing the zone for a new schedule (`resolve_timezone`), in this order:

  1. the explicitly configured zone (JARVIS_TIMEZONE), if it is a valid IANA name;
  2. otherwise the system's zone - only when the caller could determine it RELIABLY (None means it could not);
  3. otherwise a clear `TimezoneUnavailableError`: no schedule is created that might fire at the wrong time.

A schedule stores the zone it was created in and keeps it; changing the global setting later never moves an existing schedule.

DST, resolved the same way every time (`local_to_utc`):
  * a local time that does not exist (the clocks jumped forward over it) resolves to the first valid local minute after the gap;
  * a local time that happens twice (the clocks went back) resolves to its FIRST occurrence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from functools import lru_cache
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

UTC = timezone.utc

MAX_ZONE_NAME_CHARS = 64
#: Area/Location (and Area/Location/Sub), "UTC", "Etc/GMT+5": letters, digits, "_", "+", "-" only - never a path.
_ZONE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]*(?:/[A-Za-z0-9_+\-]+){0,2}")
#: A gap is at most a day long (Pacific/Apia skipped a whole day once); the search for its end is bounded.
_MAX_GAP_MINUTES = 48 * 60

_ADVICE = (
    "Set JARVIS_TIMEZONE to an IANA time zone name such as 'Europe/London' (Region/City), and make sure the 'tzdata' "
    "package is installed."
)


class TimezoneUnavailableError(ValueError):
    """A time zone could not be determined or loaded. Never silently replaced by a guess."""


def valid_zone_name(name: object) -> bool:
    """Whether `name` has the shape of an IANA zone name (a format check only - no database lookup)."""
    return isinstance(name, str) and 0 < len(name) <= MAX_ZONE_NAME_CHARS and _ZONE_NAME.fullmatch(name) is not None


@lru_cache(maxsize=64)
def _load(name: str) -> ZoneInfo:
    return ZoneInfo(name)


def load_zone(name: object) -> ZoneInfo:
    """The ZoneInfo for an IANA zone name, or TimezoneUnavailableError (bad name, unknown zone, or no tz database)."""
    if not valid_zone_name(name):
        raise TimezoneUnavailableError(f"{name!r} is not a valid time zone name. {_ADVICE}")
    try:
        return _load(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        raise TimezoneUnavailableError(f"The time zone '{name}' is not available on this machine. {_ADVICE}") from None


@dataclass(frozen=True)
class ZoneChoice:
    """The zone a new schedule will be created in. `source` is "configured" or "system". `warning` is non-empty when the
    configured value was set but unusable and the system zone was used instead - the caller should tell the user."""

    name: str
    source: str
    warning: str = ""


def resolve_timezone(configured: str | None, system_zone: str | None) -> ZoneChoice:
    """Pick the zone for a NEW schedule - see the module docstring for the order. `system_zone` is the machine's IANA zone
    name if it could be determined reliably, else None. Raises TimezoneUnavailableError when neither is usable."""
    configured = (configured or "").strip()
    rejected = ""
    if configured:
        try:
            load_zone(configured)
            return ZoneChoice(configured, "configured")
        except TimezoneUnavailableError:
            rejected = configured
    if system_zone:
        try:
            load_zone(system_zone)
        except TimezoneUnavailableError:
            system_zone = None
        else:
            warning = (
                f"JARVIS_TIMEZONE='{rejected}' is not a usable time zone; using the system time zone '{system_zone}' instead."
                if rejected
                else ""
            )
            return ZoneChoice(system_zone, "system", warning)
    if rejected:
        raise TimezoneUnavailableError(
            f"JARVIS_TIMEZONE='{rejected}' is not a usable time zone and the system time zone could not be determined "
            f"reliably. {_ADVICE}"
        )
    raise TimezoneUnavailableError(
        f"No time zone is configured and the system time zone could not be determined reliably, so no schedule can be "
        f"created that is certain to fire at the right time. {_ADVICE}"
    )


def _exists(naive: datetime, zone: tzinfo) -> bool:
    """Whether this local wall-clock time exists in `zone` (False inside a spring-forward gap)."""
    return naive.replace(tzinfo=zone, fold=0).astimezone(UTC).astimezone(zone).replace(tzinfo=None) == naive


def local_to_utc(naive: datetime, zone: tzinfo) -> datetime:
    """The UTC instant for a local wall-clock time in `zone`, resolved deterministically across DST:
    a nonexistent time -> the first valid local minute after the gap; an ambiguous time -> its first occurrence."""
    if naive.tzinfo is not None:
        raise ValueError("local_to_utc needs a naive (wall-clock) datetime")
    candidate = naive
    for _ in range(_MAX_GAP_MINUTES + 1):
        if _exists(candidate, zone):
            return candidate.replace(tzinfo=zone, fold=0).astimezone(UTC)  # fold=0: the first of two occurrences
        candidate = candidate.replace(second=0, microsecond=0) + timedelta(minutes=1)
    raise TimezoneUnavailableError("could not resolve a local time across a daylight-saving gap")
