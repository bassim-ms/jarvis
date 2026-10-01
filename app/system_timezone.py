"""
The machine's own IANA time zone - or None when it cannot be determined RELIABLY.

This is the one place that reads the machine (environment, /etc files, the Windows registry, the clock) to find the system zone, kept
OUT of `app/scheduling/`, which is pure. Its answer is only ever an INPUT to `app.scheduling.timezones.resolve_timezone`, as the
fallback behind an explicit JARVIS_TIMEZONE.

"Reliably" is enforced, not assumed. A candidate name is found from, in order:
  1. the TZ environment variable;
  2. /etc/timezone or the /etc/localtime link (Linux, macOS);
  3. the Windows registry's TimeZoneKeyName, translated to its IANA name (a fixed table of common Windows zones).
and is accepted ONLY IF it loads from the tz database AND its UTC offset right now equals the system clock's own offset. A
translation that is missing, or wrong, or a stale setting disagrees with the clock and yields None - never a guess. The caller then
fails clearly when asked to create a time-based schedule.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone

from app.scheduling.timezones import TimezoneUnavailableError, load_zone

#: Windows time-zone key name -> IANA name (the primary zone for each). Anything not listed is "not reliable": the offset check
#: below catches a wrong entry, and an unlisted zone simply returns None (set JARVIS_TIMEZONE).
_WINDOWS_TO_IANA = {
    "UTC": "Etc/UTC",
    "GMT Standard Time": "Europe/London",
    "Greenwich Standard Time": "Atlantic/Reykjavik",
    "W. Europe Standard Time": "Europe/Berlin",
    "Central Europe Standard Time": "Europe/Budapest",
    "Central European Standard Time": "Europe/Warsaw",
    "Romance Standard Time": "Europe/Paris",
    "GTB Standard Time": "Europe/Bucharest",
    "FLE Standard Time": "Europe/Kiev",
    "E. Europe Standard Time": "Europe/Chisinau",
    "Turkey Standard Time": "Europe/Istanbul",
    "Russian Standard Time": "Europe/Moscow",
    "Israel Standard Time": "Asia/Jerusalem",
    "Egypt Standard Time": "Africa/Cairo",
    "South Africa Standard Time": "Africa/Johannesburg",
    "W. Central Africa Standard Time": "Africa/Lagos",
    "E. Africa Standard Time": "Africa/Nairobi",
    "Arab Standard Time": "Asia/Riyadh",
    "Arabic Standard Time": "Asia/Baghdad",
    "Arabian Standard Time": "Asia/Dubai",
    "Iran Standard Time": "Asia/Tehran",
    "Azerbaijan Standard Time": "Asia/Baku",
    "Georgian Standard Time": "Asia/Tbilisi",
    "Caucasus Standard Time": "Asia/Yerevan",
    "Afghanistan Standard Time": "Asia/Kabul",
    "Pakistan Standard Time": "Asia/Karachi",
    "India Standard Time": "Asia/Kolkata",
    "Sri Lanka Standard Time": "Asia/Colombo",
    "Nepal Standard Time": "Asia/Kathmandu",
    "Bangladesh Standard Time": "Asia/Dhaka",
    "SE Asia Standard Time": "Asia/Bangkok",
    "China Standard Time": "Asia/Shanghai",
    "Taipei Standard Time": "Asia/Taipei",
    "Singapore Standard Time": "Asia/Singapore",
    "Malay Peninsula Standard Time": "Asia/Kuala_Lumpur",
    "Tokyo Standard Time": "Asia/Tokyo",
    "Korea Standard Time": "Asia/Seoul",
    "W. Australia Standard Time": "Australia/Perth",
    "AUS Central Standard Time": "Australia/Darwin",
    "Cen. Australia Standard Time": "Australia/Adelaide",
    "E. Australia Standard Time": "Australia/Brisbane",
    "AUS Eastern Standard Time": "Australia/Sydney",
    "New Zealand Standard Time": "Pacific/Auckland",
    "Hawaiian Standard Time": "Pacific/Honolulu",
    "Alaskan Standard Time": "America/Anchorage",
    "Pacific Standard Time": "America/Los_Angeles",
    "US Mountain Standard Time": "America/Phoenix",
    "Mountain Standard Time": "America/Denver",
    "Central Standard Time": "America/Chicago",
    "Canada Central Standard Time": "America/Regina",
    "Mexico Standard Time": "America/Mexico_City",
    "Eastern Standard Time": "America/New_York",
    "Atlantic Standard Time": "America/Halifax",
    "Newfoundland Standard Time": "America/St_Johns",
    "E. South America Standard Time": "America/Sao_Paulo",
    "Argentina Standard Time": "America/Buenos_Aires",
}

_ZONEINFO_MARKER = "zoneinfo/"
_WINDOWS_TZ_KEY = r"SYSTEM\CurrentControlSet\Control\TimeZoneInformation"


def _read_text(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError:
        return None


def _read_link(path: str) -> str | None:
    try:
        return os.path.realpath(path)
    except OSError:
        return None


def _windows_registry_zone() -> str | None:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _WINDOWS_TZ_KEY) as key:
            value, _kind = winreg.QueryValueEx(key, "TimeZoneKeyName")
        return value if isinstance(value, str) else None
    except (ImportError, OSError):
        return None


def _candidates(
    environ: Mapping[str, str],
    platform: str,
    read_text: Callable[[str], str | None],
    read_link: Callable[[str], str | None],
    windows_zone: Callable[[], str | None],
) -> list[str]:
    names: list[str] = []
    tz = (environ.get("TZ") or "").strip().lstrip(":")
    if tz:
        names.append(tz)
    if platform != "win32":
        text = read_text("/etc/timezone")
        if text and text.strip():
            names.append(text.strip().splitlines()[0].strip())
        target = read_link("/etc/localtime")
        if target and _ZONEINFO_MARKER in target.replace("\\", "/"):
            names.append(target.replace("\\", "/").split(_ZONEINFO_MARKER, 1)[1])
    else:
        windows_name = windows_zone()
        if windows_name and windows_name in _WINDOWS_TO_IANA:
            names.append(_WINDOWS_TO_IANA[windows_name])
    return names


def detect_system_timezone(
    *,
    environ: Mapping[str, str] | None = None,
    platform: str | None = None,
    read_text: Callable[[str], str | None] = _read_text,
    read_link: Callable[[str], str | None] = _read_link,
    windows_zone: Callable[[], str | None] = _windows_registry_zone,
    now: datetime | None = None,
    system_offset: timedelta | None = None,
) -> str | None:
    """The system's IANA zone name if it can be determined reliably, else None. The first candidate that loads AND whose current
    UTC offset matches the system clock's own offset wins. Every input is injectable so the logic is testable on any machine."""
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    now_utc = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    expected = system_offset if system_offset is not None else now_utc.astimezone().utcoffset()
    for name in _candidates(environ, platform, read_text, read_link, windows_zone):
        try:
            zone = load_zone(name)
        except TimezoneUnavailableError:
            continue
        if now_utc.astimezone(zone).utcoffset() == expected:
            return name
    return None
