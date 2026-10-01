"""
Slice 1 - scheduling foundation: time zones. Zone choice (configured -> reliable system zone -> a clear failure, never a guess),
deterministic DST resolution, the system-zone detector's reliability checks, and the purity of app/scheduling.
"""

import pathlib
import re
from datetime import datetime, timedelta, timezone

import pytest

from app.scheduling.timezones import (
    UTC,
    TimezoneUnavailableError,
    ZoneChoice,
    load_zone,
    local_to_utc,
    resolve_timezone,
    valid_zone_name,
)
from app.system_timezone import _WINDOWS_TO_IANA, detect_system_timezone

NY = load_zone("America/New_York")
LONDON = load_zone("Europe/London")
LORD_HOWE = load_zone("Australia/Lord_Howe")


# ====================================== loading a zone ===================================================================


@pytest.mark.parametrize("name", ["UTC", "Asia/Dubai", "Europe/London", "America/Argentina/Buenos_Aires", "Etc/GMT+5", "America/Port-au-Prince"])
def test_valid_iana_names_load(name):
    assert load_zone(name).key == name and valid_zone_name(name)


@pytest.mark.parametrize("name", ["", None, 5, "../etc/passwd", "/etc/localtime", "Europe//London", "x" * 65, "Europe/London\n", "a b", "Europe\\London"])
def test_malformed_names_are_refused_before_any_lookup(name):
    assert not valid_zone_name(name)
    with pytest.raises(TimezoneUnavailableError):
        load_zone(name)


def test_an_unknown_zone_fails_clearly_and_the_advice_names_no_particular_place():
    with pytest.raises(TimezoneUnavailableError) as caught:
        load_zone("Nowhere/Atlantis")
    message = str(caught.value)
    assert "Nowhere/Atlantis" in message and "tzdata" in message and "JARVIS_TIMEZONE" in message
    assert "Dubai" not in message  # nothing about this deployment is hard-coded into the application


# ====================================== choosing the zone for a NEW schedule =============================================


def test_an_explicit_valid_zone_wins_even_when_the_system_zone_differs():
    assert resolve_timezone("Europe/London", "Asia/Dubai") == ZoneChoice("Europe/London", "configured", "")
    assert resolve_timezone("  Europe/London  ", None) == ZoneChoice("Europe/London", "configured", "")


@pytest.mark.parametrize("configured", ["", None, "   "])
def test_without_a_configured_zone_a_reliably_determined_system_zone_is_used(configured):
    assert resolve_timezone(configured, "Europe/London") == ZoneChoice("Europe/London", "system", "")


def test_a_misconfigured_zone_falls_back_to_the_system_zone_but_says_so():
    choice = resolve_timezone("Not/AZone", "Europe/London")
    assert choice.name == "Europe/London" and choice.source == "system"
    assert "Not/AZone" in choice.warning and "Europe/London" in choice.warning  # never silent


def test_no_usable_zone_fails_clearly_instead_of_creating_a_schedule_that_may_fire_at_the_wrong_time():
    for configured, system in ((None, None), ("", None), ("   ", ""), (None, "Not/AZone")):
        with pytest.raises(TimezoneUnavailableError) as caught:
            resolve_timezone(configured, system)
        assert "JARVIS_TIMEZONE" in str(caught.value) and "reliably" in str(caught.value)
    with pytest.raises(TimezoneUnavailableError) as caught:
        resolve_timezone("Not/AZone", None)
    assert "Not/AZone" in str(caught.value)  # a bad setting is named, not hidden


def test_a_schedules_zone_is_its_own_changing_the_global_choice_later_moves_nothing():
    from datetime import time

    from app.scheduling.schedule import Schedule, next_due

    stored = Schedule.daily("Europe/London", time(19, 0))
    after = datetime(2026, 10, 1, 0, 0, tzinfo=UTC)
    before = next_due(stored, after)
    resolve_timezone("Asia/Dubai", None)  # the global setting "changes" - resolution has no memory and no effect on a schedule
    assert stored.tz == "Europe/London" and next_due(stored, after) == before == datetime(2026, 10, 1, 18, 0, tzinfo=UTC)


# ====================================== DST: gaps and overlaps resolve deterministically =================================


def at(zone, *args):
    return local_to_utc(datetime(*args), zone)


def test_an_ordinary_local_time_converts_exactly():
    assert at(NY, 2026, 1, 15, 19, 0) == datetime(2026, 1, 16, 0, 0, tzinfo=UTC)
    assert at(load_zone("Asia/Dubai"), 2026, 10, 1, 19, 0) == datetime(2026, 10, 1, 15, 0, tzinfo=UTC)


@pytest.mark.parametrize("hour, minute", [(2, 0), (2, 1), (2, 30), (2, 59), (3, 0)])
def test_a_time_that_does_not_exist_resolves_to_the_first_valid_minute_after_the_gap(hour, minute):
    # New York, 2026-03-08: clocks jump 02:00 -> 03:00, so 02:00-02:59 never happens.
    resolved = at(NY, 2026, 3, 8, hour, minute)
    assert resolved == datetime(2026, 3, 8, 7, 0, tzinfo=UTC)  # 03:00 EDT
    assert resolved.astimezone(NY).replace(tzinfo=None) == datetime(2026, 3, 8, 3, 0)


def test_the_gap_resolution_is_the_same_in_every_gap_kind():
    assert at(LONDON, 2026, 3, 29, 1, 30).astimezone(LONDON).replace(tzinfo=None) == datetime(2026, 3, 29, 2, 0)  # 01:00 -> 02:00
    assert at(LORD_HOWE, 2026, 10, 4, 2, 15).astimezone(LORD_HOWE).replace(tzinfo=None) == datetime(2026, 10, 4, 2, 30)  # a 30-minute gap
    skipped_day = load_zone("Pacific/Apia")  # Samoa skipped 30 December 2011 entirely: a 24-hour gap
    assert at(skipped_day, 2011, 12, 30, 10, 0).astimezone(skipped_day).replace(tzinfo=None) == datetime(2011, 12, 31, 0, 0)


@pytest.mark.parametrize("hour, minute, expected_utc", [(1, 0, (5, 0)), (1, 30, (5, 30)), (1, 59, (5, 59))])
def test_a_time_that_happens_twice_resolves_to_its_first_occurrence(hour, minute, expected_utc):
    # New York, 2026-11-01: clocks go 02:00 EDT -> 01:00 EST, so 01:00-01:59 happens twice. The FIRST (EDT) is used.
    assert at(NY, 2026, 11, 1, hour, minute) == datetime(2026, 11, 1, *expected_utc, tzinfo=UTC)


def test_the_overlap_rule_holds_elsewhere():
    assert at(LONDON, 2026, 10, 25, 1, 30) == datetime(2026, 10, 25, 0, 30, tzinfo=UTC)  # BST (+1) first
    assert at(LORD_HOWE, 2026, 4, 5, 1, 45) == datetime(2026, 4, 4, 14, 45, tzinfo=UTC)  # +11:00 first, before the 30-minute fall back


def test_local_to_utc_refuses_an_aware_datetime():
    with pytest.raises(ValueError):
        local_to_utc(datetime(2026, 1, 1, 9, 0, tzinfo=UTC), NY)


# ====================================== the system-zone detector (outside the pure package) ==============================

SUMMER = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)


def detect(**kwargs):
    kwargs.setdefault("environ", {})
    kwargs.setdefault("read_text", lambda path: None)
    kwargs.setdefault("read_link", lambda path: None)
    kwargs.setdefault("windows_zone", lambda: None)
    kwargs.setdefault("now", SUMMER)
    return detect_system_timezone(**kwargs)


def test_a_windows_zone_is_translated_and_accepted_only_when_the_offset_agrees_with_the_clock():
    windows = dict(platform="win32", windows_zone=lambda: "Arabian Standard Time")
    assert detect(**windows, system_offset=timedelta(hours=4)) == "Asia/Dubai"
    assert detect(**windows, system_offset=timedelta(hours=5)) is None  # the clock disagrees: not reliable


def test_an_unlisted_or_missing_windows_zone_is_not_reliable():
    assert detect(platform="win32", windows_zone=lambda: "Some Unlisted Standard Time", system_offset=timedelta(hours=4)) is None
    assert detect(platform="win32", windows_zone=lambda: None, system_offset=timedelta(hours=4)) is None


def test_every_windows_translation_is_a_zone_the_database_can_load():
    for windows_name, iana in _WINDOWS_TO_IANA.items():
        assert load_zone(iana), windows_name


def test_dst_zones_are_checked_against_the_offset_in_force_now():
    windows = dict(platform="win32", windows_zone=lambda: "GMT Standard Time")
    assert detect(**windows, system_offset=timedelta(hours=1)) == "Europe/London"  # summer: BST
    assert detect(**windows, now=datetime(2026, 1, 15, 12, 0, tzinfo=UTC), system_offset=timedelta(0)) == "Europe/London"  # winter: GMT
    assert detect(**windows, system_offset=timedelta(0)) is None  # claims GMT in July: disagrees


@pytest.mark.parametrize("tz", ["Europe/London", ":Europe/London"])
def test_the_tz_environment_variable_is_a_candidate_on_posix(tz):
    assert detect(platform="linux", environ={"TZ": tz}, system_offset=timedelta(hours=1)) == "Europe/London"


def test_etc_files_are_candidates_on_posix():
    assert detect(platform="linux", read_text=lambda p: "Europe/London\n" if p == "/etc/timezone" else None, system_offset=timedelta(hours=1)) == "Europe/London"
    link = lambda p: "/usr/share/zoneinfo/Europe/London" if p == "/etc/localtime" else None  # noqa: E731
    assert detect(platform="darwin", read_link=link, system_offset=timedelta(hours=1)) == "Europe/London"


def test_the_first_candidate_that_loads_and_agrees_wins_and_garbage_is_skipped():
    assert detect(platform="linux", environ={"TZ": "Garbage/Zone"}, read_text=lambda p: "Europe/London", system_offset=timedelta(hours=1)) == "Europe/London"
    assert detect(platform="linux", environ={"TZ": "Asia/Dubai"}, read_text=lambda p: "Europe/London", system_offset=timedelta(hours=1)) == "Europe/London"


def test_nothing_determinable_means_none_never_a_guess():
    assert detect(platform="linux", system_offset=timedelta(hours=4)) is None
    assert detect(platform="win32", system_offset=timedelta(hours=4)) is None


def test_the_live_detector_answers_with_a_loadable_zone_or_none():
    live = detect_system_timezone()
    assert live is None or load_zone(live)


# ====================================== purity: no clock, no I/O, no scheduler ===========================================


def test_app_scheduling_is_pure_it_reads_no_clock_no_environment_no_files_no_network():
    root = pathlib.Path(__file__).resolve().parent.parent / "app" / "scheduling"
    forbidden = [
        r"\.now\(", r"\.today\(", r"utcnow", r"time\.time\(", r"time\.monotonic", r"\bopen\(", r"os\.environ", r"os\.getenv",
        r"^\s*import (os|sys|socket|subprocess|threading|sqlite3|httpx|requests|urllib)\b",
        r"^\s*from (os|sys|socket|subprocess|threading|sqlite3|httpx|requests|urllib)\b",
        r"apscheduler|croniter|dateutil",
    ]
    files = sorted(root.glob("*.py"))
    assert {f.name for f in files} == {"__init__.py", "runs.py", "schedule.py", "timezones.py"}
    for path in files:
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert not re.search(pattern, text, re.MULTILINE), (path.name, pattern)
