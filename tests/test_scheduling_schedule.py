"""
Slice 1 - scheduling foundation: the Schedule value type, its strict persistence, and the pure occurrence calculation
(`next_due`, `previous_due`, `resolve_once`), including daylight-saving behaviour and randomised property tests.
"""

import random
from datetime import datetime, time, timedelta, timezone

import pytest

from app.reminders.base import Recurrence, RecurrenceUnit
from app.scheduling.schedule import (
    DEFAULT_MIN_INTERVAL,
    Schedule,
    ScheduleError,
    ScheduleKind,
    next_due,
    previous_due,
    resolve_once,
)
from app.scheduling.timezones import UTC, TimezoneUnavailableError, load_zone

DUBAI, NYC, LONDON, LORD_HOWE, KOLKATA = "Asia/Dubai", "America/New_York", "Europe/London", "Australia/Lord_Howe", "Asia/Kolkata"


def utc(*args):
    return datetime(*args, tzinfo=UTC)


def local(instant, tz):
    return instant.astimezone(load_zone(tz)).replace(tzinfo=None)


# ====================================== construction and validation ======================================================


def test_every_kind_can_be_built():
    assert Schedule.daily(DUBAI, time(19, 0)).kind is ScheduleKind.DAILY
    assert Schedule.weekdays(DUBAI, time(19, 0)).kind is ScheduleKind.WEEKDAYS
    assert Schedule.weekly(DUBAI, time(18, 0), (1,)).days == (1,)
    assert Schedule.once(DUBAI, datetime(2026, 10, 2, 18, 0)).kind is ScheduleKind.ONCE
    every = Schedule.every(DUBAI, Recurrence(RecurrenceUnit.HOURS, 2), utc(2026, 10, 1, 10, 0))
    assert every.kind is ScheduleKind.INTERVAL and every.anchor == utc(2026, 10, 1, 10, 0)


def test_weekly_days_are_normalised_sorted_and_deduplicated():
    assert Schedule.weekly(DUBAI, time(9, 0), [5, 1, 3, 1]).days == (1, 3, 5)


@pytest.mark.parametrize(
    "build",
    [
        lambda: Schedule.daily(DUBAI, time(19, 0, 30)),  # seconds are not allowed
        lambda: Schedule.daily(DUBAI, time(19, 0, tzinfo=timezone.utc)),  # a time of day is local, never aware
        lambda: Schedule.daily(DUBAI, "19:00"),
        lambda: Schedule.weekly(DUBAI, time(9, 0), ()),
        lambda: Schedule.weekly(DUBAI, time(9, 0), (0,)),
        lambda: Schedule.weekly(DUBAI, time(9, 0), (8,)),
        lambda: Schedule.weekly(DUBAI, time(9, 0), (True,)),
        lambda: Schedule.once(DUBAI, datetime(2026, 10, 2, 18, 0, tzinfo=UTC)),
        lambda: Schedule.once(DUBAI, datetime(2026, 10, 2, 18, 0, 30)),
        lambda: Schedule.every(DUBAI, Recurrence(RecurrenceUnit.MINUTES, 10), utc(2026, 10, 1)),  # under the 15-minute minimum
        lambda: Schedule.every(DUBAI, Recurrence(RecurrenceUnit.MINUTES, 14), utc(2026, 10, 1)),
        lambda: Schedule.every(DUBAI, Recurrence(RecurrenceUnit.MINUTES, 15), datetime(2026, 10, 1)),  # the anchor must be aware
        lambda: Schedule.every(DUBAI, Recurrence(RecurrenceUnit.WEEKS, 100), utc(2026, 10, 1)),  # over 400 days
        lambda: Schedule(ScheduleKind.DAILY, DUBAI, at=time(9, 0), days=(1,)),  # a daily schedule takes no days
        lambda: Schedule(ScheduleKind.DAILY, DUBAI),  # ... but needs a time
        lambda: Schedule(ScheduleKind.ONCE, DUBAI, at=time(9, 0), once_local=datetime(2026, 1, 1, 9, 0)),  # no stray fields
        lambda: Schedule("daily", DUBAI, at=time(9, 0)),
    ],
)
def test_invalid_schedules_are_refused(build):
    with pytest.raises(ScheduleError):
        build()


def test_a_new_schedule_in_an_unavailable_zone_fails_clearly_at_creation():
    with pytest.raises(TimezoneUnavailableError):
        Schedule.daily("Nowhere/Atlantis", time(19, 0))
    with pytest.raises(ScheduleError):
        Schedule(ScheduleKind.DAILY, "../etc", at=time(9, 0))


def test_the_15_minute_minimum_applies_to_new_schedules_only_and_is_configurable():
    anchor = utc(2026, 10, 1)
    assert DEFAULT_MIN_INTERVAL == timedelta(minutes=15)
    assert Schedule.every(DUBAI, Recurrence(RecurrenceUnit.MINUTES, 15), anchor).interval.interval == 15
    with pytest.raises(ScheduleError):
        Schedule.every(DUBAI, Recurrence(RecurrenceUnit.MINUTES, 29), anchor, min_interval=timedelta(minutes=30))
    assert Schedule.every(DUBAI, Recurrence(RecurrenceUnit.MINUTES, 30), anchor, min_interval=timedelta(minutes=30))
    # a stored schedule that was valid when it was written is still loaded, whatever today's minimum is
    stored = {"kind": "interval", "tz": DUBAI, "interval": {"unit": "minutes", "every": 5}, "anchor": "2026-10-01T10:00:00+00:00"}
    assert Schedule.from_dict(stored).interval.interval == 5


def test_a_schedule_is_immutable_and_hashable():
    daily = Schedule.daily(DUBAI, time(19, 0))
    with pytest.raises(Exception):
        daily.tz = "UTC"
    every = Schedule.every(DUBAI, Recurrence(RecurrenceUnit.HOURS, 2), utc(2026, 10, 1))
    assert len({daily, Schedule.daily(DUBAI, time(19, 0)), every, Schedule.every(DUBAI, Recurrence(RecurrenceUnit.HOURS, 2), utc(2026, 10, 1))}) == 2
    assert daily != Schedule.daily(NYC, time(19, 0))


# ====================================== persistence: strict, small, exact ================================================

SCHEDULES = [
    Schedule.daily(DUBAI, time(19, 0)),
    Schedule.weekdays(LONDON, time(7, 5)),
    Schedule.weekly(NYC, time(18, 0), (1, 3, 7)),
    Schedule.once(DUBAI, datetime(2026, 10, 2, 18, 0)),
    Schedule.every(KOLKATA, Recurrence(RecurrenceUnit.HOURS, 6), utc(2026, 10, 1, 10, 0)),
]


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_a_schedule_survives_a_round_trip_through_plain_json(schedule):
    import json

    revived = Schedule.from_dict(json.loads(json.dumps(schedule.to_dict())))
    assert revived == schedule and hash(revived) == hash(schedule)


def test_the_stored_shape_is_small_and_exact():
    assert SCHEDULES[0].to_dict() == {"kind": "daily", "tz": DUBAI, "at": "19:00"}
    assert SCHEDULES[2].to_dict() == {"kind": "weekly", "tz": NYC, "at": "18:00", "days": [1, 3, 7]}
    assert SCHEDULES[3].to_dict() == {"kind": "once", "tz": DUBAI, "once": "2026-10-02T18:00"}
    assert SCHEDULES[4].to_dict() == {
        "kind": "interval", "tz": KOLKATA, "interval": {"unit": "hours", "every": 6}, "anchor": "2026-10-01T10:00:00+00:00",
    }


@pytest.mark.parametrize(
    "data",
    [
        None, [], "daily", {}, {"kind": "yearly", "tz": DUBAI}, {"kind": "daily"}, {"kind": "daily", "tz": DUBAI},
        {"kind": "daily", "tz": DUBAI, "at": "7pm"}, {"kind": "daily", "tz": DUBAI, "at": "24:00"}, {"kind": "daily", "tz": DUBAI, "at": "19:00:30"},
        {"kind": "daily", "tz": DUBAI, "at": "19:00", "extra": 1}, {"kind": "weekly", "tz": DUBAI, "at": "09:00", "days": "1"},
        {"kind": "weekly", "tz": DUBAI, "at": "09:00", "days": [True]}, {"kind": "weekly", "tz": DUBAI, "at": "09:00", "days": []},
        {"kind": "once", "tz": DUBAI, "once": "2026-10-02 18:00"}, {"kind": "once", "tz": DUBAI, "once": "2026-13-02T18:00"},
        {"kind": "interval", "tz": DUBAI, "interval": {"unit": "fortnights", "every": 1}, "anchor": "2026-10-01T10:00:00+00:00"},
        {"kind": "interval", "tz": DUBAI, "interval": {"unit": "hours", "every": 1}, "anchor": "2026-10-01T10:00:00"},
        {"kind": "interval", "tz": DUBAI, "interval": {"unit": "hours", "every": 1}},
        {"kind": "interval", "tz": DUBAI, "interval": {"unit": "hours", "every": True}, "anchor": "2026-10-01T10:00:00+00:00"},
        {"kind": "interval", "tz": DUBAI, "interval": {"unit": "hours"}, "anchor": "2026-10-01T10:00:00+00:00"},
        {"kind": "daily", "tz": DUBAI, "at": "19:00", "anchor": "2026-10-01T10:00:00+00:00"},
        {"kind": "daily", "tz": "x" * 80, "at": "19:00"},
    ],
)
def test_loading_is_strict_nothing_unexpected_is_guessed_at(data):
    with pytest.raises(ScheduleError):
        Schedule.from_dict(data)


@pytest.mark.parametrize(
    "schedule, text",
    [
        (SCHEDULES[0], "every day at 7:00 PM (Asia/Dubai)"),
        (SCHEDULES[1], "every weekday at 7:05 AM (Europe/London)"),
        (SCHEDULES[2], "every Monday, Wednesday and Sunday at 6:00 PM (America/New_York)"),
        (Schedule.weekly(DUBAI, time(0, 0), (4,)), "every Thursday at 12:00 AM (Asia/Dubai)"),
        (Schedule.weekly(DUBAI, time(12, 30), (2, 5)), "every Tuesday and Friday at 12:30 PM (Asia/Dubai)"),
        (SCHEDULES[3], "on Friday 2 October 2026 at 6:00 PM (Asia/Dubai)"),
        (SCHEDULES[4], "every 6 hours"),
        (Schedule.every(DUBAI, Recurrence(RecurrenceUnit.HOURS, 1), utc(2026, 10, 1)), "every 1 hour"),
    ],
)
def test_describe_is_plain_deterministic_words(schedule, text):
    assert schedule.describe() == text


# ====================================== next_due: the occurrences ========================================================


def test_daily_before_at_and_after_the_time_and_always_strictly_after():
    daily = Schedule.daily(DUBAI, time(19, 0))  # 15:00 UTC
    assert next_due(daily, utc(2026, 10, 1, 14, 59, 59)) == utc(2026, 10, 1, 15, 0)
    assert next_due(daily, utc(2026, 10, 1, 15, 0)) == utc(2026, 10, 2, 15, 0)  # exactly at the time -> the NEXT one
    assert next_due(daily, utc(2026, 10, 1, 15, 0, 1)) == utc(2026, 10, 2, 15, 0)
    assert next_due(daily, utc(2026, 10, 1, 21, 30)) == utc(2026, 10, 2, 15, 0)  # 01:30 local next day: 19:00 that day is ahead


def test_the_local_date_is_what_counts_not_the_utc_date():
    daily = Schedule.daily(DUBAI, time(1, 0))  # 21:00 UTC the previous day
    assert next_due(daily, utc(2026, 10, 1, 20, 59)) == utc(2026, 10, 1, 21, 0)
    assert next_due(daily, utc(2026, 10, 1, 21, 0)) == utc(2026, 10, 2, 21, 0)


def test_weekdays_skip_the_weekend():
    weekdays = Schedule.weekdays(DUBAI, time(19, 0))
    friday = utc(2026, 10, 2, 15, 0)  # Friday 19:00 Dubai (2026-10-02 is a Friday)
    assert next_due(weekdays, friday - timedelta(seconds=1)) == friday
    assert next_due(weekdays, friday) == utc(2026, 10, 5, 15, 0)  # Monday
    assert next_due(weekdays, utc(2026, 10, 3, 9, 0)) == utc(2026, 10, 5, 15, 0)  # Saturday morning -> Monday


def test_weekly_on_chosen_days_across_week_month_and_year_ends():
    weekly = Schedule.weekly(DUBAI, time(18, 0), (1, 3))  # Mondays and Wednesdays, 14:00 UTC
    assert next_due(weekly, utc(2026, 10, 1, 0, 0)) == utc(2026, 10, 5, 14, 0)  # Thursday -> Monday 5 Oct
    assert next_due(weekly, utc(2026, 10, 5, 14, 0)) == utc(2026, 10, 7, 14, 0)  # Monday -> Wednesday
    assert next_due(weekly, utc(2026, 10, 7, 14, 0)) == utc(2026, 10, 12, 14, 0)  # Wednesday -> next Monday
    assert next_due(weekly, utc(2026, 12, 30, 14, 0)) == utc(2027, 1, 4, 14, 0)  # across the year end
    assert next_due(Schedule.weekly(DUBAI, time(9, 0), (7,)), utc(2026, 2, 28, 12, 0)) == utc(2026, 3, 1, 5, 0)  # Sunday 1 Mar


def test_a_leap_day_is_an_ordinary_day_for_a_daily_schedule():
    daily = Schedule.daily(DUBAI, time(19, 0))
    assert next_due(daily, utc(2028, 2, 28, 15, 0)) == utc(2028, 2, 29, 15, 0)
    assert next_due(daily, utc(2028, 2, 29, 15, 0)) == utc(2028, 3, 1, 15, 0)


def test_a_one_time_schedule_has_one_occurrence_then_none():
    once = Schedule.once(DUBAI, datetime(2026, 10, 2, 18, 0))  # 14:00 UTC
    assert next_due(once, utc(2026, 10, 1, 0, 0)) == utc(2026, 10, 2, 14, 0)
    assert next_due(once, utc(2026, 10, 2, 14, 0)) is None  # exactly at it: nothing strictly after
    assert next_due(once, utc(2026, 10, 3, 0, 0)) is None
    assert previous_due(once, utc(2026, 10, 1, 0, 0)) is None and previous_due(once, utc(2026, 10, 2, 14, 0)) == utc(2026, 10, 2, 14, 0)


def test_an_interval_jumps_straight_to_the_next_slot_never_replaying_missed_ones():
    every = Schedule.every(DUBAI, Recurrence(RecurrenceUnit.MINUTES, 30), utc(2026, 10, 1, 10, 0))
    assert next_due(every, utc(2026, 9, 1)) == utc(2026, 10, 1, 10, 0)  # before the anchor: the anchor is the first run
    assert next_due(every, utc(2026, 10, 1, 10, 0)) == utc(2026, 10, 1, 10, 30)
    assert next_due(every, utc(2026, 10, 1, 10, 29, 59)) == utc(2026, 10, 1, 10, 30)
    assert next_due(every, utc(2026, 10, 8, 10, 5)) == utc(2026, 10, 8, 10, 30)  # a week later: one step, no catching up
    assert previous_due(every, utc(2026, 10, 8, 10, 5)) == utc(2026, 10, 8, 10, 0) and previous_due(every, utc(2026, 9, 1)) is None


def test_next_due_refuses_naive_times():
    with pytest.raises(ScheduleError):
        next_due(Schedule.daily(DUBAI, time(19, 0)), datetime(2026, 10, 1, 12, 0))
    with pytest.raises(ScheduleError):
        previous_due(Schedule.daily(DUBAI, time(19, 0)), datetime(2026, 10, 1, 12, 0))


def test_any_aware_time_zone_for_the_input_gives_the_same_answer():
    daily = Schedule.daily(DUBAI, time(19, 0))
    same_instant = datetime(2026, 10, 1, 8, 0, tzinfo=load_zone("America/Los_Angeles"))  # 15:00 UTC
    assert next_due(daily, same_instant) == next_due(daily, same_instant.astimezone(UTC)) == utc(2026, 10, 2, 15, 0)


# ====================================== DST ==============================================================================


def test_a_daily_time_in_a_spring_forward_gap_runs_once_at_the_first_valid_minute():
    daily = Schedule.daily(NYC, time(2, 30))
    assert next_due(daily, utc(2026, 3, 7, 12, 0)) == utc(2026, 3, 8, 7, 0)  # 03:00 EDT on the gap day
    assert local(next_due(daily, utc(2026, 3, 7, 12, 0)), NYC) == datetime(2026, 3, 8, 3, 0)
    assert next_due(daily, utc(2026, 3, 8, 7, 0)) == utc(2026, 3, 9, 6, 30)  # the next day is an ordinary 02:30 EDT


def test_a_daily_time_in_a_fall_back_overlap_runs_exactly_once_at_the_first_occurrence():
    daily = Schedule.daily(NYC, time(1, 30))
    occurrences, cursor = [], utc(2026, 10, 30, 12, 0)
    for _ in range(4):
        cursor = next_due(daily, cursor)
        occurrences.append(cursor)
    assert occurrences == [utc(2026, 10, 31, 5, 30), utc(2026, 11, 1, 5, 30), utc(2026, 11, 2, 6, 30), utc(2026, 11, 3, 6, 30)]
    assert utc(2026, 11, 1, 6, 30) not in occurrences  # the SECOND 01:30 (EST) is never an occurrence
    assert len(set(occurrences)) == 4


def test_the_wall_clock_time_is_constant_across_a_transition():
    daily = Schedule.daily(LONDON, time(19, 0))
    before, after = next_due(daily, utc(2026, 3, 27, 20, 0)), next_due(daily, utc(2026, 3, 29, 20, 0))
    assert local(before, LONDON).time() == local(after, LONDON).time() == time(19, 0)
    assert (before.hour, after.hour) == (19, 18)  # GMT then BST: the UTC hour moved, the local hour did not


def test_a_30_minute_dst_shift_and_a_zone_without_dst():
    lord_howe = Schedule.daily(LORD_HOWE, time(2, 15))
    assert local(next_due(lord_howe, utc(2026, 10, 3, 0, 0)), LORD_HOWE) == datetime(2026, 10, 4, 2, 30)  # gap 02:00 -> 02:30
    dubai = Schedule.daily(DUBAI, time(19, 0))
    assert {next_due(dubai, utc(2026, month, 15, 0, 0)).hour for month in range(1, 13)} == {15}  # never moves


@pytest.mark.parametrize("tz, at_", [(NYC, time(2, 30)), (NYC, time(1, 30)), (LONDON, time(1, 30)), (LORD_HOWE, time(2, 15)), (DUBAI, time(19, 0))])
def test_a_year_of_daily_occurrences_is_exactly_one_per_local_date(tz, at_):
    daily, zone = Schedule.daily(tz, at_), load_zone(tz)
    cursor, dates, instants = datetime(2025, 12, 31, 12, 0, tzinfo=UTC), [], []
    while True:
        cursor = next_due(daily, cursor)
        if cursor.astimezone(zone).date().year > 2026:
            break
        if cursor.astimezone(zone).date().year == 2026:  # (the scan starts a little before 1 Jan; only 2026 is counted)
            dates.append(cursor.astimezone(zone).date())
            instants.append(cursor)
    assert len(dates) == 365 and len(set(dates)) == 365  # one per local date
    assert all((b - a).days == 1 for a, b in zip(dates, dates[1:]))  # no date skipped
    assert instants == sorted(set(instants))  # strictly increasing, no duplicate instant


# ====================================== property tests (seeded, stdlib only) =============================================


def _random_instants(count, seed):
    rng = random.Random(seed)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    return [start + timedelta(seconds=rng.randrange(0, 2 * 365 * 24 * 3600)) for _ in range(count)]


@pytest.mark.parametrize("tz", [DUBAI, NYC, LONDON, LORD_HOWE, KOLKATA])
@pytest.mark.parametrize(
    "make",
    [
        lambda tz: Schedule.daily(tz, time(2, 30)),
        lambda tz: Schedule.daily(tz, time(19, 0)),
        lambda tz: Schedule.weekdays(tz, time(1, 30)),
        lambda tz: Schedule.weekly(tz, time(6, 45), (2, 4, 7)),
    ],
)
def test_next_due_properties_hold_for_random_times(tz, make):
    schedule, zone = make(tz), load_zone(tz)
    for after in _random_instants(250, seed=hash((tz, schedule.kind.value)) & 0xFFFF):
        result = next_due(schedule, after)
        assert result > after and result - after <= timedelta(days=8)  # strictly later, never more than a week away
        wall = result.astimezone(zone)
        assert wall.time() >= schedule.at  # exactly `at`, or - only inside a DST gap - the first valid minute after it
        if schedule.kind is ScheduleKind.WEEKDAYS:
            assert wall.isoweekday() <= 5
        if schedule.kind is ScheduleKind.WEEKLY:
            assert wall.isoweekday() in schedule.days
        assert previous_due(schedule, result) == result  # it IS an occurrence
        assert previous_due(schedule, result - timedelta(seconds=1)) <= after  # and none was skipped on the way
        assert next_due(schedule, result) > result  # always advances


def test_interval_properties_hold_for_random_times():
    every = Schedule.every(DUBAI, Recurrence(RecurrenceUnit.MINUTES, 45), utc(2026, 3, 3, 3, 3))
    for after in _random_instants(300, seed=7):
        result = next_due(every, after)
        assert result > after and (result - every.anchor) % timedelta(minutes=45) == timedelta(0)
        # before the anchor the first run IS the anchor; after it, never more than one step away
        assert result == every.anchor if after < every.anchor else result - after <= timedelta(minutes=45)


# ====================================== resolve_once: "at 7 PM" ==========================================================


def test_at_a_time_still_ahead_today_is_today_and_otherwise_tomorrow():
    now = utc(2026, 10, 1, 10, 0)  # 14:00 in Dubai
    schedule, rolled = resolve_once(time(19, 0), now, DUBAI)
    assert (schedule.once_local, rolled) == (datetime(2026, 10, 1, 19, 0), False)
    schedule, rolled = resolve_once(time(13, 0), now, DUBAI)
    assert (schedule.once_local, rolled) == (datetime(2026, 10, 2, 13, 0), True)
    schedule, rolled = resolve_once(time(14, 0), now, DUBAI)  # exactly now counts as already passed
    assert (schedule.once_local, rolled) == (datetime(2026, 10, 2, 14, 0), True)


def test_resolve_once_uses_the_local_date_and_handles_a_dst_gap():
    schedule, rolled = resolve_once(time(2, 30), utc(2026, 3, 8, 3, 0), NYC)  # 22:00 on the 7th in New York: 02:30 today has passed
    assert rolled is True and schedule.once_local == datetime(2026, 3, 8, 2, 30)  # tomorrow is the gap day ...
    assert next_due(schedule, utc(2026, 3, 8, 3, 0)) == utc(2026, 3, 8, 7, 0)  # ... where 02:30 resolves to 03:00 EDT
    earlier, rolled = resolve_once(time(2, 30), utc(2026, 3, 7, 12, 0), NYC)  # 07:00 on the 7th: 02:30 tomorrow is not "today"
    assert rolled is True
    late, rolled = resolve_once(time(2, 30), utc(2026, 3, 8, 7, 30), NYC)  # already past 03:00 EDT that day
    assert rolled is True and late.once_local == datetime(2026, 3, 9, 2, 30)


def test_resolve_once_refuses_what_it_cannot_do_safely():
    with pytest.raises(TimezoneUnavailableError):
        resolve_once(time(19, 0), utc(2026, 10, 1), "Nowhere/Atlantis")
    with pytest.raises(ScheduleError):
        resolve_once(time(19, 0, 30), utc(2026, 10, 1), DUBAI)
    with pytest.raises(ScheduleError):
        resolve_once(time(19, 0), datetime(2026, 10, 1), DUBAI)
