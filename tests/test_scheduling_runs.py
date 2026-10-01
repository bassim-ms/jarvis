"""
Slice 1 - scheduling foundation: run planning (on time / late within grace / missed, collapsed - never replayed) and the
occurrence identity that makes duplicate execution impossible, including across a daylight-saving overlap and a restart.
"""

import json
from datetime import datetime, time, timedelta, timezone

import pytest

from app.reminders.base import Recurrence, RecurrenceUnit
from app.scheduling.runs import (
    DEFAULT_GRACE,
    MAX_MISSED_COUNT,
    RunDecision,
    occurrence_key,
    plan_run,
)
from app.scheduling.schedule import Schedule, ScheduleError, next_due
from app.scheduling.timezones import UTC, load_zone

DUBAI, NYC = "Asia/Dubai", "America/New_York"
DAILY_7PM = Schedule.daily(DUBAI, time(19, 0))  # 15:00 UTC


def utc(*args, **kwargs):
    return datetime(*args, tzinfo=UTC, **kwargs)


# ====================================== plan_run ==========================================================================


def test_before_the_due_time_nothing_is_due():
    due = utc(2026, 10, 1, 15, 0)
    plan = plan_run(DAILY_7PM, due, utc(2026, 10, 1, 14, 59, 59))
    assert plan.decision is RunDecision.NOT_DUE and plan.lateness == timedelta(0) and plan.next_due == due


def test_on_time_and_late_within_the_grace_window_both_simply_run():
    due = utc(2026, 10, 1, 15, 0)
    on_time = plan_run(DAILY_7PM, due, due)
    assert (on_time.decision, on_time.lateness, on_time.scheduled_for, on_time.next_due) == (RunDecision.RUN, timedelta(0), due, utc(2026, 10, 2, 15, 0))
    late = plan_run(DAILY_7PM, due, utc(2026, 10, 1, 15, 7))
    assert late.decision is RunDecision.RUN and late.lateness == timedelta(minutes=7) and late.next_due == utc(2026, 10, 2, 15, 0)


def test_the_grace_boundary_is_exact():
    due = utc(2026, 10, 1, 15, 0)
    assert DEFAULT_GRACE == timedelta(minutes=10)
    assert plan_run(DAILY_7PM, due, due + DEFAULT_GRACE).decision is RunDecision.RUN  # exactly at the limit
    assert plan_run(DAILY_7PM, due, due + DEFAULT_GRACE + timedelta(microseconds=1)).decision is RunDecision.MISSED
    assert plan_run(DAILY_7PM, due, due + timedelta(minutes=30), grace=timedelta(hours=1)).decision is RunDecision.RUN  # a different window
    assert plan_run(DAILY_7PM, due, due, grace=timedelta(0)).decision is RunDecision.RUN


def test_three_days_offline_is_one_missed_verdict_with_the_facts_and_never_a_replay():
    due = utc(2026, 9, 28, 15, 0)  # Monday 19:00 Dubai
    now = utc(2026, 10, 1, 4, 0)  # Thursday 08:00 Dubai - Jarvis was off for three evenings
    plan = plan_run(DAILY_7PM, due, now)
    assert plan.decision is RunDecision.MISSED and plan.scheduled_for == due
    assert plan.missed_count == 3  # Monday, Tuesday, Wednesday
    assert plan.latest_missed == utc(2026, 9, 30, 15, 0)  # the most recent one, for a single catch-up notice
    assert plan.next_due == utc(2026, 10, 1, 15, 0)  # and the next FUTURE occurrence: tonight


def test_handling_a_verdict_then_planning_again_is_not_due_so_nothing_runs_twice():
    now = utc(2026, 10, 1, 4, 0)
    first = plan_run(DAILY_7PM, utc(2026, 9, 28, 15, 0), now)
    again = plan_run(DAILY_7PM, first.next_due, now)  # the store now holds `next_due` as the new due time
    assert again.decision is RunDecision.NOT_DUE and again.next_due == first.next_due
    assert plan_run(DAILY_7PM, utc(2026, 9, 28, 15, 0), now) == first  # and the same inputs always give the same verdict


def test_a_one_time_schedule_that_is_missed_is_finished_after_the_single_catch_up():
    once = Schedule.once(DUBAI, datetime(2026, 10, 1, 19, 0))
    due = next_due(once, utc(2026, 9, 1))
    missed = plan_run(once, due, utc(2026, 10, 3))
    assert missed.decision is RunDecision.MISSED and missed.missed_count == 1 and missed.latest_missed == due and missed.next_due is None
    assert plan_run(once, due, due).next_due is None  # even on time there is nothing after it


def test_a_weekly_schedule_collapses_missed_weeks_and_skips_the_weekend_for_weekdays():
    weekly = Schedule.weekly(DUBAI, time(18, 0), (1,))  # Mondays 14:00 UTC
    plan = plan_run(weekly, utc(2026, 9, 7, 14, 0), utc(2026, 10, 1, 0, 0))  # Mondays 7, 14, 21, 28 September
    assert plan.missed_count == 4 and plan.latest_missed == utc(2026, 9, 28, 14, 0) and plan.next_due == utc(2026, 10, 5, 14, 0)
    weekdays = Schedule.weekdays(DUBAI, time(19, 0))
    plan = plan_run(weekdays, utc(2026, 10, 2, 15, 0), utc(2026, 10, 6, 0, 0))  # Fri 2 Oct due; seen Tue 6 Oct 04:00 local
    assert plan.missed_count == 2 and plan.latest_missed == utc(2026, 10, 5, 15, 0)  # Friday and Monday - not the weekend


def test_an_interval_schedule_counts_missed_slots_by_arithmetic_with_no_loop():
    every = Schedule.every(DUBAI, Recurrence(RecurrenceUnit.MINUTES, 30), utc(2026, 10, 1, 10, 0))
    plan = plan_run(every, utc(2026, 10, 1, 10, 0), utc(2026, 10, 1, 12, 20))
    assert plan.decision is RunDecision.MISSED and plan.missed_count == 5  # 10:00, 10:30, 11:00, 11:30, 12:00
    assert plan.latest_missed == utc(2026, 10, 1, 12, 0) and plan.next_due == utc(2026, 10, 1, 12, 30)
    huge = plan_run(every, utc(2026, 10, 1, 10, 0), utc(2030, 1, 1))
    assert huge.missed_count == MAX_MISSED_COUNT  # capped: years offline is still one verdict


def test_a_daily_schedule_offline_for_years_is_still_one_bounded_verdict():
    plan = plan_run(DAILY_7PM, utc(2024, 1, 1, 15, 0), utc(2026, 10, 1, 4, 0))
    assert plan.decision is RunDecision.MISSED and plan.missed_count == MAX_MISSED_COUNT
    assert plan.latest_missed == utc(2026, 9, 30, 15, 0) and plan.next_due == utc(2026, 10, 1, 15, 0)  # still exact


def test_plan_run_refuses_naive_times_and_a_bad_grace():
    due = utc(2026, 10, 1, 15, 0)
    for args in ((due, datetime(2026, 10, 1, 15, 1)), (datetime(2026, 10, 1, 15, 0), due)):
        with pytest.raises(ScheduleError):
            plan_run(DAILY_7PM, *args)
    with pytest.raises(ScheduleError):
        plan_run(DAILY_7PM, due, due, grace=timedelta(seconds=-1))


def test_inputs_in_other_zones_are_normalised():
    due = datetime(2026, 10, 1, 8, 0, tzinfo=load_zone("America/Los_Angeles"))  # 15:00 UTC
    now = datetime(2026, 10, 1, 19, 3, tzinfo=load_zone(DUBAI))  # 15:03 UTC
    plan = plan_run(DAILY_7PM, due, now)
    assert plan.decision is RunDecision.RUN and plan.lateness == timedelta(minutes=3) and plan.scheduled_for == utc(2026, 10, 1, 15, 0)


# ====================================== restart: a stored item is just (schedule JSON, due_at) ===========================


def test_after_a_restart_the_stored_schedule_gives_the_same_verdict_as_the_one_in_memory():
    stored = json.dumps({"schedule": DAILY_7PM.to_dict(), "due_at": utc(2026, 9, 28, 15, 0).isoformat()})
    revived = json.loads(stored)  # the process restarted: only what was persisted remains
    schedule, due_at = Schedule.from_dict(revived["schedule"]), datetime.fromisoformat(revived["due_at"])
    now = utc(2026, 10, 1, 4, 0)
    assert plan_run(schedule, due_at, now) == plan_run(DAILY_7PM, utc(2026, 9, 28, 15, 0), now)


# ====================================== duplicate prevention: identity by the resolved instant ===========================


def test_the_key_is_owner_and_the_utc_instant_to_the_second():
    assert occurrence_key("abc123", utc(2026, 10, 1, 15, 0)) == "abc123@2026-10-01T15:00:00Z"
    assert occurrence_key("abc123", utc(2026, 10, 1, 15, 0, 0, microsecond=999_999)) == "abc123@2026-10-01T15:00:00Z"
    assert occurrence_key("abc123", utc(2026, 10, 1, 15, 0)) != occurrence_key("abc124", utc(2026, 10, 1, 15, 0))
    assert occurrence_key("abc123", utc(2026, 10, 1, 15, 0)) != occurrence_key("abc123", utc(2026, 10, 2, 15, 0))


def test_the_same_instant_has_the_same_key_whatever_zone_it_is_written_in():
    instant = utc(2026, 10, 1, 15, 0)
    keys = {occurrence_key("abc", instant.astimezone(load_zone(tz))) for tz in (DUBAI, NYC, "Europe/London", "Australia/Lord_Howe", "UTC")}
    assert keys == {"abc@2026-10-01T15:00:00Z"}


@pytest.mark.parametrize("owner", ["", None, 5, "a@b", "line\nbreak", "x" * 129])
def test_a_bad_owner_id_is_refused(owner):
    with pytest.raises(ScheduleError):
        occurrence_key(owner, utc(2026, 10, 1, 15, 0))
    with pytest.raises(ScheduleError):
        occurrence_key("abc", datetime(2026, 10, 1, 15, 0))


def test_a_daylight_saving_overlap_yields_one_occurrence_one_key_one_run():
    daily = Schedule.daily(NYC, time(1, 30))  # 01:30 happens twice on 2026-11-01 (05:30Z EDT and 06:30Z EST)
    keys, cursor = [], utc(2026, 10, 30, 12, 0)
    for _ in range(3):
        cursor = next_due(daily, cursor)
        keys.append(occurrence_key("rem1", cursor))
    assert keys == ["rem1@2026-10-31T05:30:00Z", "rem1@2026-11-01T05:30:00Z", "rem1@2026-11-02T06:30:00Z"]
    assert "rem1@2026-11-01T06:30:00Z" not in keys  # the second 01:30 never becomes an occurrence, so it can never run twice
    # the first 01:30 (05:30Z) runs on time and the store moves on to the next day; the clock then reads 01:30 AGAIN (06:30Z, EST) -
    # and, even after a restart in that hour, nothing is due: the second 01:30 is not resurrected
    handled = plan_run(daily, utc(2026, 11, 1, 5, 30), utc(2026, 11, 1, 5, 30))
    assert handled.decision is RunDecision.RUN and handled.next_due == utc(2026, 11, 2, 6, 30)
    second_pass = plan_run(daily, handled.next_due, utc(2026, 11, 1, 6, 30))
    assert second_pass.decision is RunDecision.NOT_DUE and second_pass.next_due == utc(2026, 11, 2, 6, 30)


def test_a_spring_forward_gap_still_yields_a_single_well_defined_occurrence():
    daily = Schedule.daily(NYC, time(2, 30))
    occurrence = next_due(daily, utc(2026, 3, 7, 12, 0))
    assert occurrence_key("rem1", occurrence) == "rem1@2026-03-08T07:00:00Z"  # resolved to 03:00 EDT
    assert next_due(daily, occurrence) == utc(2026, 3, 9, 6, 30)
