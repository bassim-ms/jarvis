"""
The shared request -> Schedule builder (app/scheduling/builder.py), moved out of ReminderService with NO behaviour change so that a
Scheduled Job is created by exactly the same rules as a reminder. The reminder test files are the main guard for "no behaviour
change" (they run through ReminderService, which delegates here); these tests exercise the builder directly and check that the only
thing a caller can vary - the noun in a refusal - varies and nothing else.
"""

from datetime import datetime, time, timedelta, timezone

import pytest

from app.reminders.language import parse_clock
from app.scheduling.builder import (
    MAX_HORIZON,
    REMINDER_WORDING,
    ScheduleSpec,
    ScheduleSpecError,
    Wording,
    ask_meridiem,
    build_schedule,
    needs_meridiem,
)
from app.scheduling.schedule import Schedule, ScheduleError, ScheduleKind

UTC = timezone.utc
DUBAI = "Asia/Dubai"
#: Thursday 1 October 2026, 14:00 in Dubai (UTC+4)
NOW = datetime(2026, 10, 1, 10, 0, tzinfo=UTC)
#: Friday 2 October 2026, 10:00 in Dubai
FRIDAY = datetime(2026, 10, 2, 6, 0, tzinfo=UTC)
MIN = timedelta(minutes=15)


def build(now=NOW, tz=DUBAI, wording=REMINDER_WORDING, min_interval=MIN, **fields):
    return build_schedule(ScheduleSpec(**fields), tz, now, min_interval=min_interval, wording=wording)


def at(text):
    return parse_clock(text)


def test_tomorrow_at_a_time_is_a_one_time_schedule_in_the_given_zone():
    schedule, due, rolled = build(clock=at("7 pm"), day="tomorrow")
    assert schedule.kind is ScheduleKind.ONCE and schedule.tz == DUBAI and not rolled
    assert due == datetime(2026, 10, 2, 15, 0, tzinfo=UTC)


def test_a_bare_hour_alone_means_the_next_time_the_clock_reads_it():
    schedule, due, rolled = build(clock=at("7"))
    assert due == datetime(2026, 10, 1, 15, 0, tzinfo=UTC)  # it is 2 PM: 7 AM already passed today, so the next 7 is 7 PM
    assert schedule.kind is ScheduleKind.ONCE


@pytest.mark.parametrize("fields", [{"day": "tomorrow"}, {"repeat": "daily"}])
def test_a_bare_hour_with_a_day_or_a_repetition_asks_am_or_pm(fields):
    with pytest.raises(ScheduleSpecError, match="AM or PM"):
        build(clock=at("7"), **fields)
    assert needs_meridiem(at("7"), fields.get("day"), fields.get("repeat"))


def test_the_meridiem_question_names_both_readings():
    assert ask_meridiem(at("7")).startswith("Do you mean 7:00 AM or 7:00 PM?")
    assert not needs_meridiem(at("7 pm"), "tomorrow", None) and not needs_meridiem(at("7"), None, None)


def test_every_weekday_every_day_and_chosen_days():
    schedule, due, _ = build(clock=at("6 pm"), repeat="weekdays")
    assert schedule.kind is ScheduleKind.WEEKDAYS and due == datetime(2026, 10, 1, 14, 0, tzinfo=UTC)
    schedule, _, _ = build(clock=at("07:00"), repeat="daily")
    assert schedule.kind is ScheduleKind.DAILY
    schedule, due, _ = build(clock=at("5 pm"), repeat="weekly", weekdays=(1, 3))
    assert schedule.kind is ScheduleKind.WEEKLY and schedule.days == (1, 3)
    assert due == datetime(2026, 10, 5, 13, 0, tzinfo=UTC)  # the next Monday, 5 PM Dubai


def test_a_repetition_needs_a_time_and_cannot_also_have_a_day_and_weekly_needs_days():
    with pytest.raises(ScheduleSpecError, match="What time should it repeat at"):
        build(repeat="daily")
    with pytest.raises(ScheduleSpecError, match="can't also be for one particular day"):
        build(clock=at("7 pm"), repeat="daily", day="tomorrow")
    with pytest.raises(ScheduleSpecError, match="Which day of the week"):
        build(clock=at("7 pm"), repeat="weekly")
    with pytest.raises(ScheduleSpecError, match="daily, on weekdays"):
        build(clock=at("7 pm"), repeat="fortnightly")


def test_an_elapsed_interval_starts_one_interval_from_now_or_after_first_in():
    schedule, due, _ = build(every=("hours", 2))
    assert schedule.kind is ScheduleKind.INTERVAL and due == NOW + timedelta(hours=2)
    _, due, _ = build(every=("hours", 2), first_in=timedelta(minutes=5))
    assert due == NOW + timedelta(minutes=5)


def test_a_new_interval_shorter_than_the_minimum_is_refused_by_the_schedule_layer():
    with pytest.raises(ScheduleError, match="at least 15 minutes"):
        build(every=("minutes", 5))
    assert build(every=("minutes", 5), min_interval=timedelta(minutes=1))[0].interval.interval == 5
    # a Scheduled Job asks for a longer minimum: the same builder, a different limit
    with pytest.raises(ScheduleError, match="at least 60 minutes"):
        build(every=("minutes", 30), min_interval=timedelta(minutes=60))


def test_an_interval_cannot_also_have_a_time_or_day_and_the_unit_must_be_known():
    with pytest.raises(ScheduleSpecError, match="can't also have a fixed time or day"):
        build(every=("hours", 2), clock=at("7 pm"))
    with pytest.raises(ScheduleSpecError, match="not 'fortnights'"):
        build(every=("fortnights", 1))


def test_day_words():
    assert build(clock=at("7 pm"), day="today")[1] == datetime(2026, 10, 1, 15, 0, tzinfo=UTC)
    assert build(clock=at("7 pm"), day="+3")[1] == datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
    assert build(clock=at("7 pm"), day="2026-12-25")[1] == datetime(2026, 12, 25, 15, 0, tzinfo=UTC)
    assert build(clock=at("7 pm"), day="12-25")[1] == datetime(2026, 12, 25, 15, 0, tzinfo=UTC)
    assert build(clock=at("7 pm"), day="friday")[1] == datetime(2026, 10, 2, 15, 0, tzinfo=UTC)


def test_next_friday_is_strictly_the_next_one():
    assert build(now=FRIDAY, clock=at("7 pm"), day="friday")[1] == datetime(2026, 10, 2, 15, 0, tzinfo=UTC)  # still ahead today
    assert build(now=FRIDAY, clock=at("7 pm"), day="next friday")[1] == datetime(2026, 10, 9, 15, 0, tzinfo=UTC)


def test_a_time_that_has_passed_today_is_refused_with_a_reason():
    with pytest.raises(ScheduleSpecError, match="has already passed today"):
        build(clock=at("9 am"), day="today")
    with pytest.raises(ScheduleSpecError, match="has already passed that date"):
        build(clock=at("9 am"), day="2026-10-01")


@pytest.mark.parametrize("day, message", [("+0", "up to a year ahead"), ("+400", "up to a year ahead"), ("someday", "couldn't understand the day")])
def test_unusable_day_words_are_refused(day, message):
    with pytest.raises(ScheduleSpecError, match=message):
        build(clock=at("7 pm"), day=day)


def test_no_time_given_asks_the_callers_own_question():
    with pytest.raises(ScheduleSpecError, match="What time should I remind you"):
        build(day="tomorrow")
    custom = Wording(subject="scheduled job", when_question="When should I do it?")
    with pytest.raises(ScheduleSpecError, match="When should I do it"):
        build(day="tomorrow", wording=custom)


def test_the_noun_in_a_refusal_is_the_only_caller_specific_word():
    custom = Wording(subject="scheduled job", when_question="When?")
    with pytest.raises(ScheduleSpecError) as reminder:
        build(every=("hours", 2), day="tomorrow")
    with pytest.raises(ScheduleSpecError) as other:
        build(every=("hours", 2), day="tomorrow", wording=custom)
    assert str(reminder.value) == "A reminder that repeats every so often can't also have a fixed time or day."
    assert str(other.value) == str(reminder.value).replace("reminder", "scheduled job")


def test_the_reminder_wording_is_exactly_what_reminder_v2_said():
    assert REMINDER_WORDING.subject == "reminder"
    assert REMINDER_WORDING.when_question == "What time should I remind you? For example: tomorrow at 7 PM."


def test_the_builder_is_deterministic_and_reads_no_clock():
    first = build(clock=at("6 pm"), repeat="weekdays")
    assert first == build(clock=at("6 pm"), repeat="weekdays")
    assert build(now=NOW + timedelta(days=1), clock=at("6 pm"), repeat="weekdays")[1] != first[1]


def test_the_horizon_is_the_one_reminders_have_always_had():
    assert MAX_HORIZON == timedelta(days=800)
    from app.reminders import service

    assert service.MAX_HORIZON is MAX_HORIZON  # one definition, still importable from where it used to live
    assert service.ask_meridiem is ask_meridiem and service.needs_meridiem is needs_meridiem
