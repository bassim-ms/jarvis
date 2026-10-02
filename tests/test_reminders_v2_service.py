"""
Reminder V2 - ReminderService and the store underneath it: absolute / daily / weekday / weekly / interval reminders, the time zone
each one keeps, daylight saving (through the scheduling foundation), pause / resume / delete, duplicate prevention, restart
persistence, and that reminders created before V2 still load and behave as they did.
"""

import json
import sqlite3
from datetime import datetime, time, timedelta, timezone

import pytest

from app.reminders.base import Recurrence, RecurrenceUnit, ReminderStatus
from app.reminders.language import parse_clock
from app.reminders.service import ReminderRequest, ReminderRequestError, ReminderService
from app.reminders.sqlite_store import SQLiteReminderStore
from app.scheduling.runs import occurrence_key
from app.scheduling.schedule import Schedule, ScheduleKind
from app.scheduling.timezones import ZoneChoice

UTC = timezone.utc
DUBAI, NYC = "Asia/Dubai", "America/New_York"
#: Thursday 1 October 2026, 14:00 in Dubai (UTC+4, no daylight saving)
NOW = datetime(2026, 10, 1, 10, 0, tzinfo=UTC)


class Clock:
    def __init__(self, now=NOW):
        self.now = now

    def __call__(self):
        return self.now

    def advance(self, **kw):
        self.now += timedelta(**kw)


def make(store=None, tz=DUBAI, clock=None, **kw):
    store = store or SQLiteReminderStore(":memory:")
    clock = clock or Clock()
    return ReminderService(store, clock=clock, zone_resolver=lambda: ZoneChoice(tz, "configured"), **kw), store, clock


def at(text):
    return parse_clock(text)


def req(text="study", clock=None, **kw):
    return ReminderRequest(text=text, clock=at(clock) if isinstance(clock, str) else clock, **kw)


# ====================================== absolute one-time ==================================================================


def test_tomorrow_at_7pm_is_stored_as_local_time_in_the_users_zone():
    service, store, _ = make()
    created = service.create(req(clock="7 pm", day="tomorrow"))
    assert created.due == datetime(2026, 10, 2, 15, 0, tzinfo=UTC)  # 19:00 Dubai
    stored = store.get(created.reminder.id)
    assert stored.due_at == "2026-10-02T15:00:00+00:00" and stored.status is ReminderStatus.PENDING
    assert stored.schedule.kind is ScheduleKind.ONCE and stored.timezone == DUBAI
    assert stored.recurrence is None
    assert stored.schedule.describe() == "on Friday 2 October 2026 at 7:00 PM (Asia/Dubai)"


def test_today_later_is_fine_and_today_already_past_is_refused_not_rolled():
    service, store, _ = make()
    assert service.create(req("call Mom", clock="8 pm", day="today")).due == datetime(2026, 10, 1, 16, 0, tzinfo=UTC)
    with pytest.raises(ReminderRequestError, match="already passed today"):
        service.create(req("early", clock="1 pm", day="today"))
    assert len(store.list()) == 1  # nothing was created by the refused request


def test_a_bare_time_with_no_day_is_the_next_time_the_clock_reads_that():
    service, _, clock = make()  # 14:00 Dubai
    created = service.create(req("stretch", clock=at("7")))
    assert created.due == datetime(2026, 10, 1, 15, 0, tzinfo=UTC) and not created.rolled  # 7 PM today
    clock.now = datetime(2026, 10, 1, 16, 30, tzinfo=UTC)  # 20:30 Dubai: 7 PM has gone, 7 AM is next
    again = service.create(req("stretch more", clock=at("7")))
    assert again.due == datetime(2026, 10, 2, 3, 0, tzinfo=UTC) and again.rolled  # 07:00 tomorrow


def test_a_bare_time_with_an_explicit_day_or_repetition_asks_for_am_or_pm_instead_of_guessing():
    service, store, _ = make()
    for request in (req(clock=at("7"), day="tomorrow"), req(clock=at("7"), repeat="daily")):
        with pytest.raises(ReminderRequestError, match="7:00 AM or 7:00 PM"):
            service.create(request)
    assert store.list() == []


@pytest.mark.parametrize(
    "day, expected_utc",
    [("tomorrow", (2026, 10, 2)), ("friday", (2026, 10, 2)), ("monday", (2026, 10, 5)), ("thursday", (2026, 10, 1)),  # today: 5 PM is still ahead
     ("+3", (2026, 10, 4)), ("2026-12-25", (2026, 12, 25)), ("12-25", (2026, 12, 25)), ("01-05", (2027, 1, 5))],
)
def test_days_resolve_in_the_users_zone(day, expected_utc):
    service, _, _ = make()
    created = service.create(req(clock="5 pm", day=day))  # 5 PM Dubai is 13:00 UTC
    assert created.due == datetime(*expected_utc, 13, 0, tzinfo=UTC)


def test_next_weekday_is_strictly_after_today_and_this_weekday_may_be_today():
    service, _, _ = make()  # Thursday 14:00 Dubai
    assert service.create(req("a", clock="5 pm", day="thursday")).due.day == 1  # later today
    assert service.create(req("b", clock="5 pm", day="next thursday")).due.day == 8
    assert service.create(req("c", clock="9 am", day="thursday")).due.day == 8  # 9 AM today has passed: next week


@pytest.mark.parametrize("day", ["2026-09-30", "someday", "2026-13-40", "+0", "+400", "yesterday"])
def test_a_bad_or_past_day_is_refused(day):
    service, store, _ = make()
    with pytest.raises(ReminderRequestError):
        service.create(req(clock="5 pm", day=day))
    assert store.list() == []


def test_more_than_two_years_ahead_is_refused():
    service, _, _ = make()
    with pytest.raises(ReminderRequestError, match="two years"):
        service.create(req(clock="5 pm", day="2030-01-01"))


# ====================================== repeating ===========================================================================


def test_daily_reminder():
    service, store, _ = make()
    created = service.create(req("study", clock="7 pm", repeat="daily"))
    assert created.due == datetime(2026, 10, 1, 15, 0, tzinfo=UTC)  # tonight
    assert store.get(created.reminder.id).schedule.kind is ScheduleKind.DAILY
    assert "every day at 7:00 PM (Asia/Dubai)" in service.schedule_words(created.reminder)


def test_a_daily_reminder_whose_time_has_passed_starts_tomorrow():
    service, _, _ = make()
    assert service.create(req("study", clock="9 am", repeat="daily")).due == datetime(2026, 10, 2, 5, 0, tzinfo=UTC)


def test_weekday_reminder_skips_the_weekend():
    clock = Clock(datetime(2026, 10, 2, 16, 0, tzinfo=UTC))  # Friday 20:00 Dubai, after 6 PM
    service, store, _ = make(clock=clock)
    created = service.create(req("robotics", clock="6 pm", repeat="weekdays"))
    assert created.due == datetime(2026, 10, 5, 14, 0, tzinfo=UTC)  # Monday 6 PM
    assert store.get(created.reminder.id).schedule.kind is ScheduleKind.WEEKDAYS


def test_weekly_reminder_on_chosen_days():
    service, store, _ = make()
    created = service.create(req("project", clock="5 pm", repeat="weekly", weekdays=(1, 4)))  # Mondays and Thursdays
    assert created.due == datetime(2026, 10, 1, 13, 0, tzinfo=UTC)  # this Thursday, 5 PM is still ahead
    assert store.get(created.reminder.id).schedule.days == (1, 4)
    with pytest.raises(ReminderRequestError, match="Which day"):
        service.create(req("x", clock="5 pm", repeat="weekly"))


def test_a_repeating_reminder_needs_a_time_and_cannot_also_have_a_day():
    service, _, _ = make()
    with pytest.raises(ReminderRequestError, match="What time"):
        service.create(req("x", repeat="daily"))
    with pytest.raises(ReminderRequestError, match="one particular day"):
        service.create(req("x", clock="7 pm", repeat="daily", day="tomorrow"))
    with pytest.raises(ReminderRequestError, match="daily, on weekdays"):
        service.create(req("x", clock="7 pm", repeat="monthly"))


def test_interval_reminders_use_the_existing_interval_vocabulary_and_the_minimum_gap():
    service, store, _ = make()
    created = service.create(ReminderRequest(text="drink water", every=("hours", 2)))
    assert created.due == NOW + timedelta(hours=2)
    stored = store.get(created.reminder.id)
    assert stored.schedule.kind is ScheduleKind.INTERVAL and stored.schedule.interval == Recurrence(RecurrenceUnit.HOURS, 2)
    with pytest.raises(ReminderRequestError, match="at least 15 minutes"):
        service.create(ReminderRequest(text="blink", every=("minutes", 5)))
    relaxed, _, _ = make(min_interval=timedelta(minutes=1))
    assert relaxed.create(ReminderRequest(text="blink", every=("minutes", 5))).schedule.interval.interval == 5
    with pytest.raises(ReminderRequestError):
        service.create(ReminderRequest(text="x", every=("fortnights", 1)))
    with pytest.raises(ReminderRequestError, match="can't also"):
        service.create(ReminderRequest(text="x", every=("hours", 1), clock=at("7 pm")))


# ====================================== time zones ==========================================================================


def test_each_reminder_keeps_the_zone_it_was_created_in_whatever_changes_later():
    store = SQLiteReminderStore(":memory:")
    clock = Clock()
    dubai, _, _ = make(store, DUBAI, clock)
    first = dubai.create(req("study", clock="7 pm", repeat="daily")).reminder
    new_york, _, _ = make(store, NYC, clock)  # the user moves / the setting changes
    second = new_york.create(req("call", clock="7 pm", repeat="daily")).reminder
    assert store.get(first.id).timezone == DUBAI and store.get(second.id).timezone == NYC
    assert store.get(first.id).due_at == "2026-10-01T15:00:00+00:00"  # still 7 PM Dubai
    assert store.get(second.id).due_at == "2026-10-01T23:00:00+00:00"  # 7 PM New York (EDT)
    assert "(Asia/Dubai)" in dubai.schedule_words(store.get(first.id))


def test_no_usable_zone_fails_clearly_and_creates_nothing():
    from app.scheduling.timezones import TimezoneUnavailableError

    def unavailable():
        raise TimezoneUnavailableError("No time zone is configured and the system time zone could not be determined reliably.")

    store = SQLiteReminderStore(":memory:")
    service = ReminderService(store, clock=Clock(), zone_resolver=unavailable)
    with pytest.raises(ReminderRequestError, match="JARVIS_TIMEZONE|reliably"):
        service.create(req(clock="7 pm", repeat="daily"))
    assert store.list() == []


def test_a_misconfigured_zone_falls_back_but_says_so_in_the_reply():
    store = SQLiteReminderStore(":memory:")
    warning = "JARVIS_TIMEZONE='Not/AZone' is not a usable time zone; using the system time zone 'Asia/Dubai' instead."
    service = ReminderService(store, clock=Clock(), zone_resolver=lambda: ZoneChoice(DUBAI, "system", warning))
    created = service.create(req(clock="7 pm", day="tomorrow"))
    assert created.warning == warning and created.reminder.timezone == DUBAI


def test_the_default_zone_comes_from_the_setting_then_the_system(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "timezone", "Europe/London")
    service = ReminderService(SQLiteReminderStore(":memory:"), clock=Clock())
    assert service.zone().name == "Europe/London" and service.zone().source == "configured"
    monkeypatch.setattr(settings, "timezone", "")
    monkeypatch.setattr("app.system_timezone.detect_system_timezone", lambda: None)
    with pytest.raises(ReminderRequestError):
        service.zone()


# ====================================== daylight saving (through app/scheduling) ===============================================


def test_a_time_inside_the_spring_forward_gap_resolves_to_the_first_valid_minute_after_it():
    clock = Clock(datetime(2026, 3, 7, 12, 0, tzinfo=UTC))
    service, _, _ = make(tz=NYC, clock=clock)
    created = service.create(req("early", clock="2:30 am", day="2026-03-08"))
    assert created.due == datetime(2026, 3, 8, 7, 0, tzinfo=UTC)  # 03:00 EDT, the first valid time after the gap


def test_a_time_that_happens_twice_resolves_to_its_first_occurrence():
    clock = Clock(datetime(2026, 10, 31, 12, 0, tzinfo=UTC))
    service, _, _ = make(tz=NYC, clock=clock)
    created = service.create(req("late", clock="1:30 am", day="2026-11-01"))
    assert created.due == datetime(2026, 11, 1, 5, 30, tzinfo=UTC)  # 01:30 EDT, not the repeat at 06:30Z


def test_a_daily_reminder_stays_at_the_same_local_time_across_a_clock_change():
    from zoneinfo import ZoneInfo

    from app.scheduling.schedule import next_due

    clock = Clock(datetime(2026, 10, 31, 12, 0, tzinfo=UTC))  # the day before the clocks go back in New York
    service, store, _ = make(tz=NYC, clock=clock)
    created = service.create(req("study", clock="7 pm", repeat="daily"))
    assert created.due == datetime(2026, 10, 31, 23, 0, tzinfo=UTC)  # 7 PM EDT (UTC-4)
    following = next_due(created.reminder.schedule, created.due)
    assert following == datetime(2026, 11, 2, 0, 0, tzinfo=UTC)  # 7 PM EST (UTC-5): 25 hours later, the same wall-clock time
    assert [t.astimezone(ZoneInfo(NYC)).hour for t in (created.due, following)] == [19, 19]


# ====================================== pause, resume, delete ================================================================


def test_pause_stops_a_reminder_from_ever_coming_due():
    service, store, clock = make()
    created = service.create(req("study", clock="7 pm", repeat="daily"))
    service.pause(created.reminder.id)
    assert store.get(created.reminder.id).status is ReminderStatus.PAUSED
    assert store.list_due((clock.now + timedelta(days=30)).isoformat()) == []
    with pytest.raises(ReminderRequestError, match="already paused"):
        service.pause(created.reminder.id)


def test_resume_goes_to_the_next_future_occurrence_and_never_replays_what_was_missed():
    service, store, clock = make()
    created = service.create(req("study", clock="7 pm", repeat="daily"))  # next: tonight 19:00 Dubai (15:00Z)
    service.pause(created.reminder.id)
    clock.now = datetime(2026, 10, 4, 18, 0, tzinfo=UTC)  # paused for three days; it is now 22:00 Dubai
    reminder, upcoming = service.resume(created.reminder.id)
    assert upcoming == datetime(2026, 10, 5, 15, 0, tzinfo=UTC)  # tomorrow's 7 PM, not any of the three that were skipped
    after = store.get(created.reminder.id)
    assert after.status is ReminderStatus.PENDING and after.due_at == "2026-10-05T15:00:00+00:00"
    assert store.list_due(clock.now.isoformat()) == []


def test_a_paused_one_time_reminder_whose_time_passed_cannot_be_resumed():
    service, store, clock = make()
    created = service.create(req("call", clock="8 pm", day="today"))
    service.pause(created.reminder.id)
    clock.now = datetime(2026, 10, 1, 20, 0, tzinfo=UTC)
    with pytest.raises(ReminderRequestError, match="already passed"):
        service.resume(created.reminder.id)
    assert store.get(created.reminder.id).status is ReminderStatus.PAUSED  # unchanged: the user can delete it


def test_resume_only_applies_to_a_paused_reminder():
    service, _, _ = make()
    created = service.create(req("study", clock="7 pm", repeat="daily"))
    with pytest.raises(ReminderRequestError, match="isn't paused"):
        service.resume(created.reminder.id)
    with pytest.raises(ReminderRequestError, match="couldn't find"):
        service.resume("nope")
    with pytest.raises(ReminderRequestError, match="couldn't find"):
        service.pause("")


def test_delete_removes_it_from_the_active_list_and_from_ever_firing():
    service, store, clock = make()
    created = service.create(req("study", clock="7 pm", repeat="daily"))
    other = service.create(req("other", clock="8 pm", repeat="daily"))
    service.delete(created.reminder.id)
    assert store.get(created.reminder.id).status is ReminderStatus.CANCELLED
    assert [r.id for r in service.active()] == [other.reminder.id]
    assert store.list_due((clock.now + timedelta(days=9)).isoformat())[0].id == other.reminder.id
    with pytest.raises(ReminderRequestError, match="isn't an active reminder"):
        service.delete(created.reminder.id)


def test_a_paused_reminder_can_be_deleted():
    service, store, _ = make()
    created = service.create(req("study", clock="7 pm", repeat="daily"))
    service.pause(created.reminder.id)
    service.delete(created.reminder.id)
    assert store.get(created.reminder.id).status is ReminderStatus.CANCELLED


# ====================================== listing, limits, duplicates ==========================================================


def test_active_lists_pending_and_paused_soonest_first():
    service, store, _ = make()
    late = service.create(req("late", clock="9 pm", day="tomorrow")).reminder
    soon = service.create(req("soon", clock="7 pm", day="today")).reminder
    paused = service.create(req("paused", clock="6 pm", day="today")).reminder
    service.pause(paused.id)
    gone = service.create(req("gone", clock="5 pm", day="today")).reminder
    service.delete(gone.id)
    assert [r.id for r in service.active()] == [paused.id, soon.id, late.id]


def test_creating_the_same_reminder_twice_does_not_make_two():
    service, store, _ = make()
    first = service.create(req("study", clock="7 pm", repeat="daily"))
    second = service.create(req("  Study ", clock="7 pm", repeat="daily"))  # the same words, whatever the case or spacing
    assert second.duplicate and second.reminder.id == first.reminder.id and len(store.list()) == 1
    assert not service.create(req("study", clock="8 pm", repeat="daily")).duplicate  # a different time is a different reminder


def test_the_number_of_active_reminders_is_bounded():
    service, store, _ = make(max_active=3)
    for n in range(3):
        service.create(req(f"r{n}", clock="7 pm", day="tomorrow"))
    with pytest.raises(ReminderRequestError, match="already have 3"):
        service.create(req("one too many", clock="7 pm", day="tomorrow"))
    assert len(store.list()) == 3


def test_text_is_cleaned_and_bounded():
    service, store, _ = make()
    created = service.create(req("study\x00 hard\n tonight‮", clock="7 pm", day="tomorrow"))
    assert created.reminder.text == "study hard tonight"
    for bad in ("", "   ", "\n\x00"):
        with pytest.raises(ReminderRequestError, match="what to remind"):
            service.create(req(bad, clock="7 pm", day="tomorrow"))
    with pytest.raises(ReminderRequestError, match="too long"):
        service.create(req("x" * 301, clock="7 pm", day="tomorrow"))


# ====================================== persistence: a reminder is a row, not a timer ========================================


def test_a_reminder_survives_a_restart_with_its_schedule_zone_and_due_time(tmp_path):
    path = str(tmp_path / "reminders.db")
    service, store, _ = make(SQLiteReminderStore(path))
    created = service.create(req("study", clock="7 pm", repeat="weekdays"))
    schedule, due_at = created.reminder.schedule, created.reminder.due_at
    del service, store  # the process exits

    reopened = SQLiteReminderStore(path)
    loaded = reopened.get(created.reminder.id)
    assert loaded.schedule == schedule and loaded.timezone == DUBAI and loaded.due_at == due_at
    assert loaded.status is ReminderStatus.PENDING and loaded.text == "study"


def test_pause_and_the_next_due_time_persist_too(tmp_path):
    path = str(tmp_path / "reminders.db")
    service, store, _ = make(SQLiteReminderStore(path))
    created = service.create(req("study", clock="7 pm", repeat="daily"))
    service.pause(created.reminder.id)
    reopened = SQLiteReminderStore(path)
    assert reopened.get(created.reminder.id).status is ReminderStatus.PAUSED


# ====================================== reminders created before V2 ==========================================================

OLD_SCHEMA = """
CREATE TABLE reminders (
    id TEXT PRIMARY KEY, text TEXT NOT NULL, due_at TEXT NOT NULL, status TEXT NOT NULL,
    recurrence_unit TEXT, recurrence_interval INTEGER, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE reminder_executions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, reminder_id TEXT NOT NULL, triggered_at TEXT NOT NULL, success INTEGER NOT NULL, detail TEXT NOT NULL
);
"""


def old_database(path):
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA)
    conn.execute("INSERT INTO reminders VALUES ('old1', 'water the plants', '2026-10-03T08:00:00+00:00', 'pending', NULL, NULL, '2026-09-01T00:00:00+00:00', '2026-09-01T00:00:00+00:00')")
    conn.execute("INSERT INTO reminders VALUES ('old2', 'stand up', '2026-10-01T11:00:00+00:00', 'pending', 'hours', 1, '2026-09-01T00:00:00+00:00', '2026-09-01T00:00:00+00:00')")
    conn.execute("INSERT INTO reminders VALUES ('old3', 'old news', '2026-09-02T08:00:00+00:00', 'triggered', NULL, NULL, '2026-09-01T00:00:00+00:00', '2026-09-01T00:00:00+00:00')")
    conn.execute("INSERT INTO reminder_executions (reminder_id, triggered_at, success, detail) VALUES ('old3', '2026-09-02T08:00:00+00:00', 1, '')")
    conn.commit()
    conn.close()


def test_a_database_from_before_v2_is_upgraded_in_place_and_every_old_reminder_loads(tmp_path):
    path = str(tmp_path / "old.db")
    old_database(path)
    store = SQLiteReminderStore(path)

    one_time, recurring, finished = store.get("old1"), store.get("old2"), store.get("old3")
    assert one_time.text == "water the plants" and one_time.schedule is None and one_time.timezone is None and one_time.recurrence is None
    assert recurring.recurrence == Recurrence(RecurrenceUnit.HOURS, 1) and recurring.schedule is None
    assert finished.status is ReminderStatus.TRIGGERED
    assert [e.success for e in store.get_executions("old3")] == [True]
    assert {r.id for r in store.list(ReminderStatus.PENDING)} == {"old1", "old2"}

    columns = {row[1] for row in sqlite3.connect(path).execute("PRAGMA table_info(reminders)")}
    assert "schedule" in columns and "recurrence_unit" in columns  # added, nothing removed
    reopened = SQLiteReminderStore(path)  # upgrading twice is harmless
    assert len(reopened.list()) == 3


def test_old_reminders_are_described_listed_paused_and_resumed_like_any_other(tmp_path):
    path = str(tmp_path / "old.db")
    old_database(path)
    store = SQLiteReminderStore(path)
    service, _, clock = make(store)
    assert "every 1 hour" in service.schedule_words(store.get("old2")) or "every 1 hours" in service.schedule_words(store.get("old2"))
    assert service.schedule_words(store.get("old1")).startswith("at ")
    service.pause("old2")
    clock.now = datetime(2026, 10, 1, 15, 30, tzinfo=UTC)
    _, upcoming = service.resume("old2")  # an old recurring reminder resumes at its next FUTURE hour, via the same machinery
    assert upcoming == datetime(2026, 10, 1, 16, 0, tzinfo=UTC)
    service.delete("old1")
    assert store.get("old1").status is ReminderStatus.CANCELLED


def test_a_damaged_stored_schedule_still_loads_and_is_flagged_never_guessed(tmp_path):
    path = str(tmp_path / "r.db")
    service, store, _ = make(SQLiteReminderStore(path))
    created = service.create(req("study", clock="7 pm", repeat="daily"))
    conn = sqlite3.connect(path)
    conn.execute("UPDATE reminders SET schedule = ? WHERE id = ?", ('{"kind": "yearly"}', created.reminder.id))
    conn.commit()
    conn.close()
    loaded = SQLiteReminderStore(path).get(created.reminder.id)
    assert loaded.schedule is None and loaded.schedule_invalid and loaded.text == "study"


def test_a_reminder_has_either_a_schedule_or_a_legacy_recurrence_never_both():
    store = SQLiteReminderStore(":memory:")
    with pytest.raises(ValueError):
        store.create("x", NOW.isoformat(), Recurrence(RecurrenceUnit.HOURS, 1), Schedule.daily(DUBAI, time(19, 0)))


# ====================================== duplicate prevention in the store =====================================================


def test_the_same_occurrence_cannot_be_recorded_as_delivered_twice():
    store = SQLiteReminderStore(":memory:")
    reminder = store.create("x", "2026-10-01T15:00:00+00:00", schedule=Schedule.daily(DUBAI, time(19, 0)))
    key = occurrence_key(reminder.id, datetime(2026, 10, 1, 15, 0, tzinfo=UTC))
    assert store.record_trigger(reminder.id, NOW.isoformat(), True, next_due_at="2026-10-02T15:00:00+00:00", occurrence=key,
                                expected_due_at="2026-10-01T15:00:00+00:00") is True
    # a second poller that also saw 15:00Z tries to record the very same occurrence
    assert store.record_trigger(reminder.id, NOW.isoformat(), True, next_due_at="2026-10-02T15:00:00+00:00", occurrence=key,
                                expected_due_at="2026-10-01T15:00:00+00:00") is False
    assert store.get(reminder.id).due_at == "2026-10-02T15:00:00+00:00"
    assert len(store.get_executions(reminder.id)) == 1  # the loser left no record


def test_a_stale_view_cannot_advance_a_reminder_that_has_moved_on():
    store = SQLiteReminderStore(":memory:")
    reminder = store.create("x", "2026-10-01T15:00:00+00:00", schedule=Schedule.daily(DUBAI, time(19, 0)))
    store.pause(reminder.id)
    assert store.record_trigger(reminder.id, NOW.isoformat(), True, next_due_at="2026-10-02T15:00:00+00:00",
                                expected_due_at="2026-10-01T15:00:00+00:00") is False  # paused meanwhile: not advanced
    assert store.get(reminder.id).status is ReminderStatus.PAUSED and store.get(reminder.id).due_at == "2026-10-01T15:00:00+00:00"


def test_a_failed_delivery_is_recorded_without_using_up_the_occurrence():
    store = SQLiteReminderStore(":memory:")
    reminder = store.create("x", "2026-10-01T15:00:00+00:00", schedule=Schedule.daily(DUBAI, time(19, 0)))
    key = occurrence_key(reminder.id, datetime(2026, 10, 1, 15, 0, tzinfo=UTC))
    for _ in range(2):
        assert store.record_trigger(reminder.id, NOW.isoformat(), False, "toast down", occurrence=key, expected_due_at=reminder.due_at)
    assert store.get(reminder.id).due_at == reminder.due_at  # still due, so the next poll retries it
    assert store.record_trigger(reminder.id, NOW.isoformat(), True, next_due_at="2026-10-02T15:00:00+00:00", occurrence=key, expected_due_at=reminder.due_at)


def test_the_compare_and_set_alone_stops_a_second_advance():
    store = SQLiteReminderStore(":memory:")
    reminder = store.create("x", "2026-10-01T15:00:00+00:00", schedule=Schedule.daily(DUBAI, time(19, 0)))
    args = dict(next_due_at="2026-10-02T15:00:00+00:00", expected_due_at="2026-10-01T15:00:00+00:00")  # no occurrence given
    assert store.record_trigger(reminder.id, NOW.isoformat(), True, **args) is True
    assert store.record_trigger(reminder.id, NOW.isoformat(), True, **args) is False  # the due time is no longer the expected one
    assert len(store.get_executions(reminder.id)) == 1


def test_the_unique_occurrence_record_alone_stops_a_second_delivery_record():
    store = SQLiteReminderStore(":memory:")
    reminder = store.create("x", "2026-10-01T15:00:00+00:00", schedule=Schedule.daily(DUBAI, time(19, 0)))
    key = occurrence_key(reminder.id, datetime(2026, 10, 1, 15, 0, tzinfo=UTC))
    assert store.record_trigger(reminder.id, NOW.isoformat(), True, next_due_at="2026-10-02T15:00:00+00:00", occurrence=key) is True
    # no compare-and-set this time: the occurrence itself is what is refused, and the advance is rolled back with it
    assert store.record_trigger(reminder.id, NOW.isoformat(), True, next_due_at="2026-10-03T15:00:00+00:00", occurrence=key) is False
    assert store.get(reminder.id).due_at == "2026-10-02T15:00:00+00:00" and len(store.get_executions(reminder.id)) == 1

