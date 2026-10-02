"""
Reminder V2 - the reminder tools: the original arguments still behave exactly as before, the new local-time arguments work, and
the Planner / Handle This boundary is unchanged (pause and resume are user-driven and cannot be planned or automated).
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.handle_this.base import GoalFamily
from app.handle_this.boundary import DEFAULT_BOUNDARY
from app.handle_this.factory import planner_tool_names
from app.orchestration.base import RetryPolicy
from app.orchestration.validator import PlanValidationError, validate_plan
from app.reminders.base import Recurrence, RecurrenceUnit, ReminderStatus
from app.reminders.service import ReminderService
from app.reminders.sqlite_store import SQLiteReminderStore
from app.reminders.tools import (
    CancelReminderTool,
    CompleteReminderTool,
    CreateReminderTool,
    ListRemindersTool,
    PauseReminderTool,
    ResumeReminderTool,
)
from app.scheduling.timezones import ZoneChoice
from app.tools.base import PermissionLevel, ToolResultStatus
from app.tools.decision import ToolCall, ToolCallBuilder
from app.tools.registry import ToolRegistry
from test_reminders_integration import FakeToolCallProvider

UTC = timezone.utc
NOW = datetime(2026, 10, 1, 10, 0, tzinfo=UTC)  # Thursday 14:00 in Dubai


@pytest.fixture
def rig():
    store = SQLiteReminderStore(":memory:")
    service = ReminderService(store, clock=lambda: NOW, zone_resolver=lambda: ZoneChoice("Asia/Dubai", "configured"))
    registry = ToolRegistry()
    for tool in (
        CreateReminderTool(store, service), ListRemindersTool(store, service), CompleteReminderTool(store),
        CancelReminderTool(store), PauseReminderTool(store, service), ResumeReminderTool(store, service),
    ):
        registry.register(tool)

    class Rig:
        pass

    r = Rig()
    r.store, r.service, r.registry = store, service, registry
    return r


def run(rig, name, **params):
    return rig.registry.execute(name, params=params)


# ====================================== the original arguments still work exactly as before ===================================


def test_the_original_relative_reminder_is_unchanged(rig):
    result = run(rig, "create_reminder", text="stretch", due_in_minutes=10)
    assert result.status is ToolResultStatus.SUCCESS
    assert result.data["due_at"] == (NOW + timedelta(minutes=10)).isoformat()
    assert result.data["recurrence_unit"] is None and result.data["schedule"] is None and result.data["timezone"] is None
    assert result.message == "Reminder set: 'stretch' at Thursday 1 October, 2:10 PM."  # words, not a raw timestamp


def test_the_repeating_arguments_still_work_and_new_ones_obey_the_minimum_gap(rig):
    ok = run(rig, "create_reminder", text="water", due_in_minutes=5, recurrence_unit="hours", recurrence_interval=2)
    assert ok.data["recurrence_unit"] == "hours" and ok.data["recurrence_interval"] == 2  # same fields as before
    assert ok.data["due_at"] == (NOW + timedelta(minutes=5)).isoformat()  # the first one is still due_in_minutes from now
    assert "repeats every 2 hours" in ok.message
    quick = run(rig, "create_reminder", text="blink", due_in_minutes=5, recurrence_unit="minutes", recurrence_interval=5)
    assert quick.status is ToolResultStatus.ERROR and "at least 15 minutes" in quick.message
    fifteen = run(rig, "create_reminder", text="posture", due_in_minutes=5, recurrence_unit="minutes", recurrence_interval=15)
    assert fifteen.status is ToolResultStatus.SUCCESS


def test_the_original_validation_messages_are_unchanged(rig):
    assert "must not be negative" in run(rig, "create_reminder", text="x", due_in_minutes=-1).message
    assert "recurrence_interval" in run(rig, "create_reminder", text="x", due_in_minutes=5, recurrence_unit="hours", recurrence_interval=0).message
    assert "Unknown recurrence_unit" in run(rig, "create_reminder", text="x", due_in_minutes=5, recurrence_unit="fortnights").message


def test_without_any_time_it_says_what_is_missing(rig):
    result = run(rig, "create_reminder", text="x")
    assert result.status is ToolResultStatus.ERROR and "Say when" in result.message and rig.store.list() == []


# ====================================== the new arguments ==========================================================================


def test_a_local_time_and_day(rig):
    result = run(rig, "create_reminder", text="study", at="19:00", day="tomorrow")
    assert result.status is ToolResultStatus.SUCCESS
    assert result.data["due_at"] == "2026-10-02T15:00:00+00:00" and result.data["first_due_at"] == "2026-10-02T15:00:00+00:00"
    assert result.data["timezone"] == "Asia/Dubai" and result.data["schedule"] == {"kind": "once", "tz": "Asia/Dubai", "once": "2026-10-02T19:00"}
    assert result.message == "Reminder set: 'study' on Friday 2 October 2026 at 7:00 PM (Asia/Dubai)."


def test_the_time_may_be_written_with_am_or_pm_and_a_weekday_name(rig):
    assert run(rig, "create_reminder", text="a", at="7 PM", day="friday").data["due_at"] == "2026-10-02T15:00:00+00:00"


def test_repeating_forms(rig):
    daily = run(rig, "create_reminder", text="study", at="19:00", repeat="daily")
    assert daily.data["schedule"]["kind"] == "daily" and daily.data["due_at"] == "2026-10-01T15:00:00+00:00"
    weekdays = run(rig, "create_reminder", text="robotics", at="18:00", repeat="weekdays")
    assert weekdays.data["schedule"]["kind"] == "weekdays"
    weekly = run(rig, "create_reminder", text="project", at="17:00", repeat="weekly", weekdays="monday, thursday")
    assert weekly.data["schedule"]["days"] == [1, 4] and weekly.data["due_at"] == "2026-10-01T13:00:00+00:00"
    assert run(rig, "create_reminder", text="project2", at="17:00", repeat="weekly", weekdays=["mon", "fri"]).data["schedule"]["days"] == [1, 5]


def test_creating_the_same_scheduled_reminder_twice_returns_the_existing_one(rig):
    first = run(rig, "create_reminder", text="study", at="19:00", repeat="daily")
    again = run(rig, "create_reminder", text="study", at="19:00", repeat="daily")
    assert again.data["id"] == first.data["id"] and again.data["duplicate"] and "already have that reminder" in again.message
    assert len(rig.store.list()) == 1


@pytest.mark.parametrize(
    "params, fragment",
    [
        ({"at": "7"}, None),  # a bare 7 alone is allowed: the next time the clock reads 7
        ({"at": "7", "day": "tomorrow"}, "7:00 AM or 7:00 PM"),
        ({"at": "7", "repeat": "daily"}, "7:00 AM or 7:00 PM"),
        ({"at": "25:00"}, "couldn't read the time"),
        ({"at": "banana"}, "couldn't read the time"),
        ({"day": "tomorrow"}, "What time"),
        ({"repeat": "daily"}, "What time"),
        ({"at": "19:00", "repeat": "monthly"}, "daily, weekdays or weekly"),
        ({"at": "19:00", "repeat": "weekly"}, "Which day"),
        ({"at": "19:00", "weekdays": "monday"}, "goes with repeat=weekly"),
        ({"at": "19:00", "repeat": "weekly", "weekdays": "funday"}, "weekday names"),
        ({"at": "19:00", "day": "yesterday"}, "couldn't understand the day"),
        ({"at": "13:00", "day": "today"}, "already passed today"),
        ({"at": "19:00", "due_in_minutes": 5}, "not both"),
        ({"at": "19:00", "recurrence_unit": "hours"}, "not both"),
    ],
)
def test_bad_or_conflicting_arguments_are_errors_never_guesses(rig, params, fragment):
    result = run(rig, "create_reminder", text="x", **params)
    if fragment is None:
        assert result.status is ToolResultStatus.SUCCESS
    else:
        assert result.status is ToolResultStatus.ERROR and fragment in result.message
        assert rig.store.list() == []


def test_the_callers_cannot_choose_the_time_zone(rig):
    result = run(rig, "create_reminder", text="x", at="19:00", day="tomorrow", timezone="America/New_York")
    assert result.status is ToolResultStatus.ERROR  # `timezone` is not a parameter at all: the registry rejects it
    assert "timezone" not in [p.name for p in rig.registry.get("create_reminder").parameters]


# ====================================== list, pause, resume, cancel ======================================================================


def test_list_reminders_adds_plain_words_and_accepts_the_paused_filter(rig):
    created = run(rig, "create_reminder", text="study", at="19:00", repeat="daily").data
    run(rig, "pause_reminder", reminder_id=created["id"])
    [item] = run(rig, "list_reminders", status="paused").data
    assert item["status"] == "paused" and item["description"] == "every day at 7:00 PM (Asia/Dubai)" and item["timezone"] == "Asia/Dubai"
    assert run(rig, "list_reminders", status="pending").data == []
    assert run(rig, "list_reminders", status="sleeping").status is ToolResultStatus.ERROR


def test_pause_and_resume_tools_say_what_happened_without_showing_ids(rig):
    created = run(rig, "create_reminder", text="study", at="19:00", repeat="daily").data
    paused = run(rig, "pause_reminder", reminder_id=created["id"])
    assert paused.message == "Paused: 'study'. It won't remind you until you resume it." and created["id"] not in paused.message
    resumed = run(rig, "resume_reminder", reminder_id=created["id"])
    assert resumed.data["status"] == "pending" and resumed.message.startswith("Resumed: 'study'. Next: Thursday 1 October, 7:00 PM")
    assert run(rig, "resume_reminder", reminder_id=created["id"]).status is ToolResultStatus.ERROR
    assert run(rig, "pause_reminder", reminder_id="nope").status is ToolResultStatus.ERROR


def test_cancel_is_how_a_reminder_is_deleted_and_old_complete_still_works(rig):
    first = run(rig, "create_reminder", text="a", at="19:00", repeat="daily").data["id"]
    second = run(rig, "create_reminder", text="b", at="20:00", repeat="daily").data["id"]
    assert run(rig, "cancel_reminder", reminder_id=first).status is ToolResultStatus.SUCCESS
    assert run(rig, "complete_reminder", reminder_id=second).status is ToolResultStatus.SUCCESS
    assert rig.store.get(first).status is ReminderStatus.CANCELLED and rig.store.get(second).status is ReminderStatus.COMPLETED
    assert run(rig, "cancel_reminder", reminder_id=first).status is ToolResultStatus.ERROR  # already gone


def test_cancel_and_complete_also_work_on_a_paused_reminder(rig):
    first = run(rig, "create_reminder", text="a", at="19:00", repeat="daily").data["id"]
    run(rig, "pause_reminder", reminder_id=first)
    assert run(rig, "cancel_reminder", reminder_id=first).status is ToolResultStatus.SUCCESS


# ====================================== the planner / Handle This boundary ==========================================================


def plan(rig, *steps):
    return validate_plan(list(steps), rig.registry, max_steps=5, default_retry=RetryPolicy(max_attempts=1))


def test_a_plan_can_still_create_a_reminder_including_with_the_new_arguments(rig):
    [step] = plan(rig, {"id": "s1", "tool": "create_reminder", "arguments": {"text": "decide on the laptop", "at": "19:00", "day": "tomorrow"}})
    assert step.tool_name == "create_reminder" and step.permission == PermissionLevel.SAFE_ACTION.value
    [legacy] = plan(rig, {"id": "s1", "tool": "create_reminder", "arguments": {"text": "decide", "due_in_minutes": 1440}})
    assert legacy.arguments["due_in_minutes"] == 1440


def test_a_plan_cannot_pause_or_resume_the_users_reminders(rig):
    for name in ("pause_reminder", "resume_reminder"):
        with pytest.raises(PlanValidationError):
            plan(rig, {"id": "s1", "tool": name, "arguments": {"reminder_id": "abc"}})
    assert "pause_reminder" not in planner_tool_names(rig.registry) and "resume_reminder" not in planner_tool_names(rig.registry)
    assert "create_reminder" in planner_tool_names(rig.registry)


def test_a_plan_cannot_choose_the_zone_or_add_unknown_arguments(rig):
    with pytest.raises(PlanValidationError):
        plan(rig, {"id": "s1", "tool": "create_reminder", "arguments": {"text": "x", "at": "19:00", "timezone": "Europe/London"}})


def test_handle_this_boundary_is_unchanged_reminders_remain_in_the_organization_family():
    declared = DEFAULT_BOUNDARY.declared_tools(GoalFamily.ORGANIZATION)
    assert {"create_reminder", "list_reminders", "complete_reminder", "cancel_reminder"} <= declared
    assert "pause_reminder" not in declared and "resume_reminder" not in declared  # management is not an objective's tool


def test_an_explicit_command_with_prose_still_goes_to_the_ai_extractor_not_the_whole_text_as_the_reminder(rig):
    """create_reminder now has ONE required parameter (text); because it takes prose arguments the builder must still hand
    "remind me to stretch in 10 minutes" to the extractor, not store the whole sentence as the text."""
    provider = FakeToolCallProvider(ToolCall(tool_name="create_reminder", parameters={"text": "stretch", "due_in_minutes": 10}))
    call = ToolCallBuilder(provider).build("use tool create_reminder remind me to stretch in 10 minutes", rig.registry, "create_reminder")
    assert call.parameters == {"text": "stretch", "due_in_minutes": 10}


def test_the_builder_still_binds_the_rest_of_the_message_for_ordinary_single_string_tools():
    from app.tools.base import Tool, ToolParameter, ToolResult

    class Echo(Tool):
        name, description, permission = "echo", "echo", PermissionLevel.READ
        parameters = [ToolParameter(name="text", type="string")]

        def execute(self, text):
            return ToolResult.success(message=text)

    registry = ToolRegistry()
    registry.register(Echo())

    class Boom(FakeToolCallProvider):
        def decide(self, message, candidate_tools):
            raise AssertionError("the extractor must not be used for an ordinary one-string tool")

    call = ToolCallBuilder(Boom(None)).build("use tool echo hello world", registry, "echo")
    assert call.parameters == {"text": "hello world"}


def test_a_new_repeating_reminder_is_stored_as_a_v2_interval_schedule_not_a_legacy_recurrence(rig):
    result = run(rig, "create_reminder", text="water", due_in_minutes=5, recurrence_unit="hours", recurrence_interval=2)
    stored = rig.store.get(result.data["id"])
    assert stored.recurrence is None  # no legacy recurrence ...
    assert stored.schedule.kind.value == "interval" and stored.schedule.interval.interval == 2  # ... a V2 schedule
    assert stored.schedule.interval.unit.value == "hours"
    assert stored.due_at == (NOW + timedelta(minutes=5)).isoformat()
    assert result.data["schedule"] is not None


def test_a_new_repeating_reminder_without_a_known_time_zone_is_still_created_in_utc(rig):
    from app.reminders.language import parse_clock
    from app.reminders.service import ReminderRequest, ReminderRequestError
    from app.scheduling.timezones import TimezoneUnavailableError

    def no_zone():
        raise TimezoneUnavailableError("no zone")

    service = ReminderService(rig.store, clock=lambda: NOW, zone_resolver=no_zone)
    created = service.create(ReminderRequest(text="water", every=("hours", 2)))
    assert created.reminder.schedule.tz == "UTC"
    with pytest.raises(ReminderRequestError):  # a clock-time reminder still refuses to guess
        service.create(ReminderRequest(text="study", clock=parse_clock("7 pm")))
