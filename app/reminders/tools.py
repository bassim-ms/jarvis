"""
Reminder tools (M7): adapt ReminderStore into the Tool interface, so
create/list/complete/cancel reach a reminder through the same
ToolRegistry (app/tools/registry.py) and TOOL capability
(app/capability.py, app/tools/decision.py) every other tool already uses -
no special-casing in JarvisCore, exactly like web_search (M6.9-6.16).

None of these tools start, stop, or otherwise touch ReminderScheduler -
creating a reminder here just persists a row; the separately-running
scheduler (app/reminders/scheduler.py) is what later notices it's due and
notifies. That split ("storage vs. background process") mirrors
SearchCache/SQLiteSearchCache vs. the search pipeline that reads it.

Reminder V2: `create_reminder` can also take a local time of day, a day, and a
repetition (every day / weekday / chosen weekdays / every N units) instead of
"minutes from now"; all of that is turned into a stored Schedule by
ReminderService (app/reminders/service.py), the one implementation shared with
the chat door. `pause_reminder` / `resume_reminder` manage a reminder by ID. They
are user-driven management: not offered to a Planner (`plannable = False`) and
never run by an automation (`automatable = False`), so a goal-driven plan, or
text a plan read on the web, cannot pause or resume the user's reminders.
`cancel_reminder` is how a reminder is deleted. Nothing here runs another tool,
makes a Job, or writes memory: a reminder only ever becomes a notification.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from app.notifications.text import clean_text
from app.reminders.base import (
    RecurrenceError,
    Reminder,
    ReminderStatus,
    ReminderStore,
    parse_recurrence,
)
from app.reminders.language import WEEKDAY_NUMBERS, parse_clock
from app.reminders.service import (
    Created,
    ReminderRequest,
    ReminderRequestError,
    ReminderService,
)
from app.scheduling.schedule import interval_delta
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult

_VALID_STATUSES = {member.value for member in ReminderStatus}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _repetition(reminder: Reminder):
    """The "every N units" a reminder repeats on: its legacy recurrence, or the interval of a V2 interval schedule."""
    if reminder.recurrence is not None:
        return reminder.recurrence
    if reminder.schedule is not None and reminder.schedule.interval is not None:
        return reminder.schedule.interval
    return None


def _reminder_to_dict(reminder: Reminder) -> dict[str, Any]:
    return {
        "id": reminder.id,
        "text": reminder.text,
        "due_at": reminder.due_at,
        "status": reminder.status.value,
        "recurrence_unit": (_repetition(reminder).unit.value if _repetition(reminder) else None),
        "recurrence_interval": (_repetition(reminder).interval if _repetition(reminder) else None),
        # Reminder V2: the stored schedule (None for a reminder created before V2) and the zone it was created in.
        "schedule": reminder.schedule.to_dict() if reminder.schedule is not None else None,
        "timezone": reminder.timezone,
    }


def _service_for(store: ReminderStore, service: ReminderService | None) -> ReminderService:
    return service if service is not None else ReminderService(store)


def _weekday_numbers(value: object) -> tuple[int, ...] | None:
    """"monday,wednesday" / ["mon", "fri"] / [1, 3] -> ISO weekday numbers, or None if anything in it is not a weekday."""
    if value is None or value == "":
        return ()
    items = value if isinstance(value, (list, tuple)) else str(value).replace(" and ", ",").replace(";", ",").split(",")
    found: set[int] = set()
    for item in items:
        if isinstance(item, bool):
            return None
        if isinstance(item, int) and 1 <= item <= 7:
            found.add(item)
            continue
        name = str(item).strip().lower()
        if not name:
            continue
        if name not in WEEKDAY_NUMBERS:
            return None
        found.add(WEEKDAY_NUMBERS[name])
    return tuple(sorted(found))


def _created_message(created: Created) -> str:
    schedule, reminder = created.schedule, created.reminder
    shown = clean_text(reminder.text, 80)
    when = schedule.describe()
    if created.duplicate:
        return f"You already have that reminder: '{shown}' {when}."
    message = f"Reminder set: '{shown}' {when}."
    if schedule.kind.value != "once":
        from app.reminders.timefmt import local_phrase

        message += f" The first one is {local_phrase(created.due, schedule.tz)}."
    elif created.rolled:
        message += " (That time has already passed today, so it's for tomorrow.)"
    if created.warning:
        message += f" Note: {created.warning}"
    return message


class CreateReminderTool(Tool):
    """Creates a new PENDING reminder - either `due_in_minutes` from now
    (optionally repeating every `recurrence_interval` `recurrence_unit`s
    after that), or at a LOCAL time of day (`at`) on a given `day`, or
    repeating every day / weekday / chosen weekdays (`repeat`). SAFE_ACTION:
    it only ever adds a new row - it can't affect any other reminder, and
    what it creates can only ever become a notification (see
    ReminderScheduler), never a Tool invocation."""

    name = "create_reminder"
    description = (
        "Create a reminder. Either `due_in_minutes` from now, or a local time of day `at` (24-hour HH:MM, e.g. 19:00) "
        "with an optional `day` (today, tomorrow, a weekday name, or YYYY-MM-DD) - or repeating with `repeat` "
        "(daily, weekdays, weekly + `weekdays`). The time zone is chosen by Jarvis, never by the caller."
    )
    permission = PermissionLevel.SAFE_ACTION
    #: Prose arguments ("remind me to stretch in 10 minutes"): the one kind
    #: of tool an explicit command may hand to the AI ToolCallProvider (see
    #: app/tools/command.py). Every other tool is resolved deterministically.
    natural_language_arguments = True

    def __init__(self, store: ReminderStore, service: ReminderService | None = None):
        self.store = store
        self.service = _service_for(store, service)

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="text", type="string", description="What to remind the user"
            ),
            ToolParameter(
                name="due_in_minutes",
                type="number",
                description="Minutes from now the reminder should first fire (omit when using `at`)",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="at",
                type="string",
                description="Local time of day, 24-hour HH:MM (19:00) or with AM/PM (7 PM). Omit when using due_in_minutes.",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="day",
                type="string",
                description="With `at`: today, tomorrow, a weekday name (friday, next friday), YYYY-MM-DD, or MM-DD",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="repeat",
                type="string",
                description="Repeat at `at`: daily, weekdays (Mon-Fri), or weekly (with `weekdays`)",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="weekdays",
                type="string",
                description="With repeat=weekly: comma-separated weekday names, e.g. monday,thursday",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="recurrence_unit",
                type="string",
                description=(
                    "Repeat unit: minutes, hours, days, or weeks. Omit for a "
                    "one-time reminder."
                ),
                required=False,
                default=None,
            ),
            ToolParameter(
                name="recurrence_interval",
                type="number",
                description="Repeat every N units (default 1)",
                required=False,
                default=1,
            ),
        ]

    def execute(
        self,
        text: str,
        due_in_minutes: float | None = None,
        recurrence_unit: str | None = None,
        recurrence_interval: float = 1,
        at: str | None = None,
        day: str | None = None,
        repeat: str | None = None,
        weekdays: Any = None,
    ) -> ToolResult:
        if at is not None or day is not None or repeat is not None or weekdays not in (None, ""):
            if due_in_minutes is not None or recurrence_unit is not None:
                return ToolResult.error(
                    "Give either `due_in_minutes` (with an optional recurrence) or a local time `at` - not both."
                )
            return self._create_scheduled(text, at, day, repeat, weekdays)
        if due_in_minutes is None:
            return ToolResult.error("Say when: either `due_in_minutes`, or a local time `at`.")
        if due_in_minutes < 0:
            return ToolResult.error("due_in_minutes must not be negative.")

        try:
            recurrence = parse_recurrence(recurrence_unit, recurrence_interval)
        except RecurrenceError as exc:
            return ToolResult.error(str(exc))
        if recurrence is not None:
            # A NEW repeating reminder is never closer together than the minimum (existing ones are left as they are).
            if interval_delta(recurrence) < self.service.min_interval():
                minutes = int(self.service.min_interval().total_seconds() // 60)
                return ToolResult.error(f"A repeating reminder must be at least {minutes} minutes apart.")

        if recurrence is not None:
            return self._create_repeating(text, due_in_minutes, recurrence)

        due = self.service.now() + timedelta(minutes=due_in_minutes)
        due_at = due.isoformat()

        reminder = self.store.create(text=text, due_at=due_at)
        message = f"Reminder set: '{text}' at {self.service.instant_words(due)}."
        return ToolResult.success(data=_reminder_to_dict(reminder), message=message)

    def _create_repeating(self, text: str, due_in_minutes: float, recurrence) -> ToolResult:
        """A NEW "every N units" reminder is a V2 interval schedule (first due in `due_in_minutes`), so it gets V2's missed-run
        rule. Reminders already stored with a legacy recurrence are never touched."""
        try:
            created = self.service.create(
                ReminderRequest(
                    text=text if isinstance(text, str) else "",
                    every=(recurrence.unit.value, int(recurrence.interval)),
                    first_in=timedelta(minutes=due_in_minutes),
                )
            )
        except ReminderRequestError as exc:
            return ToolResult.error(str(exc))
        data = _reminder_to_dict(created.reminder)
        if created.duplicate:
            return ToolResult.success(data=data, message=_created_message(created))
        n, unit = recurrence.interval, recurrence.unit.value
        message = f"Reminder set: '{text}' at {self.service.instant_words(created.due)}."
        message += f" It repeats every {n} {unit[:-1] if n == 1 else unit}."
        return ToolResult.success(data=data, message=message)

    def _create_scheduled(self, text: str, at: str | None, day: str | None, repeat: str | None, weekdays: Any) -> ToolResult:
        """Reminder V2: a local time of day, a day, a repetition. Everything is checked here and in ReminderService; a
        value that is not understood is an error, never a guess."""
        clock = parse_clock(at) if at is not None else None
        if at is not None and clock is None:
            return ToolResult.error(f"I couldn't read the time '{clean_text(str(at), 30)}'. Use 24-hour HH:MM, like 19:00, or 7 PM.")
        if repeat is not None and repeat not in ("daily", "weekdays", "weekly"):
            return ToolResult.error("`repeat` must be daily, weekdays or weekly.")
        numbers = _weekday_numbers(weekdays)
        if numbers is None:
            return ToolResult.error("`weekdays` must be weekday names, like monday,thursday.")
        if numbers and repeat != "weekly":
            return ToolResult.error("`weekdays` goes with repeat=weekly.")
        try:
            created = self.service.create(
                ReminderRequest(
                    text=text if isinstance(text, str) else "",
                    clock=clock,
                    day=day.strip() if isinstance(day, str) and day.strip() else None,
                    repeat=repeat,
                    weekdays=numbers,
                )
            )
        except ReminderRequestError as exc:
            return ToolResult.error(str(exc))
        data = _reminder_to_dict(created.reminder)
        data["first_due_at"] = created.due.isoformat()
        data["duplicate"] = created.duplicate
        return ToolResult.success(data=data, message=_created_message(created))


class ListRemindersTool(Tool):
    """Lists reminders, optionally filtered to one status. READ: no side
    effects."""

    name = "list_reminders"
    description = (
        "List reminders, optionally filtered by status "
        "(pending, triggered, completed, cancelled)."
    )
    permission = PermissionLevel.READ

    def __init__(self, store: ReminderStore, service: ReminderService | None = None):
        self.store = store
        self.service = _service_for(store, service)

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="status",
                type="string",
                description="Filter: pending, triggered, completed, or cancelled",
                required=False,
                default=None,
            )
        ]

    def execute(self, status: str | None = None) -> ToolResult:
        status_filter = None
        if status is not None:
            if status not in _VALID_STATUSES:
                return ToolResult.error(f"Unknown status: '{status}'")
            status_filter = ReminderStatus(status)

        reminders = self.store.list(status=status_filter)
        listed = []
        for reminder in reminders:
            entry = _reminder_to_dict(reminder)
            # plain words for people (and the chat door): when it happens, and when it next comes due
            entry["description"] = "its schedule could not be read" if reminder.schedule_invalid else self.service.schedule_words(reminder)
            entry["next"] = None if reminder.schedule_invalid else self.service.next_words(reminder)
            listed.append(entry)
        return ToolResult.success(data=listed)


class CompleteReminderTool(Tool):
    """Marks a reminder COMPLETED, stopping any future recurrence.
    SAFE_ACTION: it only ever changes that one reminder's own status."""

    name = "complete_reminder"
    description = "Mark a reminder as completed, by its ID."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, store: ReminderStore):
        self.store = store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="reminder_id", type="string", description="ID of the reminder"
            )
        ]

    def execute(self, reminder_id: str) -> ToolResult:
        if self.store.complete(reminder_id):
            return ToolResult.success(message=f"Reminder '{reminder_id}' completed.")
        return ToolResult.error(
            f"Reminder '{reminder_id}' not found or already completed/cancelled."
        )


class CancelReminderTool(Tool):
    """Marks a reminder CANCELLED. Same shape as CompleteReminderTool, kept
    as a separate tool/name since "completed" and "cancelled" are
    different user intents even though both stop future occurrences."""

    name = "cancel_reminder"
    description = "Cancel (delete) a reminder, by its ID."
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, store: ReminderStore):
        self.store = store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="reminder_id", type="string", description="ID of the reminder"
            )
        ]

    def execute(self, reminder_id: str) -> ToolResult:
        if self.store.cancel(reminder_id):
            return ToolResult.success(message=f"Reminder '{reminder_id}' cancelled.")
        return ToolResult.error(
            f"Reminder '{reminder_id}' not found or already completed/cancelled."
        )


class PauseReminderTool(Tool):
    """Pauses one active reminder so it never comes due until it is resumed. SAFE_ACTION: it changes only that reminder's own
    status. Not offered to a Planner and never run by an automation - pausing the user's reminders is the user's own call."""

    name = "pause_reminder"
    description = "Pause an active reminder, by its ID."
    permission = PermissionLevel.SAFE_ACTION
    plannable = False
    automatable = False

    def __init__(self, store: ReminderStore, service: ReminderService | None = None):
        self.store = store
        self.service = _service_for(store, service)

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="reminder_id", type="string", description="ID of the reminder")]

    def execute(self, reminder_id: str) -> ToolResult:
        try:
            reminder = self.service.pause(reminder_id)
        except ReminderRequestError as exc:
            return ToolResult.error(str(exc))
        return ToolResult.success(
            data={"id": reminder.id, "status": ReminderStatus.PAUSED.value},
            message=f"Paused: '{clean_text(reminder.text, 80)}'. It won't remind you until you resume it.",
        )


class ResumeReminderTool(Tool):
    """Resumes a paused reminder at its next FUTURE occurrence. Whatever came due while it was paused is not replayed. Same
    boundary as `pause_reminder`: user-driven, not plannable, not automatable."""

    name = "resume_reminder"
    description = "Resume a paused reminder, by its ID."
    permission = PermissionLevel.SAFE_ACTION
    plannable = False
    automatable = False

    def __init__(self, store: ReminderStore, service: ReminderService | None = None):
        self.store = store
        self.service = _service_for(store, service)

    @property
    def parameters(self) -> list[ToolParameter]:
        return [ToolParameter(name="reminder_id", type="string", description="ID of the reminder")]

    def execute(self, reminder_id: str) -> ToolResult:
        try:
            reminder, upcoming = self.service.resume(reminder_id)
        except ReminderRequestError as exc:
            return ToolResult.error(str(exc))
        from app.reminders.timefmt import local_phrase

        tz = reminder.timezone or self.service.zone_name_or_none()
        return ToolResult.success(
            data={"id": reminder.id, "status": ReminderStatus.PENDING.value, "due_at": upcoming.isoformat()},
            message=f"Resumed: '{clean_text(reminder.text, 80)}'. Next: {local_phrase(upcoming, tz)}.",
        )
