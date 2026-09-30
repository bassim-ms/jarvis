"""
Calendar tools (M9): adapt CalendarProvider into the Tool interface, so
list/create/update/delete reach a calendar event through the same
ToolRegistry (app/tools/registry.py) and TOOL capability
(app/capability.py, app/tools/decision.py) every other tool already uses -
no special-casing in JarvisCore, exactly like web_search (M6.9-6.16) and
the reminder/automation tools (M7/M8).

Permission levels are the whole point of this module's safety story:
`list_events` is READ (no side effects); `create_event`/`update_event`
are CONFIRM (a calendar event can be visible to other people, unlike a
local reminder/task); `delete_event` is DANGEROUS (destructive, and
calendar deletion isn't easily undone from Rhasspy's side). None of that
is enforced here - ToolRegistry.execute (app/tools/registry.py) already
does it, unchanged, for every tool - these classes only ever declare
their own `permission`. A direct, useful consequence: AutomationEngine
(app/automation/engine.py, M8) only ever runs READ/SAFE_ACTION tools, so
`create_event`/`update_event`/`delete_event` can never be used as an
automation's action - see app/automation/tools.py's own guard, which this
module doesn't need to duplicate or extend.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from app.calendar.base import CalendarEvent, CalendarProvider, CalendarProviderError
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult


def _event_to_dict(event: CalendarEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "title": event.title,
        "start": event.start,
        "end": event.end,
        "description": event.description,
        "location": event.location,
    }


class ListEventsTool(Tool):
    """Lists upcoming events in a time window. READ: no side effects.
    Both parameters are optional, so "use tool list_events" alone is
    enough - the same zero-required-parameter shape as PingTool in
    tests/test_tool_decision.py, letting ToolCallBuilder build the call
    deterministically with no AI involved."""

    name = "list_events"
    description = "List upcoming calendar events in the next `days_ahead` days."
    permission = PermissionLevel.READ

    def __init__(self, provider: CalendarProvider):
        self.provider = provider

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="days_ahead",
                type="number",
                description="How many days ahead to list events for (default 7)",
                required=False,
                default=7,
            ),
            ToolParameter(
                name="max_results",
                type="number",
                description="Maximum number of events to return (default 20)",
                required=False,
                default=20,
            ),
        ]

    def execute(self, days_ahead: float = 7, max_results: float = 20) -> ToolResult:
        now = datetime.now(timezone.utc)
        time_min = now.isoformat()
        time_max = (now + timedelta(days=days_ahead)).isoformat()

        try:
            events = self.provider.list_events(
                time_min=time_min, time_max=time_max, max_results=int(max_results)
            )
        except CalendarProviderError as exc:
            return ToolResult.error(str(exc))

        return ToolResult.success(data=[_event_to_dict(e) for e in events])


class CreateEventTool(Tool):
    """Creates a new calendar event. CONFIRM: a created event can be
    visible to other people (or trigger their own notifications), so it
    requires confirmation before running - ToolRegistry.execute enforces
    this exactly like any other CONFIRM tool."""

    name = "create_event"
    description = "Create a calendar event with a title and start/end time."
    permission = PermissionLevel.CONFIRM
    #: Prose arguments ("remind me to stretch in 10 minutes"): the one kind
    #: of tool an explicit command may hand to the AI ToolCallProvider (see
    #: app/tools/command.py). Every other tool is resolved deterministically.
    natural_language_arguments = True

    def __init__(self, provider: CalendarProvider):
        self.provider = provider

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="title", type="string", description="Event title"),
            ToolParameter(
                name="start", type="string", description="Start time, ISO 8601"
            ),
            ToolParameter(name="end", type="string", description="End time, ISO 8601"),
            ToolParameter(
                name="description",
                type="string",
                description="Event description",
                required=False,
                default="",
            ),
            ToolParameter(
                name="location",
                type="string",
                description="Event location",
                required=False,
                default="",
            ),
        ]

    def execute(
        self,
        title: str,
        start: str,
        end: str,
        description: str = "",
        location: str = "",
    ) -> ToolResult:
        try:
            event = self.provider.create_event(
                title=title, start=start, end=end, description=description, location=location
            )
        except CalendarProviderError as exc:
            return ToolResult.error(str(exc))

        return ToolResult.success(
            data=_event_to_dict(event), message=f"Event '{title}' created."
        )


class UpdateEventTool(Tool):
    """Updates an existing calendar event's fields. CONFIRM: same
    other-people-visible reasoning as CreateEventTool."""

    name = "update_event"
    description = "Update a calendar event's title/time/description/location, by its ID."
    permission = PermissionLevel.CONFIRM
    #: Prose arguments ("remind me to stretch in 10 minutes"): the one kind
    #: of tool an explicit command may hand to the AI ToolCallProvider (see
    #: app/tools/command.py). Every other tool is resolved deterministically.
    natural_language_arguments = True

    def __init__(self, provider: CalendarProvider):
        self.provider = provider

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="event_id", type="string", description="ID of the event"),
            ToolParameter(
                name="title",
                type="string",
                description="New title",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="start",
                type="string",
                description="New start time, ISO 8601",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="end",
                type="string",
                description="New end time, ISO 8601",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="description",
                type="string",
                description="New description",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="location",
                type="string",
                description="New location",
                required=False,
                default=None,
            ),
        ]

    def execute(
        self,
        event_id: str,
        title: str | None = None,
        start: str | None = None,
        end: str | None = None,
        description: str | None = None,
        location: str | None = None,
    ) -> ToolResult:
        try:
            event = self.provider.update_event(
                event_id,
                title=title,
                start=start,
                end=end,
                description=description,
                location=location,
            )
        except CalendarProviderError as exc:
            return ToolResult.error(str(exc))

        return ToolResult.success(
            data=_event_to_dict(event), message=f"Event '{event_id}' updated."
        )


class DeleteEventTool(Tool):
    """Deletes a calendar event. DANGEROUS: destructive and not easily
    undone from Rhasspy's side - requires explicit/strong confirmation,
    ToolRegistry.execute's existing DANGEROUS handling, unchanged."""

    name = "delete_event"
    description = "Delete a calendar event, by its ID."
    permission = PermissionLevel.DANGEROUS

    def __init__(self, provider: CalendarProvider):
        self.provider = provider

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(name="event_id", type="string", description="ID of the event")
        ]

    def execute(self, event_id: str) -> ToolResult:
        try:
            self.provider.delete_event(event_id)
        except CalendarProviderError as exc:
            return ToolResult.error(str(exc))

        return ToolResult.success(message=f"Event '{event_id}' deleted.")
