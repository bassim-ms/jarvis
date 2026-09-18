"""
Tests for the calendar Tools (app/calendar/tools.py): ListEventsTool,
CreateEventTool, UpdateEventTool, DeleteEventTool - both directly and
through ToolRegistry, proving permission levels (READ/CONFIRM/DANGEROUS)
are enforced exactly like every other tool in this codebase, and that a
CalendarProviderError becomes a normal ToolResult.error rather than
propagating.
"""

from app.calendar.base import CalendarEvent, CalendarProvider, CalendarProviderError
from app.calendar.tools import (
    CreateEventTool,
    DeleteEventTool,
    ListEventsTool,
    UpdateEventTool,
)
from app.tools.base import PermissionLevel, ToolResultStatus
from app.tools.registry import ToolRegistry


class FakeCalendarProvider(CalendarProvider):
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.created: list[tuple] = []
        self.updated: list[tuple] = []
        self.deleted: list[str] = []

    def list_events(self, time_min=None, time_max=None, max_results=20):
        if self.error:
            raise self.error
        return [CalendarEvent(id="e1", title="Standup", start="s", end="e")]

    def create_event(self, title, start, end, description="", location=""):
        if self.error:
            raise self.error
        self.created.append((title, start, end, description, location))
        return CalendarEvent(
            id="new1", title=title, start=start, end=end, description=description, location=location
        )

    def update_event(self, event_id, title=None, start=None, end=None, description=None, location=None):
        if self.error:
            raise self.error
        self.updated.append((event_id, title, start, end, description, location))
        return CalendarEvent(id=event_id, title=title or "x", start=start or "s", end=end or "e")

    def delete_event(self, event_id):
        if self.error:
            raise self.error
        self.deleted.append(event_id)


def make_registry(provider: CalendarProvider | None = None):
    provider = provider or FakeCalendarProvider()
    registry = ToolRegistry()
    registry.register(ListEventsTool(provider=provider))
    registry.register(CreateEventTool(provider=provider))
    registry.register(UpdateEventTool(provider=provider))
    registry.register(DeleteEventTool(provider=provider))
    return registry, provider


# --- permissions -------------------------------------------------------


def test_permission_levels_match_the_locked_scope():
    registry, _ = make_registry()

    assert registry.get("list_events").permission is PermissionLevel.READ
    assert registry.get("create_event").permission is PermissionLevel.CONFIRM
    assert registry.get("update_event").permission is PermissionLevel.CONFIRM
    assert registry.get("delete_event").permission is PermissionLevel.DANGEROUS


def test_list_events_runs_automatically_with_no_parameters():
    registry, _ = make_registry()

    result = registry.execute("list_events")

    assert result.status == ToolResultStatus.SUCCESS
    assert len(result.data) == 1


def test_create_event_requires_confirmation_and_never_runs_without_it():
    registry, provider = make_registry()
    params = {"title": "x", "start": "s", "end": "e"}

    pending = registry.execute("create_event", params=params)
    assert pending.status == ToolResultStatus.CONFIRMATION_REQUIRED
    assert pending.permission == PermissionLevel.CONFIRM
    assert provider.created == []

    declined = registry.execute("create_event", params=params, confirmed=False)
    assert declined.status == ToolResultStatus.DENIED
    assert provider.created == []

    approved = registry.execute("create_event", params=params, confirmed=True)
    assert approved.status == ToolResultStatus.SUCCESS
    assert provider.created == [("x", "s", "e", "", "")]


def test_update_event_requires_confirmation_and_never_runs_without_it():
    registry, provider = make_registry()
    params = {"event_id": "e1", "title": "new"}

    pending = registry.execute("update_event", params=params)
    assert pending.status == ToolResultStatus.CONFIRMATION_REQUIRED
    assert provider.updated == []

    approved = registry.execute("update_event", params=params, confirmed=True)
    assert approved.status == ToolResultStatus.SUCCESS
    assert provider.updated == [("e1", "new", None, None, None, None)]


def test_delete_event_requires_dangerous_confirmation_and_never_runs_without_it():
    registry, provider = make_registry()
    params = {"event_id": "e1"}

    pending = registry.execute("delete_event", params=params)
    assert pending.status == ToolResultStatus.CONFIRMATION_REQUIRED
    assert pending.permission == PermissionLevel.DANGEROUS
    assert provider.deleted == []

    declined = registry.execute("delete_event", params=params, confirmed=False)
    assert declined.status == ToolResultStatus.DENIED
    assert provider.deleted == []

    approved = registry.execute("delete_event", params=params, confirmed=True)
    assert approved.status == ToolResultStatus.SUCCESS
    assert provider.deleted == ["e1"]


# --- provider errors surface as ToolResult.error, never propagate ---------


def test_list_events_surfaces_provider_error():
    registry, _ = make_registry(FakeCalendarProvider(error=CalendarProviderError("not authorized")))

    result = registry.execute("list_events")

    assert result.status == ToolResultStatus.ERROR
    assert "not authorized" in result.message


def test_create_event_surfaces_provider_error():
    registry, _ = make_registry(FakeCalendarProvider(error=CalendarProviderError("boom")))

    result = registry.execute(
        "create_event", params={"title": "x", "start": "s", "end": "e"}, confirmed=True
    )

    assert result.status == ToolResultStatus.ERROR
    assert "boom" in result.message


def test_delete_event_surfaces_provider_error():
    registry, _ = make_registry(FakeCalendarProvider(error=CalendarProviderError("gone")))

    result = registry.execute("delete_event", params={"event_id": "e1"}, confirmed=True)

    assert result.status == ToolResultStatus.ERROR
    assert "gone" in result.message
