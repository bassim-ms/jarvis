"""
Default tool registration for the running application.

This is the one place that knows the concrete set of tools Rhasspy ships
with by default (`web_search`, the reminder tools since M7, the
automation tools since M8, and the calendar/task tools since M9). It
exists so the actual application (app/main.py) can get a fully-populated
ToolRegistry, while JarvisCore itself (app/core.py) stays unaware of any
specific tool - its own default, when no registry is injected, stays an
empty ToolRegistry (see
test_tools.py::test_core_exposes_a_tool_registry_by_default).
"""

from app.automation.base import AutomationStore
from app.automation.engine import AutomationEngine
from app.automation.factory import create_default_automation_store, create_default_engine
from app.automation.tools import (
    CancelAutomationTool,
    CreateAutomationTool,
    DisableAutomationTool,
    EnableAutomationTool,
    ListAutomationsTool,
    RunAutomationTool,
)
from app.calendar.base import CalendarProvider
from app.calendar.factory import create_default_calendar_provider
from app.calendar.tools import (
    CreateEventTool,
    DeleteEventTool,
    ListEventsTool,
    UpdateEventTool,
)
from app.config import settings
from app.memory.base import MemoryStore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.reminders.base import ReminderStore
from app.reminders.factory import create_default_reminder_store
from app.reminders.tools import (
    CancelReminderTool,
    CompleteReminderTool,
    CreateReminderTool,
    ListRemindersTool,
)
from app.search.cached import CachedSearchProvider
from app.search.factory import (
    create_default_search_cache,
    create_default_search_provider,
)
from app.search.quality import ResultQualityProvider
from app.tasks.base import TaskStore
from app.tasks.factory import create_default_task_store
from app.tasks.tools import CancelTaskTool, CompleteTaskTool, CreateTaskTool, ListTasksTool
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool


def register_default_tools(
    registry: ToolRegistry,
    reminder_store: ReminderStore | None = None,
    automation_store: AutomationStore | None = None,
    memory: MemoryStore | None = None,
    calendar_provider: CalendarProvider | None = None,
    task_store: TaskStore | None = None,
) -> AutomationEngine:
    """Register every tool Rhasspy ships with by default onto `registry`,
    and return the AutomationEngine built alongside the automation tools
    (see below) - the one piece of default wiring that isn't itself a
    Tool, so a caller that also needs to start/stop it (app/main.py) gets
    it back rather than having to reconstruct an equivalent one.

    Takes a registry rather than owning one, so callers - the application
    singleton, tests, or a future dashboard - decide what registry gets
    the defaults, instead of this module hardcoding where they end up.

    web_search is wired as WebSearchTool -> ResultQualityProvider ->
    CachedSearchProvider -> SearchProviderManager -> (SearXNGProvider, ...),
    built from app/search/factory.py, so JARVIS_SEARCH_PROVIDER_ORDER and
    JARVIS_SEARCH_CACHE_TTL both take effect without WebSearchTool,
    SearchProviderManager, or JarvisCore knowing anything about SQLite,
    caching, or result deduplication.

    `reminder_store`/`automation_store`/`memory` each default to a fresh
    store at their own `settings.*_db_path` when not given - callers that
    also need to start a ReminderScheduler/AutomationEngine against the
    *same* store, or want automation conditions to see JarvisCore's own
    facts (app/main.py passes `core.memory`), pass one in explicitly, so
    tool execution and the background pollers' own reads/writes both see
    the same persisted rows instead of independent SQLite connections
    opened from the same default path.

    Automation wiring: CreateAutomationTool/ListAutomationsTool/
    EnableAutomationTool/DisableAutomationTool/CancelAutomationTool
    operate on `automation_store` directly; RunAutomationTool instead
    holds the AutomationEngine itself, since running an automation means
    evaluating its conditions and dispatching its action through
    `registry` (see app/automation/engine.py) - `registry` is passed to
    AutomationEngine here even before this function has finished
    registering every tool onto it, which is safe because
    ToolRegistry.execute looks up a tool by name only when actually
    called, not at construction time, so any tool registered later (by
    this function or afterward) is still visible to the engine.

    `calendar_provider`/`task_store` follow the same
    default-unless-given pattern as the reminder/automation stores.
    Calendar tools carry their own permission split - `list_events` is
    READ; `create_event`/`update_event` are CONFIRM; `delete_event` is
    DANGEROUS (see app/calendar/tools.py) - enforced entirely by
    ToolRegistry.execute, unchanged; AutomationEngine (M8) therefore can
    never run the three calendar write tools unattended, with no changes
    needed here or in app/automation/*. Task tools are all READ/
    SAFE_ACTION, same as the reminder tools.
    """
    provider = ResultQualityProvider(
        CachedSearchProvider(
            create_default_search_provider(), create_default_search_cache()
        )
    )
    registry.register(WebSearchTool(provider=provider))

    reminder_store = reminder_store or create_default_reminder_store(
        settings.reminders_db_path
    )
    registry.register(CreateReminderTool(store=reminder_store))
    registry.register(ListRemindersTool(store=reminder_store))
    registry.register(CompleteReminderTool(store=reminder_store))
    registry.register(CancelReminderTool(store=reminder_store))

    automation_store = automation_store or create_default_automation_store(
        settings.automations_db_path
    )
    memory = memory or SQLiteMemoryStore(settings.memory_db_path)
    engine = create_default_engine(
        automation_store,
        registry,
        reminder_store,
        memory,
        settings.automation_poll_interval_seconds,
    )
    registry.register(
        CreateAutomationTool(
            store=automation_store, tools=registry, reminder_store=reminder_store
        )
    )
    registry.register(ListAutomationsTool(store=automation_store))
    registry.register(EnableAutomationTool(store=automation_store))
    registry.register(DisableAutomationTool(store=automation_store))
    registry.register(CancelAutomationTool(store=automation_store))
    registry.register(RunAutomationTool(engine=engine))

    calendar_provider = calendar_provider or create_default_calendar_provider()
    registry.register(ListEventsTool(provider=calendar_provider))
    registry.register(CreateEventTool(provider=calendar_provider))
    registry.register(UpdateEventTool(provider=calendar_provider))
    registry.register(DeleteEventTool(provider=calendar_provider))

    task_store = task_store or create_default_task_store(settings.tasks_db_path)
    registry.register(CreateTaskTool(store=task_store))
    registry.register(ListTasksTool(store=task_store))
    registry.register(CompleteTaskTool(store=task_store))
    registry.register(CancelTaskTool(store=task_store))

    return engine


def create_default_registry() -> ToolRegistry:
    """Build a fresh ToolRegistry populated with the default tools."""
    registry = ToolRegistry()
    register_default_tools(registry)
    return registry
