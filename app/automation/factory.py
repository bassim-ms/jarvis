"""
Builds the application's default AutomationStore/AutomationEngine from
config - the one place construction happens, same role
app/reminders/factory.py plays for the reminder store/scheduler.
"""

from app.automation.base import AutomationStore
from app.automation.engine import AutomationEngine
from app.automation.sqlite_store import SQLiteAutomationStore
from app.memory.base import MemoryStore
from app.reminders.base import ReminderStore
from app.tools.registry import ToolRegistry


def create_default_automation_store(db_path: str) -> AutomationStore:
    return SQLiteAutomationStore(db_path)


def create_default_engine(
    store: AutomationStore,
    tools: ToolRegistry,
    reminder_store: ReminderStore,
    memory: MemoryStore,
    poll_interval_seconds: float,
) -> AutomationEngine:
    return AutomationEngine(
        store=store,
        tools=tools,
        reminder_store=reminder_store,
        memory=memory,
        poll_interval_seconds=poll_interval_seconds,
    )
