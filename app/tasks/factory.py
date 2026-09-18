"""
Builds the application's default TaskStore from config - the one place
construction happens, same role app/reminders/factory.py plays for the
reminder store.
"""

from app.tasks.base import TaskStore
from app.tasks.sqlite_store import SQLiteTaskStore


def create_default_task_store(db_path: str) -> TaskStore:
    return SQLiteTaskStore(db_path)
