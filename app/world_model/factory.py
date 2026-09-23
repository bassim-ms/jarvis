"""
Builds the application's default WorldModelStore/WorldModelService from
config - the one place construction happens, same role every other
app/*/factory.py plays for its own store (see e.g.
app/orchestration/factory.py).
"""

from app.world_model.base import WorldModelStore
from app.world_model.service import WorldModelService
from app.world_model.sqlite_store import SQLiteWorldModelStore


def create_default_world_model_store(db_path: str) -> WorldModelStore:
    return SQLiteWorldModelStore(db_path)


def create_default_world_model_service(store: WorldModelStore) -> WorldModelService:
    """Takes a store rather than building one itself - callers pass
    `create_default_world_model_store(settings.world_model_db_path)`
    explicitly, the same "factory takes db_path, caller supplies
    settings" split every other app/*/factory.py in this codebase
    already uses (see e.g. app/tasks/factory.py)."""
    return WorldModelService(store)
