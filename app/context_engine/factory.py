"""
Builds the application's default ContextEngine - the one place
construction happens, same role every other app/*/factory.py plays for
its own service (see e.g. app/world_model/factory.py).
"""

from typing import Callable

from app.context_engine.engine import ContextEngine
from app.memory.base import MemoryStore
from app.world_model.base import TaskLink
from app.world_model.service import WorldModelService


def create_default_context_engine(
    world_model: WorldModelService,
    memory: MemoryStore | None = None,
    *,
    task_label_provider: Callable[[TaskLink], str | None] | None = None,
) -> ContextEngine:
    """Takes a WorldModelService (and optionally a MemoryStore) rather
    than building either itself - callers pass
    `create_default_world_model_service(create_default_world_model_store(
    settings.world_model_db_path))` explicitly, the same "factory takes
    its dependencies, caller supplies settings" split every other
    app/*/factory.py in this codebase already uses (see e.g.
    app/world_model/factory.py, app/orchestration/factory.py)."""
    return ContextEngine(world_model, memory, task_label_provider=task_label_provider)
