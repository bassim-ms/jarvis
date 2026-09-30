"""
Builds the write service from the app's existing World Model and Memory - one
place, like every other factory. The staging/journal tables live in the World
Model's database file (their own tables, their own connection); nothing about the
World Model's or Memory's schema changes.
"""

from app.memory.base import MemoryStore
from app.memory_policy.service import MemoryWriteService
from app.memory_policy.store import SQLiteWriteStore
from app.world_model.service import WorldModelService


def create_default_memory_write_service(
    world_model: WorldModelService, memory: MemoryStore, db_path: str, *, confirmer=None
) -> MemoryWriteService:
    return MemoryWriteService(world_model, memory, SQLiteWriteStore(db_path), confirmer=confirmer)
