"""
Builds the application's default Workspace from config - the one place construction happens, same role app/notes/factory.py plays for the
note store. Nothing touches the disk here: the folder is created by the first write, not at startup.
"""

from app.config import settings
from app.workspace.base import Workspace


def create_default_workspace() -> Workspace:
    # `enabled` reads the setting on EVERY call, so the kill switch takes effect at once.
    return Workspace(
        settings.workspace_dir,
        enabled=lambda: settings.workspace_enabled,
        max_file_bytes=settings.workspace_max_file_bytes,
        max_files=settings.workspace_max_files,
        max_total_bytes=settings.workspace_max_total_bytes,
        writes_per_minute=settings.workspace_writes_per_minute,
    )
