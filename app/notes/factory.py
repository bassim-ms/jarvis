"""
Builds the application's default NoteStore from config - the one place
construction happens, same role app/tasks/factory.py plays for the task
store.
"""

from app.notes.base import NoteStore
from app.notes.sqlite_store import SQLiteNoteStore


def create_default_note_store(db_path: str) -> NoteStore:
    return SQLiteNoteStore(db_path)
