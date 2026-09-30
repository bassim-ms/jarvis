"""
SQLite-backed implementation of NoteStore, using Python's stdlib sqlite3 -
the same pattern as SQLiteTaskStore/SQLiteReminderStore: one connection held
for the store's lifetime, guarded by a lock since FastAPI and the JobEngine
thread can both call into a store, backed by a plain local file so notes
survive a restart. Its own database file (settings.notes_db_path) - never
the memory or World Model databases.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime, timezone

from app.notes.base import Note, NoteStore


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteNoteStore(NoteStore):
    """NoteStore backed by a local SQLite database file."""

    def __init__(self, db_path: str = "jarvis_notes.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    source_job_id TEXT,
                    source_step_id TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_notes_job ON notes (source_job_id);
                """
            )
            self._conn.commit()

    @staticmethod
    def _row_to_note(row: sqlite3.Row) -> Note:
        return Note(
            id=row["id"],
            title=row["title"],
            content=row["content"],
            created_at=row["created_at"],
            source_job_id=row["source_job_id"],
            source_step_id=row["source_step_id"],
        )

    def create(
        self,
        title: str,
        content: str,
        source_job_id: str | None = None,
        source_step_id: str | None = None,
    ) -> Note:
        note_id = uuid.uuid4().hex
        created_at = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO notes (id, title, content, created_at, source_job_id, source_step_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (note_id, title, content, created_at, source_job_id, source_step_id),
            )
            self._conn.commit()
        return Note(
            id=note_id,
            title=title,
            content=content,
            created_at=created_at,
            source_job_id=source_job_id,
            source_step_id=source_step_id,
        )

    def get(self, note_id: str) -> Note | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        return self._row_to_note(row) if row else None

    def list(self, job_id: str | None = None, limit: int = 20) -> list[Note]:
        with self._lock:
            if job_id is None:
                rows = self._conn.execute(
                    "SELECT * FROM notes ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM notes WHERE source_job_id = ? "
                    "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                    (job_id, limit),
                ).fetchall()
        return [self._row_to_note(row) for row in rows]
