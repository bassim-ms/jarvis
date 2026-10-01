"""
SQLite-backed implementation of MemoryStore, using Python's stdlib sqlite3.

Keeps a single open connection for the store's lifetime (needed so ":memory:"
databases work, and so file-based ones persist), guarded by a lock since
FastAPI can call into a store from more than one thread.
"""

import sqlite3
import threading
import uuid
from datetime import datetime, timezone

from app.memory.base import MemoryStore, Message


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteMemoryStore(MemoryStore):
    """MemoryStore backed by a local SQLite database file."""

    def __init__(self, db_path: str = "jarvis_memory.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        # "Forget" must really forget: have SQLite overwrite deleted content with zeros instead of leaving it in
        # the file's free pages.
        self._conn.execute("PRAGMA secure_delete = ON")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS facts (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    def create_conversation(self) -> str:
        conversation_id = uuid.uuid4().hex
        with self._lock:
            self._conn.execute(
                "INSERT INTO conversations (id, created_at) VALUES (?, ?)",
                (conversation_id, _now()),
            )
            self._conn.commit()
        return conversation_id

    def conversation_exists(self, conversation_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return row is not None

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO messages (conversation_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (conversation_id, role, content, _now()),
            )
            self._conn.commit()

    def get_messages(self, conversation_id: str) -> list[Message]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM messages "
                "WHERE conversation_id = ? ORDER BY id ASC",
                (conversation_id,),
            ).fetchall()
        return [Message(role=row["role"], content=row["content"]) for row in rows]

    def save_fact(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO facts (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = excluded.updated_at
                """,
                (key, value, _now()),
            )
            self._conn.commit()

    def delete_fact(self, key: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM facts WHERE key = ?", (key,))
            self._conn.commit()
        return cursor.rowcount > 0

    def get_facts(self) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute("SELECT key, value FROM facts").fetchall()
        return {row["key"]: row["value"] for row in rows}
