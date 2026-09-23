"""
SQLite-backed SearchCache implementation.

Uses Python's stdlib sqlite3 directly, following the same pattern as
SQLiteMemoryStore (app/memory/sqlite_store.py): one connection held for the
store's lifetime, guarded by a lock since FastAPI can call in from more than
one thread, backed by a plain file so cached results survive Rhasspy
restarts. No Redis or other external service.
"""

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

from app.search.base import SearchResult
from app.search.cache import SearchCache


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteSearchCache(SearchCache):
    """SearchCache backed by a local SQLite database file."""

    def __init__(
        self, db_path: str = "jarvis_search_cache.db", ttl_seconds: int = 600
    ):
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS search_cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    def get(self, key: str) -> list[SearchResult] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value, expires_at FROM search_cache WHERE key = ?",
                (key,),
            ).fetchone()

        if row is None or row["expires_at"] < _now():
            return None

        return [SearchResult(**item) for item in json.loads(row["value"])]

    def set(self, key: str, results: list[SearchResult]) -> None:
        value = json.dumps(
            [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "source": r.source,
                    "query": r.query,
                }
                for r in results
            ]
        )
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=self.ttl_seconds)
        ).isoformat()

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO search_cache (key, value, expires_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    expires_at = excluded.expires_at
                """,
                (key, value, expires_at),
            )
            self._conn.commit()
