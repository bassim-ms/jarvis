"""
SQLite-backed implementation of TokenStore, using Python's stdlib sqlite3 -
the exact same pattern as SQLiteReminderStore/SQLiteMemoryStore: one
connection held for the store's lifetime, guarded by a lock, backed by a
plain local file so a completed OAuth authorization survives a Rhasspy
restart with no special recovery code - `get()` simply returns whatever
was last `save()`d.

The refresh token is a sensitive credential (it grants ongoing account
access, not just a single request) - this file never logs it, never
prints it, and never includes it in an exception message. The database
file itself matches the existing `*.db` .gitignore rule (never committed),
the same trust boundary this codebase already gives `.env`; `_restrict_permissions`
best-effort tightens the file's own OS permissions on top of that (a
no-op on platforms/filesystems that don't support it, e.g. most Windows
setups - the .gitignore rule is the real, always-effective protection).
"""

import os
import sqlite3
import stat
import threading

from app.calendar.base import OAuthToken, TokenStore

#: The single stored token always lives at this fixed row id - there is
#: deliberately no keying by user/account (see TokenStore's docstring).
_ROW_ID = 1


class SQLiteTokenStore(TokenStore):
    """TokenStore backed by a local SQLite database file."""

    def __init__(self, db_path: str = "jarvis_calendar_token.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        self._restrict_permissions()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS oauth_token (
                    id INTEGER PRIMARY KEY,
                    access_token TEXT NOT NULL,
                    refresh_token TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    token_type TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    def _restrict_permissions(self) -> None:
        """Best-effort: restrict the token file to owner read/write only.
        A no-op (never raises) on a platform/filesystem that doesn't
        support POSIX-style permission bits, e.g. most Windows setups -
        the .gitignore rule (never committed) is what actually matters
        there; this is defense in depth where the OS honors it."""
        if self.db_path == ":memory:":
            return
        try:
            os.chmod(self.db_path, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass

    def get(self) -> OAuthToken | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM oauth_token WHERE id = ?", (_ROW_ID,)
            ).fetchone()
        if row is None:
            return None
        return OAuthToken(
            access_token=row["access_token"],
            refresh_token=row["refresh_token"],
            expires_at=row["expires_at"],
            scope=row["scope"],
            token_type=row["token_type"],
        )

    def save(self, token: OAuthToken) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO oauth_token
                    (id, access_token, refresh_token, expires_at, scope, token_type)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    access_token = excluded.access_token,
                    refresh_token = excluded.refresh_token,
                    expires_at = excluded.expires_at,
                    scope = excluded.scope,
                    token_type = excluded.token_type
                """,
                (
                    _ROW_ID,
                    token.access_token,
                    token.refresh_token,
                    token.expires_at,
                    token.scope,
                    token.token_type,
                ),
            )
            self._conn.commit()
        self._restrict_permissions()

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM oauth_token WHERE id = ?", (_ROW_ID,))
            self._conn.commit()
