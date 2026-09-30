"""
The attention queue: where a notification waits when it must not interrupt.

The existing notification ledger records only THAT a key was claimed (for at-most-
once delivery) - it holds no content and so cannot hold anything back. This is the
one small store that can: a suppressed event is kept whole, so it can be presented
later, and nothing that was held back is ever lost.

One row per logical event key (a repeat of the same key is a duplicate, not a second
row). It preserves the ORIGINAL event, the importance and urgency signal it arrived
with, why it was held, when it arrived and when it expires. Like the ledger it lives
in the application's existing SQLite database (the connection and lock it is given),
or in memory for tests.

Statuses:
  silent      stored, not meant to be presented as a notification (low importance...)
  queued      waiting for a better moment or a channel
  escalation  URGENT, but no channel is allowed to carry it: waiting for one
  handed_off  given to the notification router (delivery itself is the router's)
  presented_in_chat  an urgent event nobody could be reached about, shown in the chat reply
  expired     its expiry passed, or it was no longer relevant, before it could be presented
"""

from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from app.notifications.attention import AttentionSignal, Importance, SignalSource
from app.notifications.base import NotificationEvent, NotificationKind, NotificationPriority

WAITING_STATUSES = ("queued", "escalation")


@dataclass(frozen=True)
class QueuedItem:
    event: NotificationEvent
    signal: AttentionSignal
    status: str
    level: str  # the Attention level that was decided (its name)
    reason: str
    channels: tuple[str, ...]
    created_at: str
    updated_at: str

    @property
    def key(self) -> str:
        return self.event.key


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _event_to_json(event: NotificationEvent) -> str:
    return json.dumps(
        {
            "key": event.key,
            "kind": event.kind.value,
            "priority": int(event.priority),
            "text": event.text,
            "conversation_id": event.conversation_id,
            "job_id": event.job_id,
            "interaction_code": event.interaction_code,
        }
    )


def _event_from_json(raw: str) -> NotificationEvent:
    d = json.loads(raw)
    return NotificationEvent(
        key=d["key"],
        kind=NotificationKind(d["kind"]),
        priority=NotificationPriority(d["priority"]),
        text=d["text"],
        conversation_id=d["conversation_id"],
        job_id=d["job_id"],
        interaction_code=d["interaction_code"],
    )


def _signal_to_json(signal: AttentionSignal) -> str:
    return json.dumps(
        {
            "importance": int(signal.importance),
            "source": signal.source.value,
            "urgent": signal.urgent,
            "time_critical": signal.time_critical,
            "deadline": _iso(signal.deadline),
            "expires_at": _iso(signal.expires_at),
        }
    )


def _signal_from_json(raw: str) -> AttentionSignal:
    d = json.loads(raw)
    return AttentionSignal(
        importance=Importance(d["importance"]),
        source=SignalSource(d["source"]),
        urgent=d["urgent"],
        time_critical=d["time_critical"],
        deadline=_parse(d["deadline"]),
        expires_at=_parse(d["expires_at"]),
    )


class AttentionQueue(ABC):
    @abstractmethod
    def add(self, item: QueuedItem) -> bool:
        """Store `item` under its key. False (nothing changed) if that key is already stored."""

    @abstractmethod
    def get(self, key: str) -> QueuedItem | None: ...

    @abstractmethod
    def update(
        self, key: str, *, status: str, level: str, reason: str, channels: tuple[str, ...], now: datetime
    ) -> None: ...

    @abstractmethod
    def waiting(self) -> list[QueuedItem]:
        """Items still waiting for presentation (queued / escalation), oldest first."""

    @abstractmethod
    def count(self, prefix: str) -> int:
        """How many stored keys start with `prefix` (every status)."""

    @abstractmethod
    def all(self) -> list[QueuedItem]:
        """Everything stored, oldest first."""


class InMemoryAttentionQueue(AttentionQueue):
    def __init__(self) -> None:
        self._items: dict[str, QueuedItem] = {}
        self._lock = threading.Lock()

    def add(self, item: QueuedItem) -> bool:
        with self._lock:
            if item.key in self._items:
                return False
            self._items[item.key] = item
            return True

    def get(self, key: str) -> QueuedItem | None:
        with self._lock:
            return self._items.get(key)

    def update(self, key, *, status, level, reason, channels, now) -> None:
        with self._lock:
            item = self._items.get(key)
            if item is not None:
                self._items[key] = replace(
                    item, status=status, level=level, reason=reason, channels=tuple(channels), updated_at=now.isoformat()
                )

    def waiting(self) -> list[QueuedItem]:
        with self._lock:
            return sorted(
                (i for i in self._items.values() if i.status in WAITING_STATUSES), key=lambda i: i.created_at
            )

    def all(self) -> list[QueuedItem]:
        with self._lock:
            return sorted(self._items.values(), key=lambda i: i.created_at)

    def count(self, prefix: str) -> int:
        with self._lock:
            return sum(1 for k in self._items if k.startswith(prefix))


class SQLiteAttentionQueue(AttentionQueue):
    """The queue in the application's existing SQLite database (same connection-and-lock
    sharing as SQLiteNotificationLedger)."""

    def __init__(self, connection: sqlite3.Connection, lock: threading.Lock):
        self._conn = connection
        self._lock = lock
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS attention_queue (
                    key        TEXT PRIMARY KEY,
                    event      TEXT NOT NULL,
                    signal     TEXT NOT NULL,
                    status     TEXT NOT NULL,
                    level      TEXT NOT NULL,
                    reason     TEXT NOT NULL,
                    channels   TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    @classmethod
    def for_job_store(cls, job_store) -> "SQLiteAttentionQueue":
        return cls(job_store.connection, job_store.lock)

    @staticmethod
    def _row(row) -> QueuedItem:
        return QueuedItem(
            event=_event_from_json(row[1]),
            signal=_signal_from_json(row[2]),
            status=row[3],
            level=row[4],
            reason=row[5],
            channels=tuple(json.loads(row[6])),
            created_at=row[7],
            updated_at=row[8],
        )

    def add(self, item: QueuedItem) -> bool:
        with self._lock:
            try:
                cursor = self._conn.execute(
                    "INSERT OR IGNORE INTO attention_queue "
                    "(key, event, signal, status, level, reason, channels, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        item.key, _event_to_json(item.event), _signal_to_json(item.signal), item.status,
                        item.level, item.reason, json.dumps(list(item.channels)), item.created_at, item.updated_at,
                    ),
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            return cursor.rowcount == 1

    def get(self, key: str) -> QueuedItem | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM attention_queue WHERE key = ?", (key,)).fetchone()
        return self._row(row) if row else None

    def update(self, key, *, status, level, reason, channels, now) -> None:
        with self._lock:
            try:
                self._conn.execute(
                    "UPDATE attention_queue SET status = ?, level = ?, reason = ?, channels = ?, updated_at = ? "
                    "WHERE key = ?",
                    (status, level, reason, json.dumps(list(channels)), now.isoformat(), key),
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def waiting(self) -> list[QueuedItem]:
        marks = ",".join("?" for _ in WAITING_STATUSES)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM attention_queue WHERE status IN ({marks}) ORDER BY created_at, rowid", WAITING_STATUSES
            ).fetchall()
        return [self._row(r) for r in rows]

    def all(self) -> list[QueuedItem]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM attention_queue ORDER BY created_at, rowid").fetchall()
        return [self._row(r) for r in rows]

    def count(self, prefix: str) -> int:
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM attention_queue WHERE key LIKE ? ESCAPE '\\'", (escaped + "%",)
            ).fetchone()
        return int(row[0])


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
