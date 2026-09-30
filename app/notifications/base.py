"""
Notifications: a small, channel-neutral way for Jarvis to tell the user
something.

The Job/Interaction system asks for exactly one thing -
`router.notify(user, event)` - and never talks to a toast, a speaker or a
phone directly. A `NotificationEvent` is plain, finished, channel-neutral
text plus a priority: whatever delivers it (a desktop toast today; speech, a
phone, a watch or glasses later) receives the same event and can speak or
show the very same words. Adding a channel is `router.add_channel(...)`; no
Job code changes.

Guarantees, all deliberately boring:

- **Non-critical.** `notify` never raises and never blocks on a channel. A
  channel that throws or hangs affects only its own delivery attempt, which
  runs on its own daemon thread.
- **At most once per logical event.** Every event carries a stable `key`
  (e.g. "interaction:<id>:reminder:2"). The router first CLAIMS the key in a
  persistent ledger (a primary-key insert); only the caller that wins the
  claim delivers. Engine ticks, retries, duplicate polls and a process
  restart therefore cannot repeat a notification. The price of at-most-once
  is that a crash between the claim and the delivery loses that one
  notification - never repeats it.
- **Text is finished before it gets here.** The router does not compose or
  rewrite anything; events are validated (bounded, one line, no control
  characters) so a channel can rely on that.

PRIORITIES. LOW, NORMAL, HIGH. The numeric gaps leave room for a future
CRITICAL above HIGH without renumbering anything; it is intentionally not
defined yet. Each channel declares the lowest priority it accepts.

Nothing here resolves, confirms, executes or changes anything: delivering a
notification is output only.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum

#: The single local user. `notify(user, event)` takes a user so a future
#: multi-user or multi-device setup needs no signature change.
DEFAULT_USER = "owner"

MAX_KEY_CHARS = 200
MAX_TEXT_CHARS = 500


class NotificationPriority(IntEnum):
    LOW = 10  # informational: fine to miss
    NORMAL = 20  # the user's attention is wanted, but nothing is on fire
    HIGH = 30  # something failed or is about to lapse
    # CRITICAL = 40 is reserved for later; nothing may assume HIGH is the top.


class NotificationKind(Enum):
    WAITING = "waiting"  # a question/confirmation was just opened
    REMINDER = "reminder"
    EXPIRY_WARNING = "expiry_warning"
    EXPIRED = "expired"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class NotificationEvent:
    """One logical notification, ready to be spoken or shown.

    `key` identifies the LOGICAL event for deduplication. `text` is the whole
    message, addressed to the user, understandable read or heard. `job_id` /
    `conversation_id` / `interaction_code` are routing metadata for channels
    (they are not part of `text`)."""

    key: str
    kind: NotificationKind
    priority: NotificationPriority
    text: str
    conversation_id: str | None = None
    job_id: str | None = None
    interaction_code: str | None = None
    #: Structured metadata set ONLY by trusted code that KNOWS the event is time-critical
    #: (never derived from `text`, which is display only). It is the one thing that lets the
    #: attention layer consider an event URGENT (and then only together with HIGH importance
    #: and a trusted source - see app/notifications/attention.py). No existing producer sets it.
    time_critical: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.time_critical, bool):
            raise ValueError("time_critical must be a bool.")
        if not isinstance(self.key, str) or not self.key or len(self.key) > MAX_KEY_CHARS:
            raise ValueError(f"key must be a non-empty string of at most {MAX_KEY_CHARS} characters.")
        if not isinstance(self.kind, NotificationKind):
            raise ValueError("kind must be a NotificationKind.")
        if not isinstance(self.priority, NotificationPriority):
            raise ValueError("priority must be a NotificationPriority.")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be non-empty.")
        if len(self.text) > MAX_TEXT_CHARS:
            raise ValueError(f"text can be at most {MAX_TEXT_CHARS} characters.")
        if any(unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") for ch in self.text):
            raise ValueError("text must be a single line without control characters.")


class NotificationChannel(ABC):
    """One way of reaching the user. `deliver` should raise on failure; the
    router isolates and records it. `min_priority` is the lowest priority this
    channel is willing to carry (a desktop toast may refuse LOW)."""

    name: str = ""
    min_priority: NotificationPriority = NotificationPriority.LOW

    def accepts(self, event: NotificationEvent) -> bool:
        return event.priority >= self.min_priority

    @abstractmethod
    def deliver(self, user: str, event: NotificationEvent) -> None:
        """Present `event` to `user`. Output only."""


# --- the delivery ledger -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class LedgerEntry:
    key: str
    kind: str
    priority: int
    job_id: str | None
    claimed_at: str
    delivered: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()


class NotificationLedger(ABC):
    """Persistent record of which logical events were claimed for delivery."""

    @abstractmethod
    def claim(self, event: NotificationEvent, now: datetime) -> bool:
        """Atomically record `event.key`. True only for the caller that
        recorded it first; False if it was already claimed."""

    @abstractmethod
    def record_delivery(self, key: str, channel: str, ok: bool) -> None: ...

    @abstractmethod
    def get(self, key: str) -> LedgerEntry | None: ...

    @abstractmethod
    def count(self, prefix: str) -> int:
        """How many claimed keys start with `prefix`."""


class InMemoryNotificationLedger(NotificationLedger):
    """Non-persistent ledger (tests, or when no database is wanted)."""

    def __init__(self) -> None:
        self._entries: dict[str, LedgerEntry] = {}
        self._lock = threading.Lock()

    def claim(self, event: NotificationEvent, now: datetime) -> bool:
        with self._lock:
            if event.key in self._entries:
                return False
            self._entries[event.key] = LedgerEntry(
                event.key, event.kind.value, int(event.priority), event.job_id, now.isoformat()
            )
            return True

    def record_delivery(self, key: str, channel: str, ok: bool) -> None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            delivered, failed = list(entry.delivered), list(entry.failed)
            (delivered if ok else failed).append(channel)
            self._entries[key] = LedgerEntry(
                entry.key, entry.kind, entry.priority, entry.job_id, entry.claimed_at,
                tuple(delivered), tuple(failed),
            )

    def get(self, key: str) -> LedgerEntry | None:
        with self._lock:
            return self._entries.get(key)

    def count(self, prefix: str) -> int:
        with self._lock:
            return sum(1 for k in self._entries if k.startswith(prefix))


class SQLiteNotificationLedger(NotificationLedger):
    """The ledger in the application's existing SQLite database. It uses the
    connection and lock it is given (in the app, the Job store's, so there is
    one database file and one writer discipline); it holds no lock while a
    channel delivers."""

    def __init__(self, connection: sqlite3.Connection, lock: threading.Lock):
        self._conn = connection
        self._lock = lock
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notifications (
                    key        TEXT PRIMARY KEY,
                    kind       TEXT NOT NULL,
                    priority   INTEGER NOT NULL,
                    job_id     TEXT,
                    claimed_at TEXT NOT NULL,
                    delivered  TEXT NOT NULL DEFAULT '[]',
                    failed     TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            self._conn.commit()

    @classmethod
    def for_job_store(cls, job_store) -> "SQLiteNotificationLedger":
        return cls(job_store.connection, job_store.lock)

    def claim(self, event: NotificationEvent, now: datetime) -> bool:
        with self._lock:
            try:
                cursor = self._conn.execute(
                    "INSERT OR IGNORE INTO notifications (key, kind, priority, job_id, claimed_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (event.key, event.kind.value, int(event.priority), event.job_id, now.isoformat()),
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            return cursor.rowcount == 1

    def record_delivery(self, key: str, channel: str, ok: bool) -> None:
        column = "delivered" if ok else "failed"
        with self._lock:
            try:
                row = self._conn.execute(
                    f"SELECT {column} FROM notifications WHERE key = ?", (key,)
                ).fetchone()
                if row is None:
                    return
                names = json.loads(row[0])
                names.append(channel)
                self._conn.execute(
                    f"UPDATE notifications SET {column} = ? WHERE key = ?", (json.dumps(names), key)
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def get(self, key: str) -> LedgerEntry | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT key, kind, priority, job_id, claimed_at, delivered, failed "
                "FROM notifications WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return LedgerEntry(
            key=row[0], kind=row[1], priority=row[2], job_id=row[3], claimed_at=row[4],
            delivered=tuple(json.loads(row[5])), failed=tuple(json.loads(row[6])),
        )

    def count(self, prefix: str) -> int:
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM notifications WHERE key LIKE ? ESCAPE '\\'", (escaped + "%",)
            ).fetchone()
        return int(row[0])


# --- the router ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DeliveryResult:
    """What `notify` did. `accepted` means this call won the claim and handed
    the event to `channels`; it does NOT mean a channel succeeded (delivery is
    asynchronous and recorded in the ledger). `duplicate` means the event was
    already claimed earlier and nothing was sent."""

    accepted: bool
    duplicate: bool = False
    channels: tuple[str, ...] = field(default_factory=tuple)


def _run_in_daemon_thread(work: Callable[[], None]) -> None:
    threading.Thread(target=work, name="jarvis-notification", daemon=True).start()


class NotificationRouter:
    """Delivers each logical event at most once, to every channel that
    accepts its priority, without ever raising or waiting on a channel.

    `dispatcher` runs one channel's delivery (default: its own daemon
    thread, so a hung channel can never stall the caller or another channel);
    tests pass a synchronous one."""

    def __init__(
        self,
        channels: Iterable[NotificationChannel] = (),
        ledger: NotificationLedger | None = None,
        *,
        dispatcher: Callable[[Callable[[], None]], None] | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        self._channels: list[NotificationChannel] = list(channels)
        self.ledger = ledger or InMemoryNotificationLedger()
        self._dispatch = dispatcher or _run_in_daemon_thread
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @property
    def channels(self) -> tuple[NotificationChannel, ...]:
        return tuple(self._channels)

    def add_channel(self, channel: NotificationChannel) -> None:
        self._channels.append(channel)

    def already_sent(self, key: str) -> bool:
        """Whether `key` was ever claimed (best-effort: False if the ledger
        can't be read)."""
        try:
            return self.ledger.get(key) is not None
        except Exception:  # noqa: BLE001
            return False

    def sent_count(self, prefix: str) -> int:
        try:
            return self.ledger.count(prefix)
        except Exception:  # noqa: BLE001
            return 0

    def notify(
        self, user: str, event: NotificationEvent, *, only: Iterable[str] | None = None
    ) -> DeliveryResult:
        """Claim `event.key`, then hand the event to each accepting channel.
        NEVER raises: a broken ledger, dispatcher or channel just means this
        notification is not (fully) delivered - the caller (a Job) is
        unaffected.

        `only` (default None: every accepting channel, exactly as always) restricts
        delivery to the channels of those names - how the attention policy keeps an
        event off a channel it must not use. An empty `only` delivers nowhere and, like
        any call that could reach no channel, claims nothing."""
        if only is not None:
            allowed = set(only)
            if not allowed:
                return DeliveryResult(accepted=False)
        try:
            if not isinstance(user, str) or not user or not isinstance(event, NotificationEvent):
                return DeliveryResult(accepted=False)
            if not self.ledger.claim(event, self._clock()):
                return DeliveryResult(accepted=False, duplicate=True)
        except Exception:  # noqa: BLE001 - if it can't be claimed it can't be de-duplicated: send nothing
            return DeliveryResult(accepted=False)

        chosen: list[str] = []
        for channel in list(self._channels):
            try:
                if only is not None and channel.name not in allowed:
                    continue
                if not channel.accepts(event):
                    continue
                self._dispatch(self._delivery(channel, user, event))
                chosen.append(channel.name)
            except Exception:  # noqa: BLE001 - one channel must not affect another
                self._record(event.key, getattr(channel, "name", "?"), ok=False)
        return DeliveryResult(accepted=True, channels=tuple(chosen))

    def _delivery(
        self, channel: NotificationChannel, user: str, event: NotificationEvent
    ) -> Callable[[], None]:
        def deliver() -> None:
            try:
                channel.deliver(user, event)
            except Exception:  # noqa: BLE001 - recorded, never propagated
                self._record(event.key, channel.name, ok=False)
            else:
                self._record(event.key, channel.name, ok=True)

        return deliver

    def _record(self, key: str, channel: str, *, ok: bool) -> None:
        try:
            self.ledger.record_delivery(key, channel, ok)
        except Exception:  # noqa: BLE001 - bookkeeping must never surface either
            pass
