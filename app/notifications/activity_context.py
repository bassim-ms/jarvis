"""
Explicit, temporary activity context - the first (and so far only) source of "what is the
user doing right now?" for the attention policy.

    "I'm in class."   "I'm studying."   "I'm in a meeting until 3 PM."   "I'm free now."

The user says it; Jarvis records it as a TEMPORARY context (a value, when it was set, when
it ends, and where it came from) and the attention gate reads it. Nothing is detected,
inferred or guessed, and nothing here is memory: it is not saved as a fact, a preference or
a World Model entity (the memory write policy never sees it), and it ends on its own.

Later sources (calendar, a device, a sensor) feed the same thing: something that answers
`current_name()` with a context name.

WHAT COUNTS AS A STATEMENT. The whole message must be the statement - a first-person present
tense one, optionally with a duration ("for the next 50 minutes", "for an hour") or an end
time ("until 3 PM"). A question, a future or past tense ("I was in class", "I'm in class
tomorrow"), or a sentence that merely contains the words ("I'm studying the history of Rome,
can you explain it?") is ordinary conversation and sets nothing.

DURATIONS ARE BOUNDED. With no duration a context lasts a per-activity default (an hour or so),
and however long is asked for it never lasts more than MAX_DURATION: a forgotten "I'm in class"
cannot silence Jarvis for days. "I'm free now" ends the current context at once. The user is
never asked a clarification for an ordinary context-setting sentence.

Contexts: class, studying, meeting, focused_work (all held-back contexts, see attention.py),
and free (which simply clears). Anything else the policy sees is "unknown" and follows the
ordinary rules.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

CLASS, STUDYING, MEETING, FOCUSED_WORK, FREE = "class", "studying", "meeting", "focused_work", "free"

DEFAULT_DURATIONS = {
    CLASS: timedelta(minutes=60),
    STUDYING: timedelta(minutes=90),
    MEETING: timedelta(minutes=60),
    FOCUSED_WORK: timedelta(minutes=90),
}
MAX_DURATION = timedelta(hours=12)
MIN_DURATION = timedelta(minutes=1)
SOURCE_USER_STATEMENT = "user_statement"

_I = re.IGNORECASE


@dataclass(frozen=True)
class ActivityContext:
    """One temporary context. `source` is its provenance ("user_statement" today)."""

    value: str
    created_at: str
    expires_at: str
    source: str
    conversation_id: str | None = None

    def active(self, now: datetime) -> bool:
        return now < datetime.fromisoformat(self.expires_at)


class ActivityContextStore(ABC):
    @abstractmethod
    def set(self, context: ActivityContext) -> None: ...

    @abstractmethod
    def get(self) -> ActivityContext | None:
        """The stored context, expired or not."""

    @abstractmethod
    def clear(self) -> None: ...


class InMemoryActivityContextStore(ActivityContextStore):
    def __init__(self) -> None:
        self._context: ActivityContext | None = None
        self._lock = threading.Lock()

    def set(self, context: ActivityContext) -> None:
        with self._lock:
            self._context = context

    def get(self) -> ActivityContext | None:
        with self._lock:
            return self._context

    def clear(self) -> None:
        with self._lock:
            self._context = None


class SQLiteActivityContextStore(ActivityContextStore):
    """A single row in the application's existing database (the connection and lock it is given),
    so a restart inside the context's lifetime does not forget it."""

    def __init__(self, connection: sqlite3.Connection, lock: threading.Lock):
        self._conn = connection
        self._lock = lock
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS activity_context (
                    id              INTEGER PRIMARY KEY CHECK (id = 1),
                    value           TEXT NOT NULL,
                    created_at      TEXT NOT NULL,
                    expires_at      TEXT NOT NULL,
                    source          TEXT NOT NULL,
                    conversation_id TEXT
                )
                """
            )
            self._conn.commit()

    @classmethod
    def for_job_store(cls, job_store) -> "SQLiteActivityContextStore":
        return cls(job_store.connection, job_store.lock)

    def set(self, context: ActivityContext) -> None:
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT OR REPLACE INTO activity_context (id, value, created_at, expires_at, source, conversation_id) "
                    "VALUES (1, ?, ?, ?, ?, ?)",
                    (context.value, context.created_at, context.expires_at, context.source, context.conversation_id),
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def get(self) -> ActivityContext | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value, created_at, expires_at, source, conversation_id FROM activity_context WHERE id = 1"
            ).fetchone()
        return ActivityContext(*row) if row else None

    def clear(self) -> None:
        with self._lock:
            try:
                self._conn.execute("DELETE FROM activity_context")
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise


# --- reading the user's own words ----------------------------------------------------------------------------------------

_LEAD = r"^\s*(?:(?:ok|okay|hey|hi|so|well|jarvis)[,.!\s]+)*(?:i'?m|i\s+am|im)\s+(?:currently\s+|now\s+)?"
_FOR = r"(?:\s+for\s+(?:my|the|a|an)\s+(?!(?:next\s+)?(?:half|hours?|hrs?|minutes?|mins?)\b)[a-z][a-z ]{0,30}?)?"
_ACTIVITIES = (
    (CLASS, r"(?:in|at)\s+(?:a\s+|my\s+|the\s+)?(?:class|lecture|lesson)"),
    (MEETING, r"(?:in|at)\s+(?:a\s+|my\s+|the\s+)?(?:meeting|call|interview)"),
    (STUDYING, r"(?:studying|revising|doing\s+(?:my\s+)?homework)" + _FOR),
    (FOCUSED_WORK, r"(?:focusing|in\s+focus\s+mode|in\s+deep\s+work|doing\s+(?:deep|focused)\s+work)"),
)
_DURATION = (
    r"(?:\s+for\s+(?:the\s+next\s+)?(?:(?P<half>half\s+an?\s+hour)|(?P<n>\d+(?:\.\d+)?|an?)\s*(?P<unit>minutes?|mins?|hours?|hrs?))"
    r"|\s+until\s+(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ap>am|pm)?)?"
)
_END = r"\s*[.!]*\s*$"
_ACTIVITY_RES = tuple((name, re.compile(_LEAD + f"(?P<act>{body})" + _DURATION + _END, _I)) for name, body in _ACTIVITIES)
_FREE_RE = re.compile(
    r"^\s*(?:(?:ok|okay|hey|hi|so|well|jarvis)[,.!\s]+)*(?:"
    r"(?:i'?m|i\s+am|im)\s+(?:free|available|back|done|finished)(?:\s+now)?"
    r"(?:\s+with\s+(?:class|the\s+lecture|studying|the\s+meeting|my\s+meeting))?"
    r"|(?:the\s+|my\s+)?(?:class|lecture|meeting|call|studying)\s+is\s+(?:over|done|finished)"
    r"|(?:i'?m|i\s+am|im)\s+out\s+of\s+(?:class|a\s+meeting|the\s+meeting|my\s+meeting)"
    r")" + _END,
    _I,
)


@dataclass(frozen=True)
class ActivityStatement:
    """What the user's sentence said: the context (FREE clears), and how long it lasts as either a
    duration or a wall-clock end time - or neither (then the activity's default applies)."""

    value: str
    duration: timedelta | None = None
    until: tuple[int, int, str | None] | None = None  # (hour, minute, "am"/"pm"/None)


def parse_activity_statement(message: object) -> ActivityStatement | None:
    """The activity statement `message` is, in its entirety, or None."""
    if not isinstance(message, str) or not message.strip() or len(message) > 200 or "?" in message:
        return None
    if _FREE_RE.match(message):
        return ActivityStatement(FREE)
    for name, regex in _ACTIVITY_RES:
        m = regex.match(message)
        if not m:
            continue
        if m.group("half"):
            return ActivityStatement(name, duration=timedelta(minutes=30))
        if m.group("unit"):
            raw = m.group("n").lower()
            amount = 1.0 if raw in ("a", "an") else float(raw)
            unit = m.group("unit").lower()
            minutes = amount * (60 if unit.startswith(("h")) else 1)
            return ActivityStatement(name, duration=timedelta(minutes=minutes))
        if m.group("h"):
            hour, minute = int(m.group("h")), int(m.group("m") or 0)
            if not (0 <= minute < 60) or hour > 24 or (m.group("ap") and not 1 <= hour <= 12):
                return None
            return ActivityStatement(name, until=(hour % 24 if not m.group("ap") else hour, minute, (m.group("ap") or "").lower() or None))
        return ActivityStatement(name)
    return None


def _next_occurrence(now: datetime, hour: int, minute: int, meridiem: str | None) -> datetime:
    """The soonest FUTURE moment (in `now`'s own timezone) the wall clock shows this time."""
    if meridiem == "pm":
        hours = [(hour % 12) + 12]
    elif meridiem == "am":
        hours = [hour % 12]
    elif hour > 12 or hour == 0:
        hours = [hour % 24]
    else:
        hours = [hour % 12, (hour % 12) + 12]  # "until 3": the next 3 o'clock, day or night
    candidates = []
    for h in hours:
        moment = now.replace(hour=h, minute=minute, second=0, microsecond=0)
        candidates.append(moment if moment > now else moment + timedelta(days=1))
    return min(candidates)


def resolve_expiry(statement: ActivityStatement, now: datetime) -> datetime:
    """When the context ends: the stated duration or end time, else the activity's default - and
    never sooner than a minute nor later than MAX_DURATION from `now`."""
    if statement.duration is not None:
        length = statement.duration
    elif statement.until is not None:
        length = _next_occurrence(now, *statement.until) - now
    else:
        length = DEFAULT_DURATIONS.get(statement.value, timedelta(minutes=60))
    return now + max(MIN_DURATION, min(length, MAX_DURATION))


class ActivityContextService:
    """Records the user's stated activity and answers "what is it right now?". `on_change` (optional,
    zero-arg, may return how many held notifications it released) is called whenever the context is set or
    cleared - the hook by which held-back notifications are reconsidered the moment the user says
    they are free."""

    def __init__(
        self,
        store: ActivityContextStore,
        *,
        clock: Callable[[], datetime],
        on_change: Callable[[], object] | None = None,
    ):
        self.store = store
        self._clock = clock
        self.on_change = on_change

    def current(self) -> ActivityContext | None:
        """The context in force now (None if there is none or it has ended)."""
        context = self.store.get()
        return context if context is not None and context.active(self._clock()) else None

    def current_name(self) -> str:
        """For the attention gate: the context name, or "unknown" - which follows the ordinary rules."""
        context = self.current()
        return context.value if context is not None else "unknown"

    def command_reply(self, message: str, conversation_id: str | None = None) -> str | None:
        """If `message` is an activity statement, apply it and return what to tell the user; else None."""
        statement = parse_activity_statement(message)
        if statement is None:
            return None
        now = self._clock()
        if statement.value == FREE:
            self.store.clear()
            released = self._changed()
            base = "Welcome back - I'll go back to telling you things as they come."
            return base + (f" I've passed on {released} held notification{'s' if released != 1 else ''}." if released else "")
        expires = resolve_expiry(statement, now)
        self.store.set(
            ActivityContext(
                value=statement.value,
                created_at=now.isoformat(),
                expires_at=expires.isoformat(),
                source=SOURCE_USER_STATEMENT,
                conversation_id=conversation_id,
            )
        )
        self._changed()
        minutes = max(1, round((expires - now).total_seconds() / 60))
        return (
            f"Understood - I'll keep non-urgent notifications quiet for about {minutes} minute{'s' if minutes != 1 else ''}. "
            "Say \"I'm free now\" whenever you're done."
        )

    def _changed(self) -> int:
        if self.on_change is None:
            return 0
        try:
            result = self.on_change()
        except Exception:  # noqa: BLE001 - reconsidering held notifications must never break the reply
            return 0
        return result if isinstance(result, int) and not isinstance(result, bool) else 0
