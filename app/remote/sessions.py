"""
Remote sessions, nonces and retried requests - all IN MEMORY, never on disk.

A session is created only after the Device Gateway authenticated a TRUSTED device's credential (app/remote/api.py). It holds
a random 256-bit key that is returned to the device once and then used only to verify request signatures (signing.py). A
session ends when it expires, when the device disconnects, when the user ends it or revokes the device on the dashboard, or
when Jarvis restarts (a device then simply creates a new session with its credential - nothing to clean up, nothing left on
disk to steal).
"""

from __future__ import annotations

import secrets
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class Session:
    session_id: str
    device_id: str
    key: bytes
    created_at: datetime
    expires_at: datetime
    last_used: datetime
    requests: int = 0


class RemoteSessions:
    """Thread-safe store of live sessions; at most `max_per_device` per device (the oldest is ended to make room)."""

    def __init__(self, ttl: timedelta, max_per_device: int = 5):
        self.ttl = ttl
        self.max_per_device = max(1, int(max_per_device))
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, device_id: str, now: datetime) -> Session:
        session = Session(session_id="rs_" + secrets.token_urlsafe(18), device_id=device_id, key=secrets.token_bytes(32),
                          created_at=now, expires_at=now + self.ttl, last_used=now)
        with self._lock:
            self._purge(now)
            mine = sorted((s for s in self._sessions.values() if s.device_id == device_id), key=lambda s: s.created_at)
            for old in mine[: max(0, len(mine) - self.max_per_device + 1)]:
                del self._sessions[old.session_id]
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str, now: datetime) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None and session.expires_at <= now:
                del self._sessions[session_id]
                return None
            return session

    def touch(self, session: Session, now: datetime) -> None:
        with self._lock:
            session.last_used = now
            session.requests += 1

    def end(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def end_device(self, device_id: str) -> int:
        with self._lock:
            gone = [sid for sid, s in self._sessions.items() if s.device_id == device_id]
            for sid in gone:
                del self._sessions[sid]
            return len(gone)

    def active(self, now: datetime) -> list[Session]:
        with self._lock:
            self._purge(now)
            return sorted(self._sessions.values(), key=lambda s: s.created_at)

    def _purge(self, now: datetime) -> None:
        for sid in [sid for sid, s in self._sessions.items() if s.expires_at <= now]:
            del self._sessions[sid]


class NonceCache:
    """Remembers (scope, nonce) pairs for `window`: a second use inside it is a replay. Bounded: when full, the oldest are
    dropped - which is safe because a nonce older than the window is refused anyway by the timestamp check."""

    def __init__(self, window: timedelta, max_entries: int = 100_000):
        self.window = window
        self.max_entries = max_entries
        self._seen: OrderedDict[tuple[str, str], datetime] = OrderedDict()
        self._lock = threading.Lock()

    def first_use(self, scope: str, nonce: str, now: datetime) -> bool:
        key = (scope, nonce)
        with self._lock:
            while self._seen:
                oldest_key, seen_at = next(iter(self._seen.items()))
                if now - seen_at <= self.window and len(self._seen) < self.max_entries:
                    break
                del self._seen[oldest_key]
            if key in self._seen:
                return False
            self._seen[key] = now
            return True


class ReplyCache:
    """The reply already given for (device, request id), for `ttl`: a device that lost the connection and RETRIES the same
    request - even on a new session - gets the same reply instead of Jarvis doing it twice. Bounded."""

    def __init__(self, ttl: timedelta, max_entries: int = 2_000):
        self.ttl = ttl
        self.max_entries = max_entries
        self._replies: OrderedDict[tuple[str, str], tuple[datetime, dict]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, device_id: str, request_id: str, now: datetime) -> dict | None:
        with self._lock:
            found = self._replies.get((device_id, request_id))
            if found is None or now - found[0] > self.ttl:
                return None
            return found[1]

    def put(self, device_id: str, request_id: str, reply: dict, now: datetime) -> None:
        with self._lock:
            self._replies[(device_id, request_id)] = (now, reply)
            while len(self._replies) > self.max_entries:
                self._replies.popitem(last=False)
