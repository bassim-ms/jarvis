"""
Per-provider statistics, learned only from the requests Jarvis makes anyway - spec 70 (provider health monitoring), 71 (latency
measurement), 72 (reliability scoring), 73 / 74 / 885 (cost and quota tracking: token and request counts; see below). Nothing
here ever sends a request of its own.

WHAT IS RECORDED, for every REAL attempt at a provider (one skipped because it is resting or disabled was not asked, so nothing
is recorded): when, whether it answered, how long it took, the KIND of a failure (an ErrorCategory value such as "rate_limit" -
never the error's text, which can quote a provider's response) and the token counts the provider itself reported (numbers
only). Never a key, URL, prompt, reply or error message - so a snapshot, and the database row behind it, is safe to show.

RELIABILITY is the share of the last RECENT_WINDOW (50) actual attempts that got an answer, always reported together with the
sample size it is based on. Before any attempt it is None ("no data"), never an invented 100%.

PERSISTENCE: each provider's state is one JSON row in a small SQLite database (`SQLiteProviderStatsStore`, setting
`provider_stats_db_path`), saved after every recorded attempt and loaded when the manager is built - so the statistics survive a
restart. A row that is unreadable, from a newer format or internally inconsistent is not trusted: that provider starts again from
empty (and the problem is logged by type, never by content).

COST: token counts are recorded per day; a money cost is NOT computed - the free tiers cost nothing and a paid route's price
(OpenRouter's "auto") varies per model, so a figure would be invented. QUOTA: requests and tokens per day, rate-limit hits, and
- when the user configures one - usage against a daily request limit; provider-side limits are not known otherwise.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime

_log = logging.getLogger("jarvis.providers")

#: How many of the most recent actual attempts the reliability score covers.
RECENT_WINDOW = 50
#: Weight of the newest successful attempt in the smoothed average latency (exponential moving average).
LATENCY_SMOOTHING = 0.3
#: How many days of per-day counts are kept.
DAYS_KEPT = 31
STATE_VERSION = 1
#: Failure kinds a state may hold: ErrorCategory values plus "unexpected" (an adapter bug) - anything else is not trusted.
KNOWN_KINDS = frozenset({"transient", "rate_limit", "network", "auth", "other", "unexpected"})


class InvalidStatsState(ValueError):
    """A stored state that cannot be trusted (unreadable, newer format, inconsistent)."""


@dataclass
class Attempt:
    at: str  # ISO 8601 with offset
    ok: bool
    latency_ms: int
    kind: str | None = None  # failure kind; None for an answered attempt
    tokens: int | None = None  # total tokens the provider reported, if it did


@dataclass
class DayCounts:
    requests: int = 0
    failures: int = 0
    rate_limited: int = 0
    tokens: int = 0


@dataclass
class ProviderStats:
    attempts: int = 0
    successes: int = 0
    failures: int = 0
    failures_by_kind: dict[str, int] = field(default_factory=dict)
    rate_limited: int = 0
    tokens: int = 0
    #: Smoothed over ANSWERED attempts only: a failure can be instant (refused) or a full timeout, neither says how fast the
    #: provider answers.
    avg_latency_ms: float | None = None
    last_latency_ms: int | None = None
    last_success_at: str | None = None
    last_failure_at: str | None = None
    last_failure_kind: str | None = None
    recent: deque = field(default_factory=lambda: deque(maxlen=RECENT_WINDOW))
    days: dict[str, DayCounts] = field(default_factory=dict)

    # -- recording ------------------------------------------------------------------------------------------------------

    def record(self, ok: bool, latency_ms: float, at: datetime, kind: str | None = None, tokens: int | None = None) -> None:
        """One real attempt. `kind` is an ErrorCategory value for a failure (never the error's text); `tokens` the total the
        provider reported, if any."""
        latency = max(0, int(round(float(latency_ms))))
        tokens = int(tokens) if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0 else None
        stamp = at.isoformat()
        day = self.days.setdefault(at.date().isoformat(), DayCounts())
        self.attempts += 1
        self.last_latency_ms = latency
        day.requests += 1
        if tokens:
            self.tokens += tokens
            day.tokens += tokens
        if ok:
            self.successes += 1
            self.last_success_at = stamp
            self.avg_latency_ms = (
                float(latency) if self.avg_latency_ms is None
                else LATENCY_SMOOTHING * latency + (1 - LATENCY_SMOOTHING) * self.avg_latency_ms
            )
            self.recent.append(Attempt(at=stamp, ok=True, latency_ms=latency, tokens=tokens))
        else:
            kind = kind if kind in KNOWN_KINDS else "other"
            self.failures += 1
            self.failures_by_kind[kind] = self.failures_by_kind.get(kind, 0) + 1
            self.last_failure_at = stamp
            self.last_failure_kind = kind
            day.failures += 1
            if kind == "rate_limit":
                self.rate_limited += 1
                day.rate_limited += 1
            self.recent.append(Attempt(at=stamp, ok=False, latency_ms=latency, kind=kind, tokens=tokens))
        self._prune_days(at.date())

    def _prune_days(self, today: date) -> None:
        for key in sorted(self.days):
            try:
                age = (today - date.fromisoformat(key)).days
            except ValueError:
                age = DAYS_KEPT + 1
            if age >= DAYS_KEPT:
                del self.days[key]

    # -- reading --------------------------------------------------------------------------------------------------------

    @property
    def reliability(self) -> float | None:
        """Share of the recent actual attempts that got an answer (0.0-1.0), or None when there are none."""
        if not self.recent:
            return None
        return sum(1 for attempt in self.recent if attempt.ok) / len(self.recent)

    def snapshot(self, now: datetime, daily_request_limit: int | None = None) -> dict:
        """A safe, JSON-ready view as of `now` (whose local date decides what "today" is)."""
        today = self.days.get(now.date().isoformat(), DayCounts())
        answered = sum(1 for attempt in self.recent if attempt.ok)
        reliability = self.reliability
        limit = daily_request_limit if daily_request_limit and daily_request_limit > 0 else None
        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "failures_by_kind": dict(sorted(self.failures_by_kind.items())),
            "rate_limited": self.rate_limited,
            # Reliability is ALWAYS paired with the sample it is based on (up to the last 50 actual attempts).
            "reliability": None if reliability is None else round(reliability, 3),
            "reliability_sample": len(self.recent),
            "reliability_answered": answered,
            "recent_window": RECENT_WINDOW,
            "avg_latency_ms": None if self.avg_latency_ms is None else round(self.avg_latency_ms),
            "last_latency_ms": self.last_latency_ms,
            "last_success_at": self.last_success_at,
            "last_failure_at": self.last_failure_at,
            "last_failure_kind": self.last_failure_kind,
            "tokens_total": self.tokens,
            "today": {
                "date": now.date().isoformat(),
                "requests": today.requests,
                "failures": today.failures,
                "rate_limited": today.rate_limited,
                "tokens": today.tokens,
                "request_limit": limit,
                "requests_left": None if limit is None else max(0, limit - today.requests),
            },
            "cost": None,  # deliberately not estimated - see the module docstring
        }

    # -- persistence ----------------------------------------------------------------------------------------------------

    def to_state(self) -> dict:
        return {
            "version": STATE_VERSION,
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "failures_by_kind": dict(self.failures_by_kind),
            "rate_limited": self.rate_limited,
            "tokens": self.tokens,
            "avg_latency_ms": self.avg_latency_ms,
            "last_latency_ms": self.last_latency_ms,
            "last_success_at": self.last_success_at,
            "last_failure_at": self.last_failure_at,
            "last_failure_kind": self.last_failure_kind,
            "recent": [
                {"at": a.at, "ok": a.ok, "latency_ms": a.latency_ms, "kind": a.kind, "tokens": a.tokens} for a in self.recent
            ],
            "days": {
                key: {"requests": d.requests, "failures": d.failures, "rate_limited": d.rate_limited, "tokens": d.tokens}
                for key, d in sorted(self.days.items())
            },
        }

    @classmethod
    def from_state(cls, state: object) -> "ProviderStats":
        """Rebuild from `to_state()` output. Raises InvalidStatsState for anything that cannot be trusted - wrong types,
        negative or impossible counts, an unknown failure kind, a newer format - rather than showing made-up numbers."""
        if not isinstance(state, dict):
            raise InvalidStatsState("not a mapping")
        if state.get("version") != STATE_VERSION:
            raise InvalidStatsState(f"unsupported version {state.get('version')!r}")

        def count(key: str, source: dict = state) -> int:
            value = source.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise InvalidStatsState(f"bad count {key}")
            return value

        def text_or_none(key: str) -> str | None:
            value = state.get(key)
            if value is not None and not isinstance(value, str):
                raise InvalidStatsState(f"bad text {key}")
            return value

        stats = cls(
            attempts=count("attempts"), successes=count("successes"), failures=count("failures"),
            rate_limited=count("rate_limited"), tokens=count("tokens"),
            last_success_at=text_or_none("last_success_at"), last_failure_at=text_or_none("last_failure_at"),
            last_failure_kind=text_or_none("last_failure_kind"),
        )
        if stats.successes + stats.failures != stats.attempts or stats.rate_limited > stats.failures:
            raise InvalidStatsState("counts do not add up")
        kinds = state.get("failures_by_kind")
        if not isinstance(kinds, dict) or any(k not in KNOWN_KINDS for k in kinds):
            raise InvalidStatsState("bad failure kinds")
        stats.failures_by_kind = {k: count(k, kinds) for k in kinds}
        if sum(stats.failures_by_kind.values()) != stats.failures:
            raise InvalidStatsState("failure kinds do not add up")
        for key in ("avg_latency_ms", "last_latency_ms"):
            value = state.get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0):
                raise InvalidStatsState(f"bad latency {key}")
        stats.avg_latency_ms = None if state.get("avg_latency_ms") is None else float(state["avg_latency_ms"])
        stats.last_latency_ms = None if state.get("last_latency_ms") is None else int(state["last_latency_ms"])
        recent = state.get("recent")
        if not isinstance(recent, list) or len(recent) > RECENT_WINDOW or len(recent) > stats.attempts:
            raise InvalidStatsState("bad recent attempts")
        for item in recent:
            if not isinstance(item, dict) or not isinstance(item.get("at"), str) or not isinstance(item.get("ok"), bool):
                raise InvalidStatsState("bad recent attempt")
            kind = item.get("kind")
            if (item["ok"] and kind is not None) or (not item["ok"] and kind not in KNOWN_KINDS):
                raise InvalidStatsState("bad recent attempt kind")
            latency, tokens = item.get("latency_ms"), item.get("tokens")
            if not isinstance(latency, int) or isinstance(latency, bool) or latency < 0:
                raise InvalidStatsState("bad recent latency")
            if tokens is not None and (not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0):
                raise InvalidStatsState("bad recent tokens")
            stats.recent.append(Attempt(at=item["at"], ok=item["ok"], latency_ms=latency, kind=kind, tokens=tokens))
        days = state.get("days")
        if not isinstance(days, dict):
            raise InvalidStatsState("bad days")
        for key, value in days.items():
            try:
                date.fromisoformat(key)
            except (TypeError, ValueError):
                raise InvalidStatsState("bad day") from None
            if not isinstance(value, dict):
                raise InvalidStatsState("bad day counts")
            day = DayCounts(requests=count("requests", value), failures=count("failures", value),
                            rate_limited=count("rate_limited", value), tokens=count("tokens", value))
            if day.failures > day.requests or day.rate_limited > day.failures:
                raise InvalidStatsState("day counts do not add up")
            stats.days[key] = day
        return stats


class ProviderStatsStore(ABC):
    """Where each provider's statistics state is kept between runs."""

    @abstractmethod
    def load_all(self) -> dict[str, object]:
        """provider name -> its stored state (as parsed JSON; validated by the caller). Unreadable rows are skipped."""

    @abstractmethod
    def save(self, provider: str, state: dict) -> None:
        """Replace the provider's stored state."""


class SQLiteProviderStatsStore(ProviderStatsStore):
    """One row per provider: (provider, version, state JSON, updated_at). Thread-safe; the connection is shared by threads."""

    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS provider_stats ("
                " provider TEXT PRIMARY KEY, version INTEGER NOT NULL, state TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
            self._conn.commit()

    def load_all(self) -> dict[str, object]:
        with self._lock:
            rows = self._conn.execute("SELECT provider, state FROM provider_stats").fetchall()
        loaded: dict[str, object] = {}
        for provider, text in rows:
            try:
                loaded[provider] = json.loads(text)
            except (TypeError, ValueError):
                _log.warning("provider statistics for %s are unreadable; starting them again", provider)
        return loaded

    def save(self, provider: str, state: dict) -> None:
        text = json.dumps(state, separators=(",", ":"), sort_keys=True)
        with self._lock:
            self._conn.execute(
                "INSERT INTO provider_stats (provider, version, state, updated_at) VALUES (?, ?, ?, datetime('now')) "
                "ON CONFLICT(provider) DO UPDATE SET version = excluded.version, state = excluded.state, "
                "updated_at = excluded.updated_at",
                (provider, int(state.get("version", 0)), text),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def load_stats(store: ProviderStatsStore | None, names: list[str]) -> dict[str, ProviderStats]:
    """Each provider's statistics from `store` - fresh for a provider with no row or a row that cannot be trusted."""
    result = {name: ProviderStats() for name in names}
    if store is None:
        return result
    try:
        stored = store.load_all()
    except Exception as exc:  # noqa: BLE001 - statistics must never stop Jarvis from starting
        _log.warning("provider statistics could not be loaded (%s); starting from empty", type(exc).__name__)
        return result
    for name in names:
        if name in stored:
            try:
                result[name] = ProviderStats.from_state(stored[name])
            except InvalidStatsState as exc:
                _log.warning("stored statistics for %s are not trusted (%s); starting them again", name, exc)
    return result
