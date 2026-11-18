"""
AIProviderManager: ordered fallback with health tracking across AIProvider
instances - the provider-agnostic front door Jarvis Core (or any future
caller) uses instead of talking to a single AIProvider directly.

Implements AIProvider itself (generate_reply), so anything that already
takes a single AIProvider can take an AIProviderManager as a drop-in
replacement, exactly the same relationship SearchProviderManager
(app/search/manager.py) has with SearchProvider. This module only
orchestrates: it knows nothing about any specific vendor's HTTP/JSON shape,
and every provider it manages still raises its own ProviderError subclass
(GeminiError, GroqError, ...) exactly as before - see app/providers/factory.py
for how a manager is normally built from configuration, and Core is
untouched here (M12 establishes the manager; wiring Core to it is a later
milestone).

Error handling is deliberately based on the message ProviderError subclasses
already produce (e.g. "Gemini returned 503: ...", "JARVIS_GROQ_API_KEY is
not set.") rather than adding a status_code field to ProviderError - keeps
every existing adapter untouched. See classify_error() below for exactly
which substrings map to which category.

Since M13, generate_reply() also accepts an optional `requirements`
keyword argument (RequestRequirements, see app/providers/requirements.py).
When it's omitted - every call site before M13, and still the overwhelming
majority today - behavior is byte-for-byte unchanged: `order` is simply
`self.entries` in its configured order, exactly as before M13 existed.
When it's given, ProviderRouter (app/providers/router.py) computes a
reordering of the SAME entries (never fewer, never more - see its module
docstring) and that reordering is used as `order` instead. Every retry/
cooldown/classification/fallback/health-tracking line below this point is
completely unmodified by M13 - routing only ever changes which entry is
tried in which position, never what happens once an entry is tried.

Since M14, JarvisCore's DIRECT path and GeminiSearchAnswerGenerator's
SEARCH path (app/core.py, app/search/answer.py) build a RequestRequirements
per request and call this manager through call_with_requirements() below,
rather than calling generate_reply() directly - see that function's own
docstring for why.
"""

import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TypeVar

from app.providers import usage
from app.providers.base import AIProvider, ProviderError
from app.providers.requirements import RequestRequirements
from app.providers.router import ProviderRouter, RoutingDecision
from app.providers.stats import ProviderStats, ProviderStatsStore, load_stats

_log = logging.getLogger("jarvis.providers")

_STATUS_RE = re.compile(r"returned (\d{3}):")
# How long the PROVIDER says to wait, as it appears in a rate-limit error body: Google's RetryInfo (`"retryDelay": "39507s"`)
# or prose ("Please retry in 10h58m27.09s", "Please try again in 7m12s", "Please try again in 450ms",
# "try again in 2 minutes").
_RETRY_DELAY_FIELD_RE = re.compile(r'retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"')
# One "<number><unit>" piece. The units, longest spelling first so "ms" is never read as "m" (minutes) followed by "s", and
# the lookahead stops a unit that runs into more letters ("5 mo", "3 days") from counting as a shorter one.
_UNIT_PATTERN = (
    r"milliseconds?|millis|msecs?|ms"
    r"|seconds?|secs?|s"
    r"|minutes?|mins?|m"
    r"|hours?|hrs?|h"
)
_DURATION_PART = rf"(\d+(?:\.\d+)?)\s*({_UNIT_PATTERN})(?![a-z])"
_DURATION_PART_RE = re.compile(_DURATION_PART, re.IGNORECASE)
_RETRY_PROSE_RE = re.compile(
    rf"(?:retry|try again) in ((?:\d+(?:\.\d+)?\s*(?:{_UNIT_PATTERN})(?![a-z])[\s,]*(?:and\s+)?)+)", re.IGNORECASE
)
_SECONDS_PER_UNIT = {
    **dict.fromkeys(("milliseconds", "millisecond", "millis", "msecs", "msec", "ms"), 0.001),
    **dict.fromkeys(("seconds", "second", "secs", "sec", "s"), 1.0),
    **dict.fromkeys(("minutes", "minute", "mins", "min", "m"), 60.0),
    **dict.fromkeys(("hours", "hour", "hrs", "hr", "h"), 3600.0),
}

T = TypeVar("T")


class ErrorCategory(str, Enum):
    """How AIProviderManager reacts to a given ProviderError."""

    TRANSIENT = "transient"  # e.g. 5xx - retry this provider briefly, then fall back
    RATE_LIMIT = "rate_limit"  # e.g. 429 - cool this provider down, then fall back
    NETWORK = "network"  # timeout/connection failure - fall back, no retry
    AUTH = "auth"  # missing/invalid credentials - fall back, no retry
    OTHER = "other"  # anything else (e.g. unparseable response) - fall back, no retry


def classify_error(exc: BaseException) -> ErrorCategory:
    """Classify a ProviderError by the message its adapter produced.

    Every adapter in this project builds its error message the same way
    (see app/providers/gemini.py, app/providers/_openai_compatible.py,
    app/providers/cloudflare.py): "{name} returned {status}: {body}",
    "Failed to reach {name}: {exc}", or "{ENV_VAR} is not set." - so
    matching on those substrings is reliable without touching any adapter.
    """
    message = str(exc)

    match = _STATUS_RE.search(message)
    if match:
        status = int(match.group(1))
        if status == 429:
            return ErrorCategory.RATE_LIMIT
        if status in (401, 403):
            return ErrorCategory.AUTH
        if status >= 500:
            return ErrorCategory.TRANSIENT
        return ErrorCategory.OTHER

    lowered = message.lower()
    if "not set" in lowered:
        return ErrorCategory.AUTH
    if "failed to reach" in lowered or "timed out" in lowered or "timeout" in lowered:
        return ErrorCategory.NETWORK
    return ErrorCategory.OTHER


def stated_retry_seconds(message: str) -> float | None:
    """The wait a rate-limit error itself asks for, in seconds - or None when it names none. Reads only the two shapes
    documented above, in milliseconds, seconds, minutes or hours (an integer or a decimal, alone or combined as in
    "1h2m3.5s"). Anything else - no number, an unknown unit, zero, or a value too large to be a number - is None, and
    the caller uses its own default cooldown. Never negative: the number pattern has no sign."""
    match = _RETRY_DELAY_FIELD_RE.search(message)
    if match:
        seconds = float(match.group(1))
    else:
        match = _RETRY_PROSE_RE.search(message)
        if not match:
            return None
        seconds = sum(
            float(amount) * _SECONDS_PER_UNIT[unit.lower()] for amount, unit in _DURATION_PART_RE.findall(match.group(1))
        )
    return seconds if 0 < seconds < float("inf") else None


class KeyCheck(str, Enum):
    """What the provider's most recent REAL answer said about its key. Learned only from requests Jarvis was making anyway -
    nothing ever sends a test request to find out - so until a provider has been used, it is NOT_CHECKED."""

    NOT_CHECKED = "not_checked"  # no request has reached this provider since Jarvis started
    AUTHENTICATED = "authenticated"  # its latest answer was a success: it accepted the key
    REJECTED = "rejected"  # its latest answer was 401/403 (or the key is not set): the key was refused
    UNAVAILABLE = "unavailable"  # its latest attempt failed for a reason that says nothing about the key (network, timeout,
    #                              5xx, rate limit, an unusable reply) - the key is not verified by it


#: The key status Jarvis reports for one provider (GET /providers/status, `installer/jarvis_setup.py check`). A non-empty key
#: is never reported as working on its own: only a real successful answer makes it "authenticated".
KEY_STATUS_TEXT: dict[str, str] = {
    "no_key": "no key configured",
    "present_not_verified": "key present, not verified",
    "authenticated": "authenticated (the provider accepted the key)",
    "rejected": "authentication rejected (the provider refused the key)",
    "unavailable": "unavailable (the last attempt failed before the key could be checked)",
}


def key_status(key_present: bool, key_check: KeyCheck) -> str:
    """One provider's key status, a KEY_STATUS_TEXT key. Presence alone is only ever "present_not_verified"."""
    if not key_present:
        return "no_key"
    return {
        KeyCheck.NOT_CHECKED: "present_not_verified",
        KeyCheck.AUTHENTICATED: "authenticated",
        KeyCheck.REJECTED: "rejected",
        KeyCheck.UNAVAILABLE: "unavailable",
    }[key_check]


@dataclass
class ProviderHealth:
    """Runtime, non-secret health state for one managed provider.

    Never holds an API key, request URL, prompt, or response - only counters
    and timestamps - so it's always safe to log or expose (e.g. a future
    `/providers/health` diagnostic endpoint).
    """

    name: str
    enabled: bool = True
    healthy: bool = True
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    #: See KeyCheck: updated by every real attempt, never by a test request.
    key_check: KeyCheck = KeyCheck.NOT_CHECKED
    #: Latency, outcomes, rate limits, tokens and daily counts from real attempts (app/providers/stats.py) - persisted when
    #: the manager has a stats store. Updated only under the manager's statistics lock.
    stats: ProviderStats = field(default_factory=ProviderStats)

    def in_cooldown(self, now: float) -> bool:
        return self.cooldown_until > now


@dataclass
class AIProviderEntry:
    """One provider registered with the manager, under a stable name used
    for ordering, health tracking, and error messages. Kept as an explicit
    (name, provider) pair - rather than deriving a name from the provider's
    class - so tests can register several fakes without name collisions and
    a real registry (app/providers/factory.py) can use its own stable
    provider names (e.g. "gemini")."""

    name: str
    provider: AIProvider
    health: ProviderHealth = field(init=False)

    def __post_init__(self) -> None:
        self.health = ProviderHealth(name=self.name)


class AllProvidersFailedError(ProviderError):
    """Raised when every enabled provider failed or was skipped (disabled/
    cooling down). The message lists each provider's own error - never a
    prompt, key, or URL - see each adapter's own error-building discipline."""


class UnexpectedProviderError(ProviderError):
    """Raised in place of a non-ProviderError exception a provider adapter
    was never supposed to raise (see AIProvider/ProviderError in
    app/providers/base.py). The original exception's message is never
    reused here - a bug in one adapter could be holding anything, including
    a key/URL/prompt - only its type name (a class name, never data) is
    recorded, so this is always safe to surface through
    AllProvidersFailedError."""


class AIProviderManager(AIProvider):
    """Tries AIProviderEntry instances in order, with bounded retries and
    automatic fallback.

    A provider is only supposed to fail by raising ProviderError - the same
    contract every AIProvider already has. How a ProviderError is handled
    depends on classify_error():
      - TRANSIENT: retried against the SAME provider up to `retry_count`
        times (sleeping `retry_delay_seconds` between attempts), then falls
        back to the next provider.
      - RATE_LIMIT: no retry; the provider is put in cooldown for
        `cooldown_seconds` (skipped by later calls until it elapses), then
        falls back.
      - NETWORK / AUTH / OTHER: no retry; falls back immediately.
    If a provider raises anything else (a bug, not a documented
    ProviderError), that single attempt is recorded as a failure via
    UnexpectedProviderError and treated exactly like any other
    non-retryable failure - it can never abort the fallback chain for the
    remaining providers, and its original exception message is never
    exposed (see UnexpectedProviderError).
    A successful call returns immediately with no further provider tried.
    If every enabled, non-cooling-down provider fails, one
    AllProvidersFailedError is raised - never an unbounded retry/fallback
    loop, since both the provider list and retry_count are finite.
    """

    def __init__(
        self,
        entries: list[AIProviderEntry],
        *,
        retry_count: int = 2,
        retry_delay_seconds: float = 0.5,
        cooldown_seconds: float = 60.0,
        max_cooldown_seconds: float = 86400.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        stats_store: ProviderStatsStore | None = None,
        timer: Callable[[], float] = time.perf_counter,
        wall_clock: Callable[[], datetime] = lambda: datetime.now().astimezone(),
        daily_request_limits: dict[str, int] | None = None,
    ):
        if not entries:
            raise ValueError("AIProviderManager needs at least one provider.")
        names = [entry.name for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate provider names: {names}")
        # Provider statistics (app/providers/stats.py): loaded from the store (if any) so they survive a restart; every real
        # attempt is recorded and saved under one lock, so concurrent requests on different threads never lose a count.
        self.stats_store = stats_store
        self._timer = timer
        self.wall_clock = wall_clock
        #: Optional, user-configured daily request limits per provider (quota awareness only - nothing is blocked by them).
        self.daily_request_limits = dict(daily_request_limits or {})
        self._stats_lock = threading.Lock()
        self._stats_save_failed = False
        loaded = load_stats(stats_store, names)
        for entry in entries:
            entry.health.stats = loaded[entry.name]

        self.entries = list(entries)
        self.retry_count = retry_count
        self.retry_delay_seconds = retry_delay_seconds
        self.cooldown_seconds = cooldown_seconds
        #: The ceiling on a cooldown the PROVIDER asks for (see `_cooldown_for`): a garbled or hostile "retry in 9999h"
        #: can never park a provider for longer than this.
        self.max_cooldown_seconds = max_cooldown_seconds
        self._clock = clock
        self._sleep = sleep
        # M13: set on every generate_reply() call made WITH `requirements`;
        # left None otherwise. Purely observational (debugging/tests) -
        # never read by any logic in this class. Never holds a key/token/
        # header/URL or the request's own message text (see RoutingDecision).
        self.last_routing_decision: RoutingDecision | None = None

    def get_health(self) -> list[ProviderHealth]:
        """Current health snapshot for every registered provider, in
        priority order. Safe to log/expose - see ProviderHealth."""
        return [entry.health for entry in self.entries]

    def stats_snapshot(self, name: str) -> dict:
        """One provider's statistics as of now (see ProviderStats.snapshot) - read under the statistics lock."""
        entry = self._entry(name)
        with self._stats_lock:
            return entry.health.stats.snapshot(self.wall_clock(), self.daily_request_limits.get(name))

    def _record(self, entry: AIProviderEntry, ok: bool, started: float, kind: str | None, tokens: int | None) -> None:
        """Record one REAL attempt and persist the provider's state. Never raises: a statistics problem must not change the
        outcome of an AI request (a failed save is logged once, by type)."""
        try:
            latency_ms = (self._timer() - started) * 1000.0
            with self._stats_lock:
                entry.health.stats.record(ok, latency_ms, self.wall_clock(), kind=kind, tokens=tokens)
                if self.stats_store is not None:
                    self.stats_store.save(entry.name, entry.health.stats.to_state())
        except Exception as exc:  # noqa: BLE001
            if not self._stats_save_failed:
                self._stats_save_failed = True
                _log.warning("provider statistics could not be recorded (%s); requests continue", type(exc).__name__)

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Enable/disable a registered provider by name at runtime."""
        self._entry(name).health.enabled = enabled

    def _entry(self, name: str) -> AIProviderEntry:
        for entry in self.entries:
            if entry.name == name:
                return entry
        raise KeyError(f"No provider registered under '{name}'.")

    def generate_reply(
        self, message: str, *, requirements: RequestRequirements | None = None
    ) -> str:
        return self._first_acceptable(message, requirements, None)

    def generate_validated(
        self,
        message: str,
        validate: Callable[[str], T],
        *,
        requirements: RequestRequirements | None = None,
    ) -> T:
        """The first reply, in the fixed provider order, that `validate` accepts - returned as `validate`'s own result.

        For a request whose reply must have a shape (strict JSON, a schema): `validate` is the caller's existing parser /
        validator, and raises on a reply it rejects. A rejected reply is NOT a result: that provider is passed over for
        this one request (no retry of the same provider, no cooldown, its health untouched) and the next provider in the
        same order is asked. Nothing is repaired or guessed here.

        If every provider failed and at least one of them DID answer but was rejected, the LAST rejection is re-raised
        unchanged - so the caller's own error handling (its own exception type, its own fail-safe) runs exactly as it did
        when it validated a single reply itself. If no provider answered at all, AllProvidersFailedError as usual."""
        return self._first_acceptable(message, requirements, validate)

    def _first_acceptable(self, message: str, requirements: RequestRequirements | None, validate):
        if requirements is None:
            # No stale decision from a previous routed call should ever
            # remain visible once a caller makes an unrouted request.
            self.last_routing_decision = None
            order = self.entries
        else:
            decision = ProviderRouter().route(
                requirements, self.entries, now=self._clock()
            )
            self.last_routing_decision = decision
            order = [self._entry(name) for name in decision.ordered_names]

        errors: list[str] = []
        last_rejection: Exception | None = None

        for entry in order:
            health = entry.health

            if not health.enabled:
                errors.append(f"{entry.name}: disabled")
                continue

            now = self._clock()
            if health.in_cooldown(now):
                remaining = health.cooldown_until - now
                errors.append(f"{entry.name}: cooling down ({remaining:.0f}s left)")
                continue

            try:
                reply = self._call_with_retries(entry, message)
            except ProviderError as exc:
                # Manual-testing visibility only (M12 integration test prep):
                # which provider was tried and why it fell back, on the
                # server's own console - never a key/URL/prompt, since exc
                # is the same secret-safe message already used above.
                print(f"[ai-provider-manager] {entry.name} failed, falling back: {exc}")
                _log.warning("provider %s failed (%s); trying the next in order", entry.name, classify_error(exc).value)
                errors.append(f"{entry.name}: {exc}")
                continue

            health.consecutive_failures = 0
            health.healthy = True

            if validate is None:
                print(f"[ai-provider-manager] answered by: {entry.name}")
                return reply
            try:
                accepted = validate(reply)
            except Exception as rejection:  # noqa: BLE001 - the caller's validator decides; any refusal means "next provider"
                last_rejection = rejection
                # The exception's TYPE only: a validator's message can quote the reply.
                print(f"[ai-provider-manager] {entry.name} answered but its reply failed validation "
                      f"({type(rejection).__name__}), falling back")
                _log.warning("provider %s reply rejected by the caller's validation (%s); trying the next in order",
                             entry.name, type(rejection).__name__)
                errors.append(f"{entry.name}: reply failed validation ({type(rejection).__name__})")
                continue
            print(f"[ai-provider-manager] answered by: {entry.name}")
            return accepted

        if last_rejection is not None:
            raise last_rejection
        raise AllProvidersFailedError(
            "All AI providers failed: " + "; ".join(errors)
        )

    def _cooldown_for(self, exc: ProviderError) -> float:
        """How long to skip a rate-limited provider: what the provider itself asked for when it says (a daily-quota 429
        names hours, and re-asking every minute only wastes a call each time), else the configured default - never more
        than `max_cooldown_seconds`, and never negative (a misconfigured negative default or ceiling means "no rest")."""
        stated = stated_retry_seconds(str(exc))
        wait = self.cooldown_seconds if stated is None else stated
        return max(0.0, min(wait, self.max_cooldown_seconds))

    def _call_with_retries(self, entry: AIProviderEntry, message: str) -> str:
        health = entry.health
        attempts = 0

        while True:
            attempts += 1
            usage.clear()
            started = self._timer()
            try:
                reply = entry.provider.generate_reply(message)
            except ProviderError as exc:
                health.consecutive_failures += 1
                category = classify_error(exc)
                self._record(entry, False, started, category.value, usage.take())
                health.key_check = KeyCheck.REJECTED if category is ErrorCategory.AUTH else KeyCheck.UNAVAILABLE

                if category is ErrorCategory.TRANSIENT and attempts <= self.retry_count:
                    self._sleep(self.retry_delay_seconds)
                    continue

                if category is ErrorCategory.RATE_LIMIT:
                    health.cooldown_until = self._clock() + self._cooldown_for(exc)

                health.healthy = False
                raise
            except Exception as exc:
                # Adapters are only ever supposed to raise ProviderError -
                # this is a safety net for a bug in one adapter, so it
                # can't abort the fallback chain for the remaining
                # providers. Never retried (unclear whether it's
                # transient), and str(exc) is deliberately never reused -
                # buggy code could be holding anything, including a
                # key/URL/prompt - only the exception's type name (a class
                # name, never data) is recorded. BaseException subclasses
                # (KeyboardInterrupt, SystemExit) are not caught here.
                health.consecutive_failures += 1
                health.healthy = False
                health.key_check = KeyCheck.UNAVAILABLE
                self._record(entry, False, started, "unexpected", usage.take())
                raise UnexpectedProviderError(
                    f"unexpected {type(exc).__name__} (not a ProviderError)"
                ) from None
            # A real answer: whatever the caller later thinks of its content, the provider accepted the key.
            health.key_check = KeyCheck.AUTHENTICATED
            self._record(entry, True, started, None, usage.take())
            return reply


def call_with_requirements(
    provider: AIProvider, message: str, requirements: RequestRequirements
) -> str:
    """Call `provider.generate_reply`, passing `requirements` only if
    `provider` actually understands it - today, only AIProviderManager
    does (see its own generate_reply above). Every other AIProvider
    implementation (GeminiProvider, GroqProvider, a test fake, a directly-
    injected single provider, ...) is called exactly as it always has
    been, with no new keyword argument at all - so any caller that
    injects a plain AIProvider (JarvisCore's own constructor,
    GeminiSearchAnswerGenerator's own default, every existing AIProvider
    fake in this test suite) remains completely unaffected by M14's
    routing integration.

    This is the one seam JarvisCore's DIRECT path and
    GeminiSearchAnswerGenerator's SEARCH path use instead of each
    duplicating an isinstance check."""
    if isinstance(provider, AIProviderManager):
        return provider.generate_reply(message, requirements=requirements)
    return provider.generate_reply(message)


def call_structured(
    provider: AIProvider,
    message: str,
    validate: Callable[[str], T],
    *,
    requirements: RequestRequirements | None = None,
) -> T:
    """Ask for a reply that must pass `validate` (the caller's own strict parser), and return `validate`'s result.

    THE one seam for every strict-JSON workflow (capability decision, tool-call construction, context reasoning and
    planning, goal interpretation, Job planning, result evaluation, self-development planning): with the application's
    AIProviderManager, a reply that fails validation sends the request to the NEXT provider in the fixed order (see
    AIProviderManager.generate_validated) instead of failing the whole request on the first provider's bad JSON. With any
    other AIProvider (a test fake, a single injected provider) it is exactly what every caller did before: one call, then
    `validate` on its reply, the validator's own exception propagating unchanged."""
    if isinstance(provider, AIProviderManager):
        return provider.generate_validated(message, validate, requirements=requirements)
    return validate(provider.generate_reply(message))
