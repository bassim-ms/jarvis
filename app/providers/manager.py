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

import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from app.providers.base import AIProvider, ProviderError
from app.providers.requirements import RequestRequirements
from app.providers.router import ProviderRouter, RoutingDecision

_STATUS_RE = re.compile(r"returned (\d{3}):")


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
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not entries:
            raise ValueError("AIProviderManager needs at least one provider.")
        names = [entry.name for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate provider names: {names}")

        self.entries = list(entries)
        self.retry_count = retry_count
        self.retry_delay_seconds = retry_delay_seconds
        self.cooldown_seconds = cooldown_seconds
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
                errors.append(f"{entry.name}: {exc}")
                continue

            health.consecutive_failures = 0
            health.healthy = True
            print(f"[ai-provider-manager] answered by: {entry.name}")
            return reply

        raise AllProvidersFailedError(
            "All AI providers failed: " + "; ".join(errors)
        )

    def _call_with_retries(self, entry: AIProviderEntry, message: str) -> str:
        health = entry.health
        attempts = 0

        while True:
            attempts += 1
            try:
                return entry.provider.generate_reply(message)
            except ProviderError as exc:
                health.consecutive_failures += 1
                category = classify_error(exc)

                if category is ErrorCategory.TRANSIENT and attempts <= self.retry_count:
                    self._sleep(self.retry_delay_seconds)
                    continue

                if category is ErrorCategory.RATE_LIMIT:
                    health.cooldown_until = self._clock() + self.cooldown_seconds

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
                raise UnexpectedProviderError(
                    f"unexpected {type(exc).__name__} (not a ProviderError)"
                ) from None


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
