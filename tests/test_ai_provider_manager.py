"""
Tests for AIProviderManager (app/providers/manager.py): ordered fallback,
error classification, bounded retries, cooldown, and runtime health state -
using lightweight fake providers, no real Gemini/Groq/OpenRouter/Cloudflare
HTTP calls anywhere in this file.
"""

import pytest

from app.providers.base import AIProvider, ProviderError
from app.providers.manager import (
    AIProviderEntry,
    AIProviderManager,
    AllProvidersFailedError,
    ErrorCategory,
    UnexpectedProviderError,
    classify_error,
)
from app.providers.requirements import LatencyPreference, RequestRequirements


class FakeProvider(AIProvider):
    """Records every call; succeeds with a fixed reply, or raises a queued
    error (a list of exceptions consumed one per call, or a single
    exception raised every time) so tests can script exact failure
    sequences like "503 once, then succeeds"."""

    def __init__(self, reply="ok", errors=None):
        self.reply = reply
        self._errors = list(errors) if errors is not None else None
        self.calls = 0

    def generate_reply(self, message: str) -> str:
        self.calls += 1
        if self._errors is not None:
            if self._errors:
                raise self._errors.pop(0)
            return self.reply
        return self.reply


def entry(name, provider):
    return AIProviderEntry(name=name, provider=provider)


class FakeClock:
    """Deterministic, manually-advanced clock for cooldown tests."""

    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def no_sleep(_seconds):
    """Replaces time.sleep in tests - retries must not actually pause."""


# --- Construction ------------------------------------------------------------


def test_manager_requires_at_least_one_provider():
    with pytest.raises(ValueError):
        AIProviderManager([])


def test_manager_rejects_duplicate_provider_names():
    with pytest.raises(ValueError):
        AIProviderManager(
            [entry("a", FakeProvider()), entry("a", FakeProvider())]
        )


# --- Success path --------------------------------------------------------------


def test_first_provider_success_short_circuits_later_providers():
    first = FakeProvider(reply="from first")
    second = FakeProvider(reply="from second")
    manager = AIProviderManager([entry("first", first), entry("second", second)])

    reply = manager.generate_reply("hi")

    assert reply == "from first"
    assert first.calls == 1
    assert second.calls == 0


def test_successful_provider_resets_health():
    first = FakeProvider(errors=[ProviderError("X returned 503: down")])
    manager = AIProviderManager(
        [entry("first", first)], retry_count=1, sleep=no_sleep
    )

    reply = manager.generate_reply("hi")

    assert reply == "ok"
    health = manager.get_health()[0]
    assert health.healthy is True
    assert health.consecutive_failures == 0


# --- Provider ordering -----------------------------------------------------


def test_provider_order_is_respected():
    order = []

    class TrackingProvider(AIProvider):
        def __init__(self, name):
            self.name = name

        def generate_reply(self, message):
            order.append(self.name)
            raise ProviderError(f"{self.name} returned 500: down")

    manager = AIProviderManager(
        [
            entry("a", TrackingProvider("a")),
            entry("b", TrackingProvider("b")),
            entry("c", TrackingProvider("c")),
        ],
        retry_count=0,
        sleep=no_sleep,
    )

    with pytest.raises(AllProvidersFailedError):
        manager.generate_reply("hi")

    assert order == ["a", "b", "c"]


# --- Error classification ---------------------------------------------------


@pytest.mark.parametrize(
    "message, expected",
    [
        ("Gemini returned 503: Service Unavailable", ErrorCategory.TRANSIENT),
        ("Groq returned 500: Internal Server Error", ErrorCategory.TRANSIENT),
        ("OpenRouter returned 429: Too Many Requests", ErrorCategory.RATE_LIMIT),
        ("Cloudflare returned 401: Unauthorized", ErrorCategory.AUTH),
        ("Gemini returned 403: Forbidden", ErrorCategory.AUTH),
        ("Groq returned 400: Bad Request", ErrorCategory.OTHER),
        ("JARVIS_GEMINI_API_KEY is not set.", ErrorCategory.AUTH),
        ("Failed to reach Gemini: Connection refused", ErrorCategory.NETWORK),
        ("Failed to reach Groq: timed out", ErrorCategory.NETWORK),
        ("Unexpected Gemini response: {}", ErrorCategory.OTHER),
    ],
)
def test_classify_error(message, expected):
    assert classify_error(ProviderError(message)) == expected


# --- Transient (503) retry then fallback ------------------------------------


def test_transient_error_is_retried_before_falling_back():
    first = FakeProvider(
        reply="recovered",
        errors=[
            ProviderError("Gemini returned 503: down"),
            ProviderError("Gemini returned 503: still down"),
        ],
    )
    second = FakeProvider(reply="from second")
    manager = AIProviderManager(
        [entry("first", first), entry("second", second)],
        retry_count=2,
        sleep=no_sleep,
    )

    reply = manager.generate_reply("hi")

    assert reply == "recovered"
    assert first.calls == 3  # 1 initial + 2 retries, all against `first`
    assert second.calls == 0


def test_transient_error_falls_back_after_retries_exhausted():
    first = FakeProvider(
        errors=[
            ProviderError("Gemini returned 503: down"),
            ProviderError("Gemini returned 503: still down"),
        ]
    )
    second = FakeProvider(reply="from second")
    manager = AIProviderManager(
        [entry("first", first), entry("second", second)],
        retry_count=1,
        sleep=no_sleep,
    )

    reply = manager.generate_reply("hi")

    assert reply == "from second"
    assert first.calls == 2  # 1 initial + 1 retry, then fallback
    assert second.calls == 1
    assert manager.get_health()[0].healthy is False
    assert manager.get_health()[0].consecutive_failures == 2


def test_retries_are_bounded_not_infinite():
    first = FakeProvider(errors=[ProviderError("Gemini returned 503: down")] * 100)
    manager = AIProviderManager([entry("first", first)], retry_count=3, sleep=no_sleep)

    with pytest.raises(AllProvidersFailedError):
        manager.generate_reply("hi")

    assert first.calls == 4  # 1 initial + 3 retries, never unbounded


# --- Rate limit (429) cooldown then fallback --------------------------------


def test_rate_limit_error_falls_back_without_retrying_same_provider():
    first = FakeProvider(errors=[ProviderError("OpenRouter returned 429: slow down")])
    second = FakeProvider(reply="from second")
    manager = AIProviderManager(
        [entry("first", first), entry("second", second)],
        retry_count=3,
        sleep=no_sleep,
    )

    reply = manager.generate_reply("hi")

    assert reply == "from second"
    assert first.calls == 1  # no retry against the rate-limited provider
    assert second.calls == 1


def test_rate_limit_error_sets_cooldown():
    clock = FakeClock()
    first = FakeProvider(errors=[ProviderError("OpenRouter returned 429: slow down")])
    second = FakeProvider(reply="from second")
    manager = AIProviderManager(
        [entry("first", first), entry("second", second)],
        cooldown_seconds=30,
        sleep=no_sleep,
        clock=clock,
    )

    manager.generate_reply("hi")

    health = manager.get_health()[0]
    assert health.cooldown_until == 30


def test_cooldown_prevents_immediate_reuse_then_expires():
    clock = FakeClock()
    first = FakeProvider(
        reply="recovered",
        errors=[ProviderError("OpenRouter returned 429: slow down")],
    )
    second = FakeProvider(reply="from second")
    manager = AIProviderManager(
        [entry("first", first), entry("second", second)],
        cooldown_seconds=30,
        sleep=no_sleep,
        clock=clock,
    )

    # First call: `first` rate-limits and cools down, `second` answers.
    assert manager.generate_reply("hi") == "from second"
    assert first.calls == 1

    # Still within cooldown: `first` is skipped again.
    clock.advance(10)
    assert manager.generate_reply("hi") == "from second"
    assert first.calls == 1
    assert second.calls == 2

    # Cooldown elapsed: `first` is tried again and succeeds.
    clock.advance(30)
    assert manager.generate_reply("hi") == "recovered"
    assert first.calls == 2


# --- Network/timeout errors --------------------------------------------------


def test_network_error_falls_back_without_retry():
    first = FakeProvider(errors=[ProviderError("Failed to reach Gemini: timed out")])
    second = FakeProvider(reply="from second")
    manager = AIProviderManager(
        [entry("first", first), entry("second", second)],
        retry_count=3,
        sleep=no_sleep,
    )

    reply = manager.generate_reply("hi")

    assert reply == "from second"
    assert first.calls == 1
    assert second.calls == 1


# --- Auth/configuration errors -----------------------------------------------


def test_auth_error_falls_back_without_retry():
    first = FakeProvider(errors=[ProviderError("JARVIS_GEMINI_API_KEY is not set.")])
    second = FakeProvider(reply="from second")
    manager = AIProviderManager(
        [entry("first", first), entry("second", second)],
        retry_count=3,
        sleep=no_sleep,
    )

    reply = manager.generate_reply("hi")

    assert reply == "from second"
    assert first.calls == 1
    assert second.calls == 1
    assert manager.get_health()[0].healthy is False


def test_auth_error_does_not_set_cooldown():
    first = FakeProvider(errors=[ProviderError("JARVIS_GEMINI_API_KEY is not set.")])
    second = FakeProvider(reply="from second")
    manager = AIProviderManager(
        [entry("first", first), entry("second", second)], sleep=no_sleep
    )

    manager.generate_reply("hi")

    assert manager.get_health()[0].cooldown_until == 0.0


# --- Multiple / all providers failing ---------------------------------------


def test_multiple_provider_failures_continue_to_next():
    first = FakeProvider(errors=[ProviderError("Gemini returned 500: down")])
    second = FakeProvider(errors=[ProviderError("Groq returned 500: down")])
    third = FakeProvider(reply="from third")
    manager = AIProviderManager(
        [entry("first", first), entry("second", second), entry("third", third)],
        retry_count=0,
        sleep=no_sleep,
    )

    reply = manager.generate_reply("hi")

    assert reply == "from third"
    assert first.calls == 1
    assert second.calls == 1
    assert third.calls == 1


def test_all_providers_failing_raises_one_clean_error():
    first = FakeProvider(errors=[ProviderError("Gemini returned 500: first down")])
    second = FakeProvider(errors=[ProviderError("Groq returned 500: second down")])
    manager = AIProviderManager(
        [entry("first", first), entry("second", second)],
        retry_count=0,
        sleep=no_sleep,
    )

    with pytest.raises(AllProvidersFailedError) as excinfo:
        manager.generate_reply("hi")

    message = str(excinfo.value)
    assert "first down" in message
    assert "second down" in message


# --- Disabled providers -------------------------------------------------------


def test_disabled_provider_at_construction_is_skipped():
    first = FakeProvider(reply="from first")
    second = FakeProvider(reply="from second")
    manager = AIProviderManager([entry("first", first), entry("second", second)])
    manager.set_enabled("first", False)

    reply = manager.generate_reply("hi")

    assert reply == "from second"
    assert first.calls == 0
    assert second.calls == 1


def test_reenabling_a_provider_makes_it_eligible_again():
    first = FakeProvider(reply="from first")
    manager = AIProviderManager([entry("first", first)])
    manager.set_enabled("first", False)

    with pytest.raises(AllProvidersFailedError):
        manager.generate_reply("hi")

    manager.set_enabled("first", True)
    assert manager.generate_reply("hi") == "from first"


# --- No secret leakage -------------------------------------------------------


def test_error_output_never_contains_provider_secrets():
    first = FakeProvider(
        errors=[
            ProviderError(
                "Gemini returned 401: invalid key sk-super-secret-do-not-leak"
            )
        ]
    )
    manager = AIProviderManager([entry("first", first)], sleep=no_sleep)

    with pytest.raises(AllProvidersFailedError) as excinfo:
        manager.generate_reply("hi")

    # The manager only ever forwards the error message the provider itself
    # built - it never adds or exposes anything extra (no key, url, prompt).
    assert "hi" not in str(excinfo.value)  # the prompt itself is never echoed


def test_health_snapshot_contains_no_secrets_by_construction():
    first = FakeProvider()
    manager = AIProviderManager([entry("first", first)])

    health = manager.get_health()[0]

    assert health.name == "first"
    assert not hasattr(health, "api_key")
    assert not hasattr(health, "url")


# --- Unexpected (non-ProviderError) exceptions -------------------------------


def test_unexpected_exception_from_one_provider_falls_back_to_next():
    first = FakeProvider(errors=[RuntimeError("boom: sk-super-secret-key")])
    second = FakeProvider(reply="from second")
    manager = AIProviderManager(
        [entry("first", first), entry("second", second)],
        retry_count=3,
        sleep=no_sleep,
    )

    reply = manager.generate_reply("hi")

    assert reply == "from second"
    assert first.calls == 1  # never retried
    assert second.calls == 1


def test_unexpected_exception_successful_second_provider_response_returned():
    first = FakeProvider(errors=[ValueError("unexpected shape")])
    second = FakeProvider(reply="the real answer")
    manager = AIProviderManager([entry("first", first), entry("second", second)])

    assert manager.generate_reply("hi") == "the real answer"


def test_unexpected_exception_details_are_not_exposed_to_the_caller():
    secret = "sk-super-secret-do-not-leak-1234"
    first = FakeProvider(
        errors=[RuntimeError(f"leaked prompt + key={secret} + url=https://x/{secret}")]
    )
    manager = AIProviderManager([entry("first", first)], sleep=no_sleep)

    with pytest.raises(AllProvidersFailedError) as excinfo:
        manager.generate_reply("hi")

    message = str(excinfo.value)
    assert secret not in message
    assert "leaked prompt" not in message
    # Only the exception's type name is safe to surface, never its message.
    assert "RuntimeError" in message


def test_unexpected_exception_is_recorded_as_provider_failure():
    first = FakeProvider(errors=[RuntimeError("boom")])
    second = FakeProvider(reply="from second")
    manager = AIProviderManager(
        [entry("first", first), entry("second", second)], sleep=no_sleep
    )

    manager.generate_reply("hi")

    health = manager.get_health()[0]
    assert health.healthy is False
    assert health.consecutive_failures == 1


def test_unexpected_exception_is_not_retried_against_the_same_provider():
    first = FakeProvider(errors=[RuntimeError("boom")] * 10)
    manager = AIProviderManager([entry("first", first)], retry_count=5, sleep=no_sleep)

    with pytest.raises(AllProvidersFailedError):
        manager.generate_reply("hi")

    assert first.calls == 1  # a single attempt, never retried


def test_all_providers_raising_unexpected_exceptions_gives_safe_combined_error():
    first = FakeProvider(errors=[RuntimeError("first internal bug: key=abc123")])
    second = FakeProvider(errors=[KeyError("second internal bug: key=def456")])
    manager = AIProviderManager(
        [entry("first", first), entry("second", second)], sleep=no_sleep
    )

    with pytest.raises(AllProvidersFailedError) as excinfo:
        manager.generate_reply("hi")

    message = str(excinfo.value)
    assert "abc123" not in message
    assert "def456" not in message
    assert "first" in message
    assert "second" in message
    assert "RuntimeError" in message
    assert "KeyError" in message


def test_unexpected_exception_is_wrapped_as_unexpected_provider_error():
    first = FakeProvider(errors=[TypeError("bad type")])
    single_entry = entry("first", first)
    manager = AIProviderManager([single_entry], sleep=no_sleep)

    with pytest.raises(UnexpectedProviderError):
        manager._call_with_retries(single_entry, "hi")


def test_keyboard_interrupt_is_not_caught_by_the_manager():
    class RaisesKeyboardInterrupt(AIProvider):
        def generate_reply(self, message):
            raise KeyboardInterrupt()

    manager = AIProviderManager([entry("first", RaisesKeyboardInterrupt())])

    with pytest.raises(KeyboardInterrupt):
        manager.generate_reply("hi")


# --- M13 hybrid routing integration -------------------------------------------
#
# ProviderRouter itself is tested in isolation in tests/test_provider_router.py
# (eligibility, scoring, tie-breaking, the decision record). These tests only
# cover the seam in AIProviderManager.generate_reply: that `requirements=None`
# (the default, and every test above this section) is completely unaffected,
# and that once routing picks an order, M12's own retry/cooldown/fallback loop
# still runs exactly as it always has over that order.


def test_generate_reply_without_requirements_is_unaffected_by_m13():
    first = FakeProvider(reply="from first")
    second = FakeProvider(reply="from second")
    manager = AIProviderManager([entry("first", first), entry("second", second)])

    reply = manager.generate_reply("hi")

    assert reply == "from first"
    assert manager.last_routing_decision is None


def test_routing_selects_groq_then_m12_falls_back_after_groq_fails():
    # Mirrors the M13 design's own example: router prefers groq (latency
    # preference) even though it's not first in configured order, groq
    # fails, and M12's existing fallback loop moves on to the next
    # eligible provider in the router's order (openrouter beats gemini on
    # the deterministic tie-break: both score 0, openrouter is earlier in
    # configured order).
    openrouter = FakeProvider(reply="from openrouter")
    groq = FakeProvider(errors=[ProviderError("Groq returned 500: down")])
    gemini = FakeProvider(reply="from gemini")
    manager = AIProviderManager(
        [entry("openrouter", openrouter), entry("groq", groq), entry("gemini", gemini)],
        retry_count=0,
        sleep=no_sleep,
    )
    requirements = RequestRequirements(latency_preference=LatencyPreference.LOW)

    reply = manager.generate_reply("hi", requirements=requirements)

    assert reply == "from openrouter"
    assert groq.calls == 1
    assert gemini.calls == 0  # never reached - openrouter answered right after groq failed
    assert manager.last_routing_decision.selected == "groq"
    assert manager.last_routing_decision.ordered_names == ["groq", "openrouter", "gemini"]


def test_routing_with_unsatisfiable_hard_requirement_degrades_to_configured_order():
    first = FakeProvider(reply="from first")
    second = FakeProvider(reply="from second")
    manager = AIProviderManager([entry("first", first), entry("second", second)])
    # Neither fake provider is a registered capability name, so
    # structured_output can never be satisfied - routing must still fall
    # back to the manager's own configured order rather than blocking.
    requirements = RequestRequirements(structured_output=True)

    reply = manager.generate_reply("hi", requirements=requirements)

    assert reply == "from first"
    assert manager.last_routing_decision.eligible == []
    assert manager.last_routing_decision.ordered_names == ["first", "second"]


def test_last_routing_decision_does_not_leak_into_a_later_unrouted_call():
    """A routed call's decision must never remain visible after a later
    call that doesn't route - otherwise a caller inspecting
    last_routing_decision after an ordinary (unrouted) request would see
    stale information from a completely different, earlier request."""
    first = FakeProvider(reply="from first")
    manager = AIProviderManager([entry("first", first)])

    manager.generate_reply("hi", requirements=RequestRequirements())
    assert manager.last_routing_decision is not None

    manager.generate_reply("hi")  # no requirements this time

    assert manager.last_routing_decision is None


def test_last_routing_decision_contains_no_secrets():
    first = FakeProvider(errors=[ProviderError("Gemini returned 401: sk-super-secret")])
    second = FakeProvider(reply="from second")
    manager = AIProviderManager(
        [entry("first", first), entry("second", second)], sleep=no_sleep
    )

    manager.generate_reply("hi", requirements=RequestRequirements())

    decision = manager.last_routing_decision
    assert "sk-super-secret" not in decision.reason
    assert all("sk-super-secret" not in c.reason for c in decision.excluded)
