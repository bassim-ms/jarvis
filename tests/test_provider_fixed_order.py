"""
The fixed provider order, end to end: Groq -> OpenRouter -> Gemini -> Cloudflare -> FreeLLMAPI.

Every AI workflow goes through the ONE AIProviderManager and its ONE ProviderRouter, so these tests drive the real manager
with five recording fakes (never the network) through every way a workflow calls it - a plain reply, a routed reply of every
kind, and a validated strict-JSON reply - and through the real JarvisCore, tool-call, capability-decision and search-answer
code. They prove: the order is exactly the configured one on every path; a failure at ANY position moves to the next
provider in that order; retries, timeouts and rate-limit cooldowns are bounded; a reply that fails validation is never used
as a result; and a failed provider never makes a tool run twice.
"""

import json

import pytest

from app.capability import Capability, GeminiCapabilityDecisionProvider
from app.core import JarvisCore
from app.memory.sqlite_store import SQLiteMemoryStore
from app.providers import cloudflare, gemini, _openai_compatible
from app.providers.base import AIProvider, ProviderError
from app.providers.manager import (
    AIProviderEntry,
    AIProviderManager,
    AllProvidersFailedError,
    call_structured,
    call_with_requirements,
    stated_retry_seconds,
)
from app.providers.requirements import LatencyPreference, RequestRequirements, TaskType, classify_request
from app.search.answer import GeminiSearchAnswerGenerator
from app.search.base import SearchResult
from app.search.modes import SearchMode
from app.search.response import SearchResponse
from app.tools.base import PermissionLevel, Tool, ToolParameter, ToolResult
from app.tools.decision import GeminiToolCallProvider
from app.tools.registry import ToolRegistry

FIXED_ORDER = ["groq", "openrouter", "gemini", "cloudflare", "freellmapi"]

#: A personality + profile long enough to cross the old "long context" threshold that used to promote Gemini.
LONG_IDENTITY = "Jarvis's personality:\n" + ("Be warm, precise and brief. " * 300)


class Recorder(AIProvider):
    """One fake provider: appends its name to a shared log each time it is asked, then raises its scripted error or
    returns its scripted reply."""

    def __init__(self, name, log, *, reply=None, error=None, errors=None):
        self.name, self.log, self.reply = name, log, reply
        self.errors = list(errors) if errors is not None else ([error] if error is not None else [])
        self.repeat_error = error if errors is None else None
        self.calls = 0

    def generate_reply(self, message: str) -> str:
        self.calls += 1
        self.log.append(self.name)
        if self.repeat_error is not None:
            raise self.repeat_error
        if self.errors:
            raise self.errors.pop(0)
        return self.reply if self.reply is not None else f"reply from {self.name}"


def down(name):
    return ProviderError(f"Failed to reach {name}: connection refused")


def chain(answering=None, *, reply=None, order=FIXED_ORDER, **manager_options):
    """A manager over five recorders in `order`. Providers before `answering` fail (network error); `answering` and those
    after it answer. `answering=None` means every provider fails."""
    log = []
    recorders = {}
    position = order.index(answering) if answering is not None else len(order)
    for index, name in enumerate(order):
        if index < position:
            recorders[name] = Recorder(name, log, error=down(name))
        else:
            recorders[name] = Recorder(name, log, reply=reply)
    manager_options.setdefault("sleep", lambda seconds: None)
    manager = AIProviderManager([AIProviderEntry(name, recorders[name]) for name in order], **manager_options)
    return manager, log, recorders


# --- 1. The order, on every way a workflow calls the manager -------------------------------------------------------------

REQUEST_KINDS = {
    "unrouted (Job planner, evaluator, goal interpreter, tool-call builder, analyze, brief)": None,
    "chat reply with personality/profile": classify_request("hello", capability=Capability.DIRECT, context_char_count=9000),
    "chat reply, reasoning wording": classify_request("explain why", capability=Capability.DIRECT, context_char_count=9000),
    "search answer": classify_request("latest news", capability=Capability.SEARCH, context_char_count=9000),
    "tool call": classify_request("use tool x", capability=Capability.TOOL, context_char_count=50),
    "strict JSON (context reasoner / planner / self-development planner)": RequestRequirements(
        task_type=TaskType.STRUCTURED_OUTPUT, structured_output=True, reasoning=True
    ),
    "low latency": RequestRequirements(latency_preference=LatencyPreference.LOW),
}


@pytest.mark.parametrize("kind", sorted(REQUEST_KINDS))
def test_when_everything_fails_every_provider_is_tried_once_in_the_fixed_order(kind):
    manager, log, _ = chain(answering=None)

    with pytest.raises(AllProvidersFailedError):
        manager.generate_reply("hi", requirements=REQUEST_KINDS[kind])

    assert log == FIXED_ORDER


@pytest.mark.parametrize("kind", sorted(REQUEST_KINDS))
@pytest.mark.parametrize("answering", FIXED_ORDER)
def test_a_failure_at_any_position_moves_to_the_next_provider_in_order(kind, answering):
    manager, log, _ = chain(answering=answering)

    reply = manager.generate_reply("hi", requirements=REQUEST_KINDS[kind])

    assert reply == f"reply from {answering}"
    assert log == FIXED_ORDER[: FIXED_ORDER.index(answering) + 1]  # everyone before it, once each, then it - nobody after


@pytest.mark.parametrize("kind", sorted(REQUEST_KINDS))
@pytest.mark.parametrize("answering", FIXED_ORDER)
def test_a_validated_request_follows_the_same_order(kind, answering):
    manager, log, _ = chain(answering=answering, reply='{"ok": true}')

    result = manager.generate_validated("hi", json.loads, requirements=REQUEST_KINDS[kind])

    assert result == {"ok": True}
    assert log == FIXED_ORDER[: FIXED_ORDER.index(answering) + 1]


def test_the_two_seams_use_the_same_order():
    manager, log, _ = chain(answering="gemini", reply="[1]")
    requirements = RequestRequirements(structured_output=True)

    assert call_with_requirements(manager, "hi", requirements) == "[1]"
    assert call_structured(manager, "hi", json.loads, requirements=requirements) == [1]
    assert log == ["groq", "openrouter", "gemini"] * 2


def test_a_different_configured_order_is_honoured_just_as_exactly():
    order = ["cloudflare", "gemini", "freellmapi", "groq", "openrouter"]
    manager, log, _ = chain(answering=None, order=order)

    with pytest.raises(AllProvidersFailedError):
        manager.generate_reply("hi", requirements=RequestRequirements(structured_output=True, long_context=True))

    assert log == order


# --- 2. The real workflows ------------------------------------------------------------------------------------------------


class _DirectRouter:
    def __init__(self, capability, tool_name=None):
        self.capability, self.tool_name = capability, tool_name

    def route(self, message):
        from app.capability import CapabilityDecision

        return CapabilityDecision(capability=self.capability, tool_name=self.tool_name)


def test_a_normal_chat_with_personality_and_profile_asks_groq_first():
    """The exact regression: the personality/profile text makes every chat prompt 'long context', which used to put Gemini
    first whatever the configured order said."""
    manager, log, _ = chain(answering="groq", reply="At your service, sir.")
    core = JarvisCore(
        provider=manager,
        memory=SQLiteMemoryStore(":memory:"),
        capability_router=_DirectRouter(Capability.DIRECT),
        identity_context=LONG_IDENTITY,
    )

    reply = core.handle_chat("good evening")[0]

    assert reply == "At your service, sir."
    assert log == ["groq"]
    assert manager.last_routing_decision.requirements.long_context is True  # it WAS a long prompt - and still groq first


def test_a_normal_chat_falls_back_through_the_fixed_order():
    manager, log, _ = chain(answering="cloudflare", reply="Still here, sir.")
    core = JarvisCore(
        provider=manager,
        memory=SQLiteMemoryStore(":memory:"),
        capability_router=_DirectRouter(Capability.DIRECT),
        identity_context=LONG_IDENTITY,
    )

    assert core.handle_chat("good evening")[0] == "Still here, sir."
    assert log == ["groq", "openrouter", "gemini", "cloudflare"]


def test_a_search_answer_asks_groq_first():
    manager, log, _ = chain(answering="groq", reply="Python is a language [1].")
    response = SearchResponse(
        query="python", mode=SearchMode.NORMAL, queries_used=["python"],
        results=[SearchResult(title="Python", url="https://python.org", snippet="A language.")],
    )

    GeminiSearchAnswerGenerator(manager).generate(response)

    assert log == ["groq"]


def test_a_capability_decision_falls_back_in_order_past_a_reply_that_is_not_valid():
    log = []
    providers = {
        "groq": Recorder("groq", log, reply="I think this is a search"),  # not JSON
        "openrouter": Recorder("openrouter", log, reply='{"capability": "TELEPORT"}'),  # JSON, not a real capability
        "gemini": Recorder("gemini", log, reply='{"capability": "SEARCH", "tool_name": null}'),
        "cloudflare": Recorder("cloudflare", log),
        "freellmapi": Recorder("freellmapi", log),
    }
    manager = AIProviderManager([AIProviderEntry(name, providers[name]) for name in FIXED_ORDER])

    decision = GeminiCapabilityDecisionProvider(manager).decide("what happened today?")

    assert decision.capability is Capability.SEARCH
    assert log == ["groq", "openrouter", "gemini"]  # in order; nobody after the first acceptable reply


class _SendTool(Tool):
    """A tool with a visible side effect, to prove it happens exactly once."""

    name = "send"
    description = "Records one delivery."
    permission = PermissionLevel.SAFE_ACTION
    natural_language_arguments = True

    def __init__(self):
        self.deliveries = []

    @property
    def parameters(self):
        return [ToolParameter(name="to", type="string"), ToolParameter(name="amount", type="number")]

    def execute(self, to: str = "", amount: float = 0) -> ToolResult:
        self.deliveries.append((to, amount))
        return ToolResult.success(data="delivered")


@pytest.mark.parametrize("answering", FIXED_ORDER)
def test_a_failed_provider_never_makes_a_tool_run_twice(answering):
    """Wherever in the chain the tool call finally gets constructed - after network failures, a 5xx with its retry, a
    rate limit and replies that fail validation - the tool itself runs exactly once."""
    log = []
    failures = {
        "groq": Recorder("groq", log, error=down("Groq")),
        "openrouter": Recorder("openrouter", log, error=ProviderError("OpenRouter returned 503: overloaded")),
        "gemini": Recorder("gemini", log, error=ProviderError("Gemini returned 429: quota. Please retry in 30s.")),
        "cloudflare": Recorder("cloudflare", log, reply="Please go ahead and ask your question."),  # not JSON
        "freellmapi": Recorder("freellmapi", log, error=down("FreeLLMAPI")),
    }
    valid = '{"tool_name": "send", "parameters": {"to": "sam", "amount": 5}}'
    position = FIXED_ORDER.index(answering)
    providers = [
        failures[name] if index < position else Recorder(name, log, reply=valid) for index, name in enumerate(FIXED_ORDER)
    ]
    manager = AIProviderManager(
        [AIProviderEntry(name, provider) for name, provider in zip(FIXED_ORDER, providers)],
        retry_count=1, sleep=lambda seconds: None,
    )
    tool = _SendTool()
    registry = ToolRegistry()
    registry.register(tool)
    core = JarvisCore(
        provider=manager,
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        capability_router=_DirectRouter(Capability.TOOL, tool_name="send"),
    )

    core.handle_chat("use tool send five to sam")

    assert tool.deliveries == [("sam", 5)]  # exactly once
    asked = [name for index, name in enumerate(log) if name not in log[:index]]
    assert asked == FIXED_ORDER[: position + 1]  # first-asked order is the fixed order


def test_a_tool_call_nobody_can_construct_runs_nothing():
    manager, log, _ = chain(answering=None)
    tool = _SendTool()
    registry = ToolRegistry()
    registry.register(tool)
    core = JarvisCore(
        provider=manager,
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        capability_router=_DirectRouter(Capability.TOOL, tool_name="send"),
    )

    with pytest.raises(ProviderError):
        core.handle_chat("use tool send five to sam")

    assert tool.deliveries == []
    assert log == FIXED_ORDER


# --- 3. Validation: a bad reply is never a result ------------------------------------------------------------------------


def test_a_rejected_reply_is_not_retried_on_the_same_provider_and_is_not_a_health_event():
    log = []
    manager = AIProviderManager(
        [AIProviderEntry("groq", Recorder("groq", log, reply="nonsense")),
         AIProviderEntry("openrouter", Recorder("openrouter", log, reply="[1, 2]"))],
        retry_count=3, sleep=lambda seconds: None,
    )

    assert manager.generate_validated("hi", json.loads) == [1, 2]
    assert log == ["groq", "openrouter"]  # groq once - a bad reply is not "transient"
    assert all(health.healthy and health.cooldown_until == 0.0 for health in manager.get_health())


def test_when_every_reply_is_rejected_the_callers_own_error_surfaces_unchanged():
    class Rejected(RuntimeError):
        pass

    def validate(raw):
        raise Rejected(f"cannot use {raw!r}")

    manager, log, _ = chain(answering="groq", reply="nonsense")

    with pytest.raises(Rejected):
        manager.generate_validated("hi", validate)

    assert log == FIXED_ORDER  # each provider asked exactly once, in order


def test_no_answer_at_all_is_all_providers_failed_not_a_validation_error():
    manager, _, _ = chain(answering=None)

    with pytest.raises(AllProvidersFailedError) as caught:
        manager.generate_validated("hi", json.loads)

    assert all(name in str(caught.value) for name in FIXED_ORDER)


def test_a_rejection_never_puts_the_reply_text_in_the_failure_summary(capsys):
    secret_reply = "the user's private note: 4111 1111 1111 1111"
    log = []
    manager = AIProviderManager(
        [AIProviderEntry("groq", Recorder("groq", log, reply=secret_reply)),
         AIProviderEntry("openrouter", Recorder("openrouter", log, error=down("OpenRouter")))],
    )

    with pytest.raises(json.JSONDecodeError):
        manager.generate_validated("hi", json.loads)

    assert "4111" not in capsys.readouterr().out


def test_call_structured_with_a_plain_provider_is_one_call_then_the_validator():
    log = []
    plain = Recorder("plain", log, reply='{"a": 1}')

    assert call_structured(plain, "hi", json.loads) == {"a": 1}
    assert log == ["plain"]

    with pytest.raises(json.JSONDecodeError):
        call_structured(Recorder("plain", log, reply="nope"), "hi", json.loads)


# --- 4. Bounded retries -----------------------------------------------------------------------------------------------------


def test_a_5xx_is_retried_on_the_same_provider_only_retry_count_times_then_the_next_is_asked():
    log, sleeps = [], []
    manager = AIProviderManager(
        [AIProviderEntry("groq", Recorder("groq", log, error=ProviderError("Groq returned 503: overloaded"))),
         AIProviderEntry("openrouter", Recorder("openrouter", log))],
        retry_count=1, retry_delay_seconds=0.5, sleep=sleeps.append,
    )

    assert manager.generate_reply("hi") == "reply from openrouter"
    assert log == ["groq", "groq", "openrouter"]  # 1 retry, never more
    assert sleeps == [0.5]


@pytest.mark.parametrize("error", [
    "Failed to reach Groq: The read operation timed out",
    "Failed to reach Groq: connection refused",
    "Groq returned 401: bad key",
    "Groq returned 429: slow down",
    "Unexpected Groq response: {}",
])
def test_a_timeout_auth_rate_limit_or_garbled_response_is_never_retried_on_the_same_provider(error):
    log = []
    manager = AIProviderManager(
        [AIProviderEntry("groq", Recorder("groq", log, error=ProviderError(error))),
         AIProviderEntry("openrouter", Recorder("openrouter", log))],
        retry_count=5, sleep=lambda seconds: None,
    )

    assert manager.generate_reply("hi") == "reply from openrouter"
    assert log == ["groq", "openrouter"]


def test_the_worst_case_number_of_provider_calls_is_bounded():
    """Every provider answering 5xx every time: (retry_count + 1) attempts each, then a clean failure - never a loop."""
    log = []
    manager = AIProviderManager(
        [AIProviderEntry(name, Recorder(name, log, error=ProviderError(f"{name} returned 500: down"))) for name in FIXED_ORDER],
        retry_count=1, sleep=lambda seconds: None,
    )

    with pytest.raises(AllProvidersFailedError):
        manager.generate_reply("hi")

    assert log == [name for name in FIXED_ORDER for _ in range(2)]


# --- 5. Rate limits: the provider's own wait is honoured, within a ceiling -------------------------------------------------

GEMINI_DAILY_QUOTA = (
    'Gemini returned 429: {"error": {"code": 429, "message": "You exceeded your current quota... Please retry in '
    '10h58m27.096231417s.", "status": "RESOURCE_EXHAUSTED", "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", '
    '"retryDelay": "39507s"}]}}'
)


@pytest.mark.parametrize("message, expected", [
    (GEMINI_DAILY_QUOTA, 39507.0),
    ("Gemini returned 429: Please retry in 10h58m27s.", 10 * 3600 + 58 * 60 + 27),
    ("Groq returned 429: Rate limit reached. Please try again in 7m12s.", 432.0),
    ("Groq returned 429: Please try again in 2.5s", 2.5),
    ("Groq returned 429: slow down", None),
    ("Groq returned 429: retry in 0s", None),
    ("Groq returned 429: retry in a while", None),
])
def test_the_wait_a_rate_limit_error_states(message, expected):
    assert stated_retry_seconds(message) == expected


@pytest.mark.parametrize("message, expected", [
    # Milliseconds - the regression: "450ms" used to be read as 450 MINUTES (27000 s), "0.5ms" as 30 s.
    ("Groq returned 429: Rate limit reached. Please try again in 450ms.", 0.45),
    ("Groq returned 429: Please try again in 0.5ms", 0.0005),
    ("Groq returned 429: Please try again in 1ms", 0.001),
    ("Groq returned 429: Please try again in 999.9ms.", 0.9999),
    ("Groq returned 429: try again in 250 milliseconds", 0.25),
    ("Groq returned 429: try again in 1 millisecond", 0.001),
    ("Groq returned 429: try again in 20 msec", 0.02),
    # Seconds, integer and decimal, compact and spelled out.
    ("Groq returned 429: Please try again in 2s", 2.0),
    ("Groq returned 429: Please try again in 0.5s", 0.5),
    ("Groq returned 429: try again in 30 sec", 30.0),
    ("Groq returned 429: try again in 1 second", 1.0),
    ("Groq returned 429: try again in 12.25 seconds.", 12.25),
    # Minutes.
    ("Groq returned 429: Please try again in 7m", 420.0),
    ("Groq returned 429: Please try again in 1.5m", 90.0),
    ("Groq returned 429: try again in 2 minutes", 120.0),
    ("Groq returned 429: try again in 1 min", 60.0),
    # Hours.
    ("Gemini returned 429: retry in 1h", 3600.0),
    ("Gemini returned 429: retry in 1.5h", 5400.0),
    ("Gemini returned 429: retry in 2 hours", 7200.0),
    # Combinations - including milliseconds next to minutes, the exact case the old pattern confused.
    ("Groq returned 429: Please try again in 7m12.5s", 432.5),
    ("Groq returned 429: Please try again in 1m30s500ms", 90.5),
    ("Groq returned 429: try again in 1 minute and 30 seconds", 90.0),
    ("Gemini returned 429: retry in 1h2m3.5s", 3723.5),
    ("GROQ RETURNED 429: PLEASE TRY AGAIN IN 450MS", 0.45),
    # Google's RetryInfo field, integer and decimal.
    ('Gemini returned 429: {"retryDelay": "30s"}', 30.0),
    ('Gemini returned 429: {"retryDelay": "0.5s"}', 0.5),
    # No usable wait: the caller falls back to its configured default cooldown.
    ('Gemini returned 429: {"retryDelay": "0s"}', None),
    ("Groq returned 429: try again in 0ms", None),
    ("Groq returned 429: try again in 3 days", None),
    ("Groq returned 429: try again in 5 mo", None),
    ("Groq returned 429: try again in -5s", None),
    ("Groq returned 429: try again in " + "9" * 400 + "h", None),
])
def test_every_supported_unit_is_read_correctly(message, expected):
    if expected is None:
        assert stated_retry_seconds(message) is None
    else:
        assert stated_retry_seconds(message) == pytest.approx(expected)


def test_a_millisecond_rate_limit_rests_groq_for_milliseconds_not_hours():
    """The bug this guards: Groq's "try again in 450ms" parked Groq - the FIRST provider - for 7.5 hours."""
    clock = {"now": 1000.0}
    log = []
    groq_fake = Recorder("groq", log, errors=[ProviderError("Groq returned 429: Please try again in 450ms.")])
    manager = AIProviderManager(
        [AIProviderEntry("groq", groq_fake), AIProviderEntry("openrouter", Recorder("openrouter", log))],
        cooldown_seconds=60.0, clock=lambda: clock["now"],
    )

    assert manager.generate_reply("hi") == "reply from openrouter"
    assert manager.get_health()[0].cooldown_until == pytest.approx(1000.45)

    clock["now"] = 1001.0  # one second later: Groq is first in line again
    assert manager.generate_reply("hi") == "reply from groq"
    assert log == ["groq", "openrouter", "groq"]


@pytest.mark.parametrize("cooldown_seconds, max_cooldown_seconds, message, expected_rest", [
    (60.0, 86400.0, "Groq returned 429: slow down", 60.0),  # no stated wait: the default
    (60.0, 30.0, "Groq returned 429: slow down", 30.0),  # the ceiling applies to the default too
    (60.0, 86400.0, "Groq returned 429: try again in 2 hours", 7200.0),
    (60.0, 86400.0, "Groq returned 429: retry in 99999h", 86400.0),  # capped
    (-5.0, 86400.0, "Groq returned 429: slow down", 0.0),  # a misconfigured negative default never goes back in time
    (60.0, -1.0, "Groq returned 429: try again in 2s", 0.0),  # nor does a misconfigured negative ceiling
])
def test_every_cooldown_is_bounded_between_zero_and_the_ceiling(cooldown_seconds, max_cooldown_seconds, message,
                                                                 expected_rest):
    log = []
    manager = AIProviderManager(
        [AIProviderEntry("groq", Recorder("groq", log, error=ProviderError(message))),
         AIProviderEntry("openrouter", Recorder("openrouter", log))],
        cooldown_seconds=cooldown_seconds, max_cooldown_seconds=max_cooldown_seconds, clock=lambda: 500.0,
    )

    manager.generate_reply("hi")

    assert manager.get_health()[0].cooldown_until == pytest.approx(500.0 + expected_rest)


def test_an_exhausted_daily_quota_rests_the_provider_until_its_reset_instead_of_asking_every_minute():
    clock = {"now": 1000.0}
    log = []
    gemini_fake = Recorder("gemini", log, error=ProviderError(GEMINI_DAILY_QUOTA))
    manager = AIProviderManager(
        [AIProviderEntry("gemini", gemini_fake), AIProviderEntry("cloudflare", Recorder("cloudflare", log))],
        cooldown_seconds=60.0, clock=lambda: clock["now"],
    )

    assert manager.generate_reply("hi") == "reply from cloudflare"
    assert manager.get_health()[0].cooldown_until == 1000.0 + 39507.0

    for later in (1061.0, 1000.0 + 3600.0, 1000.0 + 39506.0):  # one minute, one hour, just before the reset
        clock["now"] = later
        manager.generate_reply("hi")
    assert gemini_fake.calls == 1  # not asked again while resting

    clock["now"] = 1000.0 + 39508.0  # after the reset
    manager.generate_reply("hi")
    assert gemini_fake.calls == 2


def test_a_rate_limit_that_states_no_wait_uses_the_configured_cooldown():
    clock = {"now": 0.0}
    log = []
    manager = AIProviderManager(
        [AIProviderEntry("groq", Recorder("groq", log, error=ProviderError("Groq returned 429: slow down"))),
         AIProviderEntry("openrouter", Recorder("openrouter", log))],
        cooldown_seconds=60.0, clock=lambda: clock["now"],
    )

    manager.generate_reply("hi")

    assert manager.get_health()[0].cooldown_until == 60.0


def test_a_stated_wait_is_capped_so_a_garbled_error_cannot_park_a_provider_forever():
    log = []
    manager = AIProviderManager(
        [AIProviderEntry("groq", Recorder("groq", log, error=ProviderError("Groq returned 429: retry in 99999h"))),
         AIProviderEntry("openrouter", Recorder("openrouter", log))],
        max_cooldown_seconds=86400.0, clock=lambda: 0.0,
    )

    manager.generate_reply("hi")

    assert manager.get_health()[0].cooldown_until == 86400.0


def test_a_resting_provider_keeps_its_place_in_the_order():
    clock = {"now": 0.0}
    log = []
    recorders = {name: Recorder(name, log) for name in FIXED_ORDER}
    manager = AIProviderManager([AIProviderEntry(name, recorders[name]) for name in FIXED_ORDER], clock=lambda: clock["now"])
    manager.get_health()[0].cooldown_until = 100.0  # groq resting

    assert manager.generate_reply("hi", requirements=RequestRequirements(structured_output=True)) == "reply from openrouter"
    clock["now"] = 200.0
    assert manager.generate_reply("hi", requirements=RequestRequirements(structured_output=True)) == "reply from groq"


# --- 6. One timeout for every adapter ---------------------------------------------------------------------------------------


def _capture_timeout(monkeypatch, module, response_json):
    seen = {}

    class _Response:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return response_json

    def fake_post(url, **kwargs):
        seen["timeout"] = kwargs["timeout"]
        return _Response()

    monkeypatch.setattr(module.httpx, "post", fake_post)
    return seen


def test_every_adapter_uses_the_configured_timeout_not_a_hard_coded_two_minutes(monkeypatch):
    from app.config import settings
    from app.providers.cloudflare import CloudflareProvider
    from app.providers.freellmapi import FreeLLMAPIProvider
    from app.providers.gemini import GeminiProvider
    from app.providers.groq import GroqProvider
    from app.providers.openrouter import OpenRouterProvider

    monkeypatch.setattr(settings, "ai_provider_timeout_seconds", 7.5)
    for field in ("gemini_api_key", "groq_api_key", "openrouter_api_key", "cloudflare_api_key", "cloudflare_account_id",
                  "freellmapi_api_key"):
        monkeypatch.setattr(settings, field, "test-value")
    openai_shape = {"choices": [{"message": {"content": "ok"}}]}

    seen = _capture_timeout(monkeypatch, gemini, {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})
    assert GeminiProvider().generate_reply("hi") == "ok" and seen["timeout"] == 7.5

    seen = _capture_timeout(monkeypatch, _openai_compatible, openai_shape)
    for provider in (GroqProvider(), OpenRouterProvider(), FreeLLMAPIProvider()):
        seen.clear()
        assert provider.generate_reply("hi") == "ok" and seen["timeout"] == 7.5

    seen = _capture_timeout(monkeypatch, cloudflare, {"success": True, "result": {"response": "ok"}})
    assert CloudflareProvider().generate_reply("hi") == "ok" and seen["timeout"] == 7.5


def test_no_adapter_hard_codes_a_timeout():
    from pathlib import Path

    for name in ("gemini.py", "cloudflare.py", "_openai_compatible.py", "groq.py", "openrouter.py", "freellmapi.py"):
        source = (Path(gemini.__file__).parent / name).read_text(encoding="utf-8")
        assert "timeout=120" not in source, name


# --- 7. One mechanism --------------------------------------------------------------------------------------------------------


def test_no_strict_json_workflow_calls_a_provider_around_the_seam():
    """Every strict-JSON workflow goes through call_structured; none calls `provider.generate_reply` and parses the reply
    itself (which would fail the whole request on the first provider's bad JSON) or builds an order of its own."""
    from pathlib import Path

    app = Path(gemini.__file__).parent.parent
    for relative in ("capability.py", "tools/decision.py", "handle_this/interpretation.py", "orchestration/planner.py",
                     "orchestration/evaluator.py", "context_reasoning/reasoner.py", "context_planning/planner.py",
                     "self_development/planner.py"):
        source = (app / relative).read_text(encoding="utf-8")
        assert "call_structured(" in source, relative
        assert "self.provider.generate_reply(" not in source, relative
        assert "ProviderRouter(" not in source, relative


def test_only_the_manager_builds_a_provider_order():
    from pathlib import Path

    app = Path(gemini.__file__).parent.parent
    users = sorted(
        path.relative_to(app).as_posix()
        for path in app.rglob("*.py")
        if "ProviderRouter(" in path.read_text(encoding="utf-8")
    )
    assert users == ["providers/manager.py"]
