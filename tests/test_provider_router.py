"""
Tests for ProviderRouter (app/providers/router.py): the FIXED-ORDER rule. The router returns every entry in exactly the
configured order for every kind of request - nothing is promoted for a capability, a long prompt or a latency preference -
and records unavailable providers and capability matches as information only. Uses plain fake AIProviderEntry objects;
ProviderRouter never calls generate_reply on anything, so no provider call - real or fake - happens here, and no network
call is made anywhere in this file.
"""

import pytest

from app.capability import Capability
from app.providers.base import AIProvider
from app.providers.manager import AIProviderEntry
from app.providers.requirements import LatencyPreference, RequestRequirements, TaskType, classify_request
from app.providers.router import ExcludedCandidate, ProviderRouter, RoutingDecision


class _DummyProvider(AIProvider):
    """ProviderRouter must never call generate_reply on anything - it only
    computes an order. This fake exists purely to satisfy AIProviderEntry's
    type; calling it is treated as a routing bug."""

    def generate_reply(self, message: str) -> str:
        raise AssertionError("ProviderRouter must never call a provider directly")


def make_entries(names):
    return [AIProviderEntry(name=name, provider=_DummyProvider()) for name in names]


#: The user's fixed order.
FIXED_ORDER = ["groq", "openrouter", "gemini", "cloudflare", "freellmapi"]
#: The order this project shipped with before the fixed-order rule - still a valid configuration, and still honoured exactly.
OLD_ORDER = ["gemini", "groq", "openrouter", "cloudflare"]

#: Every kind of request a Jarvis workflow makes (see the call sites of `call_with_requirements` / `call_structured`).
EVERY_KIND_OF_REQUEST = {
    "plain": RequestRequirements(),
    "chat with a long prompt (personality + profile)": classify_request("hello", capability=Capability.DIRECT, context_char_count=7284),
    "chat, reasoning wording": classify_request("explain why", capability=Capability.DIRECT, context_char_count=7284),
    "chat, coding wording": classify_request("fix this bug", capability=Capability.DIRECT, context_char_count=10),
    "search answer": classify_request("news", capability=Capability.SEARCH, context_char_count=9000),
    "tool call": classify_request("use tool x", capability=Capability.TOOL, context_char_count=10),
    "strict JSON (reasoner / planner / Job plan)": RequestRequirements(
        task_type=TaskType.STRUCTURED_OUTPUT, structured_output=True, reasoning=True
    ),
    "low latency": RequestRequirements(latency_preference=LatencyPreference.LOW),
    "everything at once": RequestRequirements(
        tool_use=True, structured_output=True, search_synthesis=True, long_context=True, reasoning=True,
        latency_preference=LatencyPreference.LOW,
    ),
}


@pytest.fixture
def router():
    return ProviderRouter()


# --- 1. The fixed order ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(EVERY_KIND_OF_REQUEST))
@pytest.mark.parametrize("order", [FIXED_ORDER, OLD_ORDER, list(reversed(FIXED_ORDER))])
def test_every_kind_of_request_gets_exactly_the_configured_order(router, kind, order):
    decision = router.route(EVERY_KIND_OF_REQUEST[kind], make_entries(order), now=0.0)

    assert decision.ordered_names == order
    assert decision.selected == order[0]
    assert decision.eligible == order
    assert decision.excluded == []


def test_gemini_is_not_promoted_for_a_long_prompt_or_strict_json(router):
    """The exact regression: with the fixed order, a chat carrying the personality/profile (over the long-context threshold)
    and a strict-JSON request both used to put Gemini first."""
    entries = make_entries(FIXED_ORDER)

    for kind in ("chat with a long prompt (personality + profile)", "search answer", "tool call",
                 "strict JSON (reasoner / planner / Job plan)"):
        decision = router.route(EVERY_KIND_OF_REQUEST[kind], entries, now=0.0)
        assert decision.ordered_names[0] == "groq", kind
        assert decision.ordered_names.index("gemini") == 2, kind


def test_groq_is_not_promoted_for_a_latency_preference(router):
    decision = router.route(RequestRequirements(latency_preference=LatencyPreference.LOW), make_entries(OLD_ORDER), now=0.0)

    assert decision.ordered_names == OLD_ORDER
    assert decision.selected == "gemini"


# --- 2. Advisory information never changes the order ------------------------------------------------------------------


def test_capability_scores_are_recorded_but_advisory(router):
    decision = router.route(RequestRequirements(structured_output=True, long_context=True), make_entries(FIXED_ORDER), now=0.0)

    assert decision.scores["gemini"] == 4  # declared for both - recorded...
    assert decision.scores["groq"] == 0
    assert decision.ordered_names == FIXED_ORDER  # ...and the order is untouched


def test_providers_unvalidated_for_strict_json_are_listed_not_excluded(router):
    decision = router.route(RequestRequirements(tool_use=True), make_entries(FIXED_ORDER), now=0.0)

    assert decision.unvalidated == ["groq", "openrouter", "cloudflare", "freellmapi"]
    assert decision.excluded == []
    assert decision.eligible == FIXED_ORDER


def test_a_plain_request_lists_nobody_as_unvalidated(router):
    decision = router.route(RequestRequirements(search_synthesis=True, long_context=True), make_entries(FIXED_ORDER), now=0.0)

    assert decision.unvalidated == []


# --- 3. Unavailable providers keep their place -------------------------------------------------------------------------


def test_disabled_provider_is_recorded_and_keeps_its_place(router):
    entries = make_entries(FIXED_ORDER)
    entries[0].health.enabled = False  # groq

    decision = router.route(RequestRequirements(), entries, now=0.0)

    assert decision.ordered_names == FIXED_ORDER  # not moved to the back
    assert decision.selected == "openrouter"  # the first AVAILABLE one, in order
    assert decision.eligible == FIXED_ORDER[1:]
    assert ExcludedCandidate(name="groq", reason="disabled") in decision.excluded


def test_cooling_down_provider_is_recorded_and_keeps_its_place(router):
    entries = make_entries(FIXED_ORDER)
    entries[1].health.cooldown_until = 100.0  # openrouter

    decision = router.route(RequestRequirements(), entries, now=50.0)

    assert decision.ordered_names == FIXED_ORDER
    assert decision.selected == "groq"
    assert {c.name: c.reason for c in decision.excluded} == {"openrouter": "cooling down (50s left)"}


def test_cooldown_that_has_expired_is_not_excluded(router):
    entries = make_entries(FIXED_ORDER)
    entries[0].health.cooldown_until = 100.0

    decision = router.route(RequestRequirements(), entries, now=200.0)

    assert decision.selected == "groq"
    assert decision.excluded == []


def test_all_unavailable_still_gives_the_configured_order(router):
    entries = make_entries(FIXED_ORDER)
    for entry in entries:
        entry.health.enabled = False

    decision = router.route(RequestRequirements(tool_use=True), entries, now=0.0)

    assert decision.ordered_names == FIXED_ORDER
    assert decision.eligible == []
    assert len(decision.excluded) == 5
    assert decision.selected == "groq"
    assert "configured order" in decision.reason


def test_empty_entries_list_selects_nothing(router):
    decision = router.route(RequestRequirements(), [], now=0.0)

    assert decision.ordered_names == []
    assert decision.selected is None


def test_routing_is_deterministic_across_repeated_calls(router):
    entries = make_entries(FIXED_ORDER)
    requirements = RequestRequirements(tool_use=True, latency_preference=LatencyPreference.LOW)

    first = router.route(requirements, entries, now=10.0)
    second = router.route(requirements, entries, now=10.0)

    assert first == second


# --- 4. Routing decision record -----------------------------------------------------------------------------------------


def test_decision_record_shape(router):
    entries = make_entries(FIXED_ORDER)
    requirements = RequestRequirements(structured_output=True)

    decision = router.route(requirements, entries, now=0.0)

    assert isinstance(decision, RoutingDecision)
    assert decision.requirements is requirements
    assert isinstance(decision.ordered_names, list)
    assert isinstance(decision.eligible, list)
    assert isinstance(decision.excluded, list)
    assert all(isinstance(c, ExcludedCandidate) for c in decision.excluded)
    assert isinstance(decision.scores, dict)
    assert isinstance(decision.unvalidated, list)
    assert isinstance(decision.reason, str) and decision.reason


def test_decision_record_never_contains_secret_shaped_text(router):
    entries = make_entries(FIXED_ORDER)
    entries[2].health.enabled = False
    requirements = RequestRequirements(tool_use=True, structured_output=True)

    decision = router.route(requirements, entries, now=0.0)

    haystack = " ".join(
        [decision.reason]
        + decision.ordered_names
        + decision.eligible
        + decision.unvalidated
        + [f"{c.name}:{c.reason}" for c in decision.excluded]
    )
    for forbidden in ("Bearer", "Authorization", "api_key", "sk-", "token="):
        assert forbidden not in haystack


def test_decision_record_never_holds_the_request_message(router):
    # RequestRequirements structurally never stores the raw message (see
    # requirements.py) - this pins that ProviderRouter doesn't add one
    # back in anywhere in the decision record.
    entries = make_entries(FIXED_ORDER)

    decision = router.route(RequestRequirements(), entries, now=0.0)

    assert not hasattr(decision.requirements, "message")
    assert not hasattr(decision, "message")
