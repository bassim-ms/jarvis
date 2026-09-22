"""
Tests for ProviderRouter (app/providers/router.py): the M13 deterministic
hybrid routing layer - eligibility filtering, scoring, deterministic
tie-breaking, and the safe-to-log RoutingDecision record. Uses plain fake
AIProviderEntry objects only; ProviderRouter never calls generate_reply on
anything (it only reorders), so no provider call - real or fake - should
ever happen here, and no network call is made anywhere in this file.
"""

import pytest

from app.providers.base import AIProvider
from app.providers.manager import AIProviderEntry
from app.providers.requirements import LatencyPreference, RequestRequirements
from app.providers.router import ExcludedCandidate, ProviderRouter, RoutingDecision


class _DummyProvider(AIProvider):
    """ProviderRouter must never call generate_reply on anything - it only
    computes an order. This fake exists purely to satisfy AIProviderEntry's
    type; calling it is treated as a routing bug."""

    def generate_reply(self, message: str) -> str:
        raise AssertionError("ProviderRouter must never call a provider directly")


def make_entries(names):
    return [AIProviderEntry(name=name, provider=_DummyProvider()) for name in names]


ALL_FOUR = ["gemini", "groq", "openrouter", "cloudflare"]


@pytest.fixture
def router():
    return ProviderRouter()


# --- 1. Basic provider selection --------------------------------------------


def test_basic_selection_with_no_requirements_keeps_configured_order(router):
    entries = make_entries(ALL_FOUR)

    decision = router.route(RequestRequirements(), entries, now=0.0)

    assert decision.ordered_names == ALL_FOUR
    assert decision.selected == "gemini"
    assert decision.eligible == ALL_FOUR
    assert decision.excluded == []


# --- 2. Capability matching --------------------------------------------------


def test_capability_match_promotes_gemini_for_structured_output(router):
    entries = make_entries(ALL_FOUR)
    requirements = RequestRequirements(structured_output=True)

    decision = router.route(requirements, entries, now=0.0)

    assert decision.selected == "gemini"
    assert decision.scores["gemini"] == 2
    assert {c.name for c in decision.excluded} == {"groq", "openrouter", "cloudflare"}
    for excluded in decision.excluded:
        assert excluded.reason == "missing capability: structured_output"


# --- 3. Latency preference ---------------------------------------------------


def test_latency_preference_promotes_groq_even_though_it_is_not_first(router):
    entries = make_entries(ALL_FOUR)  # groq is index 1, not first
    requirements = RequestRequirements(latency_preference=LatencyPreference.LOW)

    decision = router.route(requirements, entries, now=0.0)

    assert decision.selected == "groq"
    assert decision.scores["groq"] == 1
    assert decision.scores["gemini"] == 0


def test_latency_preference_normal_has_no_effect(router):
    entries = make_entries(ALL_FOUR)

    decision = router.route(
        RequestRequirements(latency_preference=LatencyPreference.NORMAL), entries, now=0.0
    )

    assert decision.selected == "gemini"
    assert all(score == 0 for score in decision.scores.values())


# --- 4. Reasoning preference --------------------------------------------------


def test_reasoning_preference_has_no_effect_since_no_provider_claims_it(router):
    """Honest, deterministic degenerate case: no provider currently
    declares ProviderCapability.REASONING (see capabilities.py's
    docstring), so this soft signal is a documented, tested no-op today -
    it never crashes and never changes selection, and will start
    mattering automatically the moment a provider's capability set ever
    adds REASONING, with no router code changes required."""
    entries = make_entries(ALL_FOUR)

    decision = router.route(RequestRequirements(reasoning=True), entries, now=0.0)

    assert decision.selected == "gemini"
    assert all(score == 0 for score in decision.scores.values())


# --- 5. Tool-use requirement --------------------------------------------------


def test_tool_use_requirement_hard_gates_to_gemini_only(router):
    entries = make_entries(ALL_FOUR)

    decision = router.route(RequestRequirements(tool_use=True), entries, now=0.0)

    assert decision.eligible == ["gemini"]
    assert decision.selected == "gemini"
    assert {c.name for c in decision.excluded} == {"groq", "openrouter", "cloudflare"}


# --- 6. Structured-output / search-synthesis requirements ---------------------


def test_structured_output_requirement_hard_gates_to_gemini_only(router):
    entries = make_entries(ALL_FOUR)

    decision = router.route(RequestRequirements(structured_output=True), entries, now=0.0)

    assert decision.eligible == ["gemini"]
    assert decision.selected == "gemini"


def test_search_synthesis_requirement_is_a_soft_preference_not_a_hard_gate(router):
    """search_synthesis deliberately does NOT exclude anyone (see
    capabilities.py's docstring: turning search evidence into a cited
    answer needs no structured/parseable output, so it's architecturally
    no different from GENERAL_CONVERSATION) - it only nudges Gemini ahead
    via score, and every other provider remains fully eligible."""
    entries = make_entries(ALL_FOUR)

    decision = router.route(RequestRequirements(search_synthesis=True), entries, now=0.0)

    assert decision.selected == "gemini"
    assert decision.scores["gemini"] == 2
    assert decision.eligible == ALL_FOUR  # nobody excluded
    assert decision.excluded == []


# --- 7. Unavailable (disabled) providers --------------------------------------


def test_disabled_provider_is_excluded_but_still_in_final_order(router):
    entries = make_entries(ALL_FOUR)
    entries[0].health.enabled = False  # gemini

    decision = router.route(RequestRequirements(), entries, now=0.0)

    assert decision.selected == "groq"
    assert decision.ordered_names[-1] == "gemini"  # still attemptable as a last resort
    assert ExcludedCandidate(name="gemini", reason="disabled") in decision.excluded


# --- 8. Cooling-down providers -------------------------------------------------


def test_cooling_down_provider_is_excluded_but_still_in_final_order(router):
    entries = make_entries(ALL_FOUR)
    entries[0].health.cooldown_until = 100.0  # gemini

    decision = router.route(RequestRequirements(), entries, now=50.0)

    assert decision.selected == "groq"
    assert decision.ordered_names[-1] == "gemini"
    excluded_reasons = {c.name: c.reason for c in decision.excluded}
    assert excluded_reasons["gemini"] == "cooling down (50s left)"


def test_cooldown_that_has_expired_is_not_excluded(router):
    entries = make_entries(ALL_FOUR)
    entries[0].health.cooldown_until = 100.0

    decision = router.route(RequestRequirements(), entries, now=200.0)

    assert decision.selected == "gemini"
    assert decision.excluded == []


# --- 9. All candidates excluded ------------------------------------------------


def test_all_excluded_degrades_safely_to_configured_order(router):
    entries = make_entries(ALL_FOUR)
    for entry in entries:
        entry.health.enabled = False

    decision = router.route(RequestRequirements(tool_use=True), entries, now=0.0)

    # Degrades to exactly the input order - plain M12 behavior - rather
    # than raising or silently dropping anything.
    assert decision.ordered_names == ALL_FOUR
    assert decision.eligible == []
    assert len(decision.excluded) == 4
    assert decision.selected == "gemini"
    assert "falling back to configured order" in decision.reason


def test_empty_entries_list_selects_nothing(router):
    decision = router.route(RequestRequirements(), [], now=0.0)

    assert decision.ordered_names == []
    assert decision.selected is None


# --- 10. Deterministic tie-breaking --------------------------------------------


def test_tie_break_falls_back_to_configured_order(router):
    # groq before openrouter in this configured order - both score 0 with
    # plain requirements, so groq must win on position alone.
    entries = make_entries(["groq", "openrouter", "gemini", "cloudflare"])

    decision = router.route(RequestRequirements(), entries, now=0.0)

    assert decision.selected == "groq"


def test_routing_is_deterministic_across_repeated_calls(router):
    entries = make_entries(ALL_FOUR)
    requirements = RequestRequirements(
        tool_use=True, latency_preference=LatencyPreference.LOW
    )

    first = router.route(requirements, entries, now=10.0)
    second = router.route(requirements, entries, now=10.0)

    assert first.ordered_names == second.ordered_names
    assert first.scores == second.scores
    assert first.selected == second.selected


# --- 11. Routing decision record ------------------------------------------------


def test_decision_record_shape(router):
    entries = make_entries(ALL_FOUR)
    requirements = RequestRequirements(structured_output=True)

    decision = router.route(requirements, entries, now=0.0)

    assert isinstance(decision, RoutingDecision)
    assert decision.requirements is requirements
    assert isinstance(decision.ordered_names, list)
    assert isinstance(decision.eligible, list)
    assert isinstance(decision.excluded, list)
    assert all(isinstance(c, ExcludedCandidate) for c in decision.excluded)
    assert isinstance(decision.scores, dict)
    assert isinstance(decision.reason, str) and decision.reason


# --- 12. No secrets in decision records -----------------------------------------


def test_decision_record_never_contains_secret_shaped_text(router):
    entries = make_entries(ALL_FOUR)
    requirements = RequestRequirements(tool_use=True, structured_output=True)

    decision = router.route(requirements, entries, now=0.0)

    haystack = " ".join(
        [decision.reason]
        + decision.ordered_names
        + decision.eligible
        + [f"{c.name}:{c.reason}" for c in decision.excluded]
    )
    for forbidden in ("Bearer", "Authorization", "api_key", "sk-", "token="):
        assert forbidden not in haystack


def test_decision_record_never_holds_the_request_message(router):
    # RequestRequirements structurally never stores the raw message (see
    # requirements.py) - this pins that ProviderRouter doesn't add one
    # back in anywhere in the decision record.
    entries = make_entries(ALL_FOUR)

    decision = router.route(RequestRequirements(), entries, now=0.0)

    assert not hasattr(decision.requirements, "message")
    assert not hasattr(decision, "message")
