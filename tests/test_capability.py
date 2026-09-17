"""
Tests for the Capability Decision Layer (app/capability.py):
Capability/CapabilityDecision representation and
DeterministicCapabilityRouter, which delegates to SearchRouter (M6.9) for
the actual DIRECT/SEARCH keyword matching rather than duplicating it -
that matching's own exhaustive behavior is covered by
tests/test_search_router.py; these tests only check the delegation and
the new representation are correct.
"""

import pytest

from app.capability import (
    Capability,
    CapabilityDecision,
    CapabilityRouter,
    DeterministicCapabilityRouter,
)
from app.search.router import SearchDecision, SearchRouter


# --- Capability representation ------------------------------------------------


def test_capability_only_has_direct_search_and_tool():
    """M6.13/M6.14 implement exactly DIRECT, SEARCH, and TOOL - no other
    future capability (MEMORY/EMAIL/CALENDAR/PRINTER/ROBOTIC_ARM/...) is
    added as an unused enum member."""
    assert {member.name for member in Capability} == {"DIRECT", "SEARCH", "TOOL"}


def test_capability_decision_holds_the_selected_capability():
    decision = CapabilityDecision(capability=Capability.SEARCH)

    assert decision.capability is Capability.SEARCH


def test_capability_decision_is_a_comparable_plain_dataclass():
    assert CapabilityDecision(capability=Capability.DIRECT) == CapabilityDecision(
        capability=Capability.DIRECT
    )
    assert CapabilityDecision(capability=Capability.DIRECT) != CapabilityDecision(
        capability=Capability.SEARCH
    )


def test_capability_router_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        CapabilityRouter()


# --- Deterministic DIRECT decisions --------------------------------------------


def test_stable_knowledge_question_routes_direct():
    router = DeterministicCapabilityRouter()

    decision = router.route("what is quantum entanglement?")

    assert decision == CapabilityDecision(capability=Capability.DIRECT)


def test_calculation_request_routes_direct():
    router = DeterministicCapabilityRouter()

    decision = router.route("what is 15% of 240?")

    assert decision.capability is Capability.DIRECT


def test_writing_request_routes_direct():
    router = DeterministicCapabilityRouter()

    decision = router.route("write a short poem about the sea")

    assert decision.capability is Capability.DIRECT


def test_coding_explanation_routes_direct():
    router = DeterministicCapabilityRouter()

    decision = router.route("explain how binary search works")

    assert decision.capability is Capability.DIRECT


# --- Deterministic SEARCH decisions --------------------------------------------


def test_explicit_search_request_routes_search():
    router = DeterministicCapabilityRouter()

    decision = router.route("please search the web for the TUM deadline")

    assert decision == CapabilityDecision(capability=Capability.SEARCH)


def test_look_up_request_routes_search():
    router = DeterministicCapabilityRouter()

    decision = router.route("can you look that up for me?")

    assert decision.capability is Capability.SEARCH


def test_freshness_request_routes_search():
    router = DeterministicCapabilityRouter()

    decision = router.route("what's the latest news on AI?")

    assert decision.capability is Capability.SEARCH


def test_live_information_request_routes_search():
    router = DeterministicCapabilityRouter()

    decision = router.route("what's the weather in Berlin right now?")

    assert decision.capability is Capability.SEARCH


def test_explicit_search_overrides_a_stable_seeming_subject():
    router = DeterministicCapabilityRouter()

    decision = router.route("search the web and explain quantum entanglement")

    assert decision.capability is Capability.SEARCH


# --- Existing false-positive protections carry through ------------------------


def test_search_used_as_an_algorithm_name_still_routes_direct():
    """The bare word "search" (as in "binary search") must not trigger
    SEARCH - this is the same false-positive protection SearchRouter
    already has (M6.9); DeterministicCapabilityRouter must not weaken it
    by delegating differently."""
    router = DeterministicCapabilityRouter()

    decision = router.route("how does depth-first search work?")

    assert decision.capability is Capability.DIRECT


# --- Delegation to SearchRouter, not duplicated logic --------------------------


def test_default_search_router_is_a_bare_search_router():
    router = DeterministicCapabilityRouter()

    assert isinstance(router.search_router, SearchRouter)


def test_injected_search_router_is_used_verbatim():
    class RecordingSearchRouter(SearchRouter):
        def __init__(self):
            self.last_message = None

        def route(self, message):
            self.last_message = message
            return SearchDecision.SEARCH

    search_router = RecordingSearchRouter()
    router = DeterministicCapabilityRouter(search_router=search_router)

    decision = router.route("anything at all")

    assert search_router.last_message == "anything at all"
    assert decision.capability is Capability.SEARCH


def test_deterministic_router_translates_direct_decision_too():
    class AlwaysDirectSearchRouter(SearchRouter):
        def route(self, message):
            return SearchDecision.DIRECT

    router = DeterministicCapabilityRouter(search_router=AlwaysDirectSearchRouter())

    decision = router.route("search the web for anything")

    assert decision.capability is Capability.DIRECT


def test_deterministic_capability_router_is_a_capability_router():
    assert isinstance(DeterministicCapabilityRouter(), CapabilityRouter)


# --- M6.14: TOOL capability representation -------------------------------------


def test_capability_decision_can_represent_tool_with_a_tool_name():
    decision = CapabilityDecision(capability=Capability.TOOL, tool_name="echo")

    assert decision.capability is Capability.TOOL
    assert decision.tool_name == "echo"


def test_capability_decision_tool_name_defaults_to_none():
    decision = CapabilityDecision(capability=Capability.SEARCH)

    assert decision.tool_name is None


# --- M6.14: deterministic TOOL routing ------------------------------------------


def test_explicit_use_tool_with_name_routes_tool():
    router = DeterministicCapabilityRouter()

    decision = router.route("use tool echo")

    assert decision == CapabilityDecision(capability=Capability.TOOL, tool_name="echo")


def test_run_tool_colon_syntax_routes_tool():
    router = DeterministicCapabilityRouter()

    decision = router.route("run tool: echo")

    assert decision == CapabilityDecision(capability=Capability.TOOL, tool_name="echo")


def test_call_and_invoke_tool_phrasings_both_route_tool():
    router = DeterministicCapabilityRouter()

    assert router.route("call tool echo").capability is Capability.TOOL
    assert router.route("invoke tool echo").capability is Capability.TOOL


def test_tool_invocation_is_case_insensitive_but_name_is_captured_verbatim():
    router = DeterministicCapabilityRouter()

    decision = router.route("USE TOOL Echo")

    assert decision.capability is Capability.TOOL
    assert decision.tool_name == "Echo"  # not lowercased/normalized


def test_use_tool_with_no_name_routes_tool_with_no_tool_name():
    """An explicit tool-invocation phrasing with nothing named is still
    TOOL - the CapabilityRouter never guesses a tool - but tool_name is
    None, the explicit typed "nothing to dispatch to" representation."""
    router = DeterministicCapabilityRouter()

    decision = router.route("use tool")

    assert decision == CapabilityDecision(capability=Capability.TOOL, tool_name=None)


def test_tool_trigger_takes_priority_over_a_search_triggering_word():
    """"weather" alone is a SEARCH live-info trigger (M6.9), but an
    explicit tool invocation naming it must still win - TOOL is the most
    deliberate/unambiguous of the three signals."""
    router = DeterministicCapabilityRouter()

    decision = router.route("use tool weather")

    assert decision == CapabilityDecision(capability=Capability.TOOL, tool_name="weather")


# --- M6.14: TOOL false-positive protection --------------------------------------


def test_merely_mentioning_a_tool_does_not_trigger_tool_capability():
    """No explicit invocation phrasing at the start of the message - stays
    DIRECT, same as a stable question. TOOL must not fire on every
    incidental mention of the word "tool"."""
    router = DeterministicCapabilityRouter()

    decision = router.route("what tool should I use for tightening this bolt?")

    assert decision.capability is Capability.DIRECT


def test_tool_word_mid_message_does_not_trigger_tool_capability():
    router = DeterministicCapabilityRouter()

    decision = router.route("I want to use tool steel in this design, is that wise?")

    assert decision.capability is Capability.DIRECT


# --- M6.14: no regression for DIRECT/SEARCH -------------------------------------


def test_stable_knowledge_question_still_routes_direct_after_tool_addition():
    router = DeterministicCapabilityRouter()

    decision = router.route("what is quantum entanglement?")

    assert decision == CapabilityDecision(capability=Capability.DIRECT)


def test_freshness_request_still_routes_search_after_tool_addition():
    router = DeterministicCapabilityRouter()

    decision = router.route("what's the latest news on AI?")

    assert decision == CapabilityDecision(capability=Capability.SEARCH)
