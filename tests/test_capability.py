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


def test_capability_only_has_direct_and_search():
    """M6.13 implements exactly DIRECT and SEARCH - no future capability
    (MEMORY/TOOL/EMAIL/CALENDAR/PRINTER/ROBOTIC_ARM/...) is added as an
    unused enum member."""
    assert {member.name for member in Capability} == {"DIRECT", "SEARCH"}


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
