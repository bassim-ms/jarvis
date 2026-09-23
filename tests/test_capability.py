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
    CapabilityDecisionError,
    CapabilityDecisionProvider,
    CapabilityRouter,
    DeterministicCapabilityRouter,
    GeminiCapabilityDecisionProvider,
    HybridCapabilityRouter,
)
from app.providers.base import AIProvider, ProviderError
from app.search.router import SearchDecision, SearchRouter


# --- Capability representation ------------------------------------------------


def test_capability_only_has_the_implemented_set():
    """M6.13/M6.14 implement DIRECT, SEARCH, and TOOL; M15 adds TASK,
    TASK_STATUS, and CONFIRMATION - no other future capability (MEMORY/
    EMAIL/CALENDAR/PRINTER/ROBOTIC_ARM/...) is added as an unused enum
    member."""
    assert {member.name for member in Capability} == {
        "DIRECT",
        "SEARCH",
        "TOOL",
        "TASK",
        "TASK_STATUS",
        "CONFIRMATION",
    }


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


# --- M6.16: natural-language "the X tool" phrasing ------------------------------


def test_use_the_x_tool_phrasing_routes_tool():
    router = DeterministicCapabilityRouter()

    decision = router.route("use the echo tool")

    assert decision == CapabilityDecision(capability=Capability.TOOL, tool_name="echo")


def test_use_the_x_tool_with_trailing_text_routes_tool():
    router = DeterministicCapabilityRouter()

    decision = router.route("use the echo tool to say hello")

    assert decision == CapabilityDecision(capability=Capability.TOOL, tool_name="echo")


def test_mention_pattern_works_without_an_article():
    router = DeterministicCapabilityRouter()

    decision = router.route("use echo tool")

    assert decision == CapabilityDecision(capability=Capability.TOOL, tool_name="echo")


def test_mention_pattern_supports_run_call_invoke_too():
    router = DeterministicCapabilityRouter()

    assert router.route("run the weather tool").tool_name == "weather"
    assert router.route("call the echo tool").tool_name == "echo"
    assert router.route("invoke the echo tool").tool_name == "echo"


def test_mention_pattern_still_requires_the_message_to_start_with_it():
    """Same start-anchoring as the original invocation pattern - a tool
    mention that isn't the message's own opening request stays DIRECT."""
    router = DeterministicCapabilityRouter()

    decision = router.route("I guess I should use the echo tool for this")

    assert decision.capability is Capability.DIRECT


def test_generic_use_a_tool_phrasing_without_a_name_does_not_match():
    """"use a tool" alone has no second "tool" word after the captured
    token, so it correctly falls through rather than misfiring."""
    router = DeterministicCapabilityRouter()

    decision = router.route("use a tool to check the weather")

    assert decision.capability is not Capability.TOOL


def test_original_invocation_pattern_still_takes_priority():
    """"use tool echo" is unambiguous under the original M6.14 pattern -
    the new M6.16 pattern must not interfere with or shadow it."""
    router = DeterministicCapabilityRouter()

    decision = router.route("use tool echo")

    assert decision == CapabilityDecision(capability=Capability.TOOL, tool_name="echo")


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


# --- M6.15: hybrid capability routing ==========================================


class FakeAIProvider(AIProvider):
    """Fake AIProvider for testing GeminiCapabilityDecisionProvider without
    a real Gemini call. Records the prompt it was given."""

    def __init__(self, reply: str = "", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.last_message: str | None = None

    def generate_reply(self, message: str) -> str:
        self.last_message = message
        if self.error:
            raise self.error
        return self.reply


class FakeCapabilityDecisionProvider(CapabilityDecisionProvider):
    """Records whether/how it was called and returns a fixed
    CapabilityDecision, or raises - decoupled from
    GeminiCapabilityDecisionProvider's own prompt/JSON logic, which is
    covered separately below."""

    def __init__(
        self,
        decision: CapabilityDecision | None = None,
        error: Exception | None = None,
    ):
        self.decision = decision
        self.error = error
        self.last_message: str | None = None
        self.call_count = 0

    def decide(self, message: str) -> CapabilityDecision:
        self.call_count += 1
        self.last_message = message
        if self.error:
            raise self.error
        return self.decision


class FakeDeterministicRouter(CapabilityRouter):
    """Always returns a fixed CapabilityDecision, ignoring the message -
    used to give HybridCapabilityRouter precise, controlled "deterministic
    stage" results without depending on real regex matching."""

    def __init__(self, decision: CapabilityDecision):
        self.decision = decision

    def route(self, message: str) -> CapabilityDecision:
        return self.decision


# --- HybridCapabilityRouter: deterministic fast path (no Gemini call) --------


def test_hybrid_router_obvious_direct_does_not_call_the_decision_provider():
    decision_provider = FakeCapabilityDecisionProvider()
    router = HybridCapabilityRouter(decision_provider=decision_provider)

    decision = router.route("what is quantum entanglement?")

    assert decision == CapabilityDecision(capability=Capability.DIRECT)
    assert decision_provider.call_count == 0


def test_hybrid_router_obvious_search_does_not_call_the_decision_provider():
    decision_provider = FakeCapabilityDecisionProvider()
    router = HybridCapabilityRouter(decision_provider=decision_provider)

    decision = router.route("what's the latest news on AI?")

    assert decision == CapabilityDecision(capability=Capability.SEARCH)
    assert decision_provider.call_count == 0


def test_hybrid_router_explicit_tool_invocation_does_not_call_the_decision_provider():
    decision_provider = FakeCapabilityDecisionProvider()
    router = HybridCapabilityRouter(decision_provider=decision_provider)

    decision = router.route("use tool echo")

    assert decision == CapabilityDecision(capability=Capability.TOOL, tool_name="echo")
    assert decision_provider.call_count == 0


def test_hybrid_router_false_positive_protection_still_intact():
    """Existing M6.9/M6.14 false-positive protections still hold through
    the hybrid router without ever reaching Gemini."""
    decision_provider = FakeCapabilityDecisionProvider()
    router = HybridCapabilityRouter(decision_provider=decision_provider)

    decision = router.route("how does depth-first search work?")

    assert decision.capability is Capability.DIRECT
    assert decision_provider.call_count == 0


# --- HybridCapabilityRouter: ambiguous path (Gemini fallback) -----------------


def test_hybrid_router_ambiguous_tool_without_name_invokes_the_decision_provider():
    decision_provider = FakeCapabilityDecisionProvider(
        decision=CapabilityDecision(capability=Capability.DIRECT)
    )
    router = HybridCapabilityRouter(decision_provider=decision_provider)

    router.route("use tool")

    assert decision_provider.call_count == 1


def test_hybrid_router_passes_the_raw_message_to_the_decision_provider():
    decision_provider = FakeCapabilityDecisionProvider(
        decision=CapabilityDecision(capability=Capability.DIRECT)
    )
    router = HybridCapabilityRouter(decision_provider=decision_provider)

    router.route("use tool")

    assert decision_provider.last_message == "use tool"


def test_hybrid_router_returns_the_providers_direct_decision():
    decision_provider = FakeCapabilityDecisionProvider(
        decision=CapabilityDecision(capability=Capability.DIRECT)
    )
    router = HybridCapabilityRouter(
        deterministic_router=FakeDeterministicRouter(
            CapabilityDecision(capability=Capability.TOOL, tool_name=None)
        ),
        decision_provider=decision_provider,
    )

    decision = router.route("use tool")

    assert decision == CapabilityDecision(capability=Capability.DIRECT)


def test_hybrid_router_returns_the_providers_search_decision():
    decision_provider = FakeCapabilityDecisionProvider(
        decision=CapabilityDecision(capability=Capability.SEARCH)
    )
    router = HybridCapabilityRouter(
        deterministic_router=FakeDeterministicRouter(
            CapabilityDecision(capability=Capability.TOOL, tool_name=None)
        ),
        decision_provider=decision_provider,
    )

    decision = router.route("use tool")

    assert decision == CapabilityDecision(capability=Capability.SEARCH)


def test_hybrid_router_returns_the_providers_tool_decision_with_tool_name():
    decision_provider = FakeCapabilityDecisionProvider(
        decision=CapabilityDecision(capability=Capability.TOOL, tool_name="echo")
    )
    router = HybridCapabilityRouter(
        deterministic_router=FakeDeterministicRouter(
            CapabilityDecision(capability=Capability.TOOL, tool_name=None)
        ),
        decision_provider=decision_provider,
    )

    decision = router.route("use tool")

    assert decision == CapabilityDecision(capability=Capability.TOOL, tool_name="echo")


def test_hybrid_router_never_calls_the_decision_provider_more_than_once():
    decision_provider = FakeCapabilityDecisionProvider(
        decision=CapabilityDecision(capability=Capability.DIRECT)
    )
    router = HybridCapabilityRouter(decision_provider=decision_provider)

    router.route("use tool")

    assert decision_provider.call_count == 1


def test_hybrid_router_default_deterministic_router_is_a_bare_one():
    router = HybridCapabilityRouter()

    assert isinstance(router.deterministic_router, DeterministicCapabilityRouter)


def test_hybrid_router_default_decision_provider_is_gemini_backed():
    router = HybridCapabilityRouter()

    assert isinstance(router.decision_provider, GeminiCapabilityDecisionProvider)


def test_hybrid_router_is_a_capability_router():
    assert isinstance(HybridCapabilityRouter(), CapabilityRouter)


# --- HybridCapabilityRouter: provider failure ---------------------------------


def test_hybrid_router_propagates_capability_decision_error_from_the_provider():
    decision_provider = FakeCapabilityDecisionProvider(
        error=CapabilityDecisionError("Gemini is down")
    )
    router = HybridCapabilityRouter(decision_provider=decision_provider)

    with pytest.raises(CapabilityDecisionError, match="Gemini is down"):
        router.route("use tool")


# --- GeminiCapabilityDecisionProvider: prompt/request shape -------------------


def test_decide_sends_the_message_and_the_three_capability_names_in_the_prompt():
    provider = FakeAIProvider(reply='{"capability": "DIRECT", "tool_name": null}')
    decision_provider = GeminiCapabilityDecisionProvider(provider=provider)

    decision_provider.decide("check my calendar for tomorrow")

    prompt = provider.last_message
    assert "check my calendar for tomorrow" in prompt
    assert "DIRECT" in prompt
    assert "SEARCH" in prompt
    assert "TOOL" in prompt


def test_decide_does_not_mention_any_specific_registered_tool_name():
    """No tool descriptions/registry contents are ever sent to Gemini -
    only the fixed capability names."""
    provider = FakeAIProvider(reply='{"capability": "DIRECT", "tool_name": null}')
    decision_provider = GeminiCapabilityDecisionProvider(provider=provider)

    decision_provider.decide("do something")

    assert "echo" not in provider.last_message
    assert "web_search" not in provider.last_message


# --- GeminiCapabilityDecisionProvider: valid structured output ----------------


def test_valid_direct_response_becomes_direct_decision():
    provider = FakeAIProvider(reply='{"capability": "DIRECT", "tool_name": null}')
    decision_provider = GeminiCapabilityDecisionProvider(provider=provider)

    decision = decision_provider.decide("some ambiguous message")

    assert decision == CapabilityDecision(capability=Capability.DIRECT)


def test_valid_search_response_becomes_search_decision():
    provider = FakeAIProvider(reply='{"capability": "SEARCH", "tool_name": null}')
    decision_provider = GeminiCapabilityDecisionProvider(provider=provider)

    decision = decision_provider.decide("some ambiguous message")

    assert decision == CapabilityDecision(capability=Capability.SEARCH)


def test_valid_tool_response_becomes_tool_decision_with_tool_name():
    provider = FakeAIProvider(reply='{"capability": "TOOL", "tool_name": "echo"}')
    decision_provider = GeminiCapabilityDecisionProvider(provider=provider)

    decision = decision_provider.decide("some ambiguous message")

    assert decision == CapabilityDecision(capability=Capability.TOOL, tool_name="echo")


def test_lowercase_capability_value_is_normalized():
    provider = FakeAIProvider(reply='{"capability": "direct", "tool_name": null}')
    decision_provider = GeminiCapabilityDecisionProvider(provider=provider)

    decision = decision_provider.decide("some ambiguous message")

    assert decision.capability is Capability.DIRECT


def test_tool_name_is_preserved_verbatim_not_normalized():
    provider = FakeAIProvider(reply='{"capability": "TOOL", "tool_name": "Echo"}')
    decision_provider = GeminiCapabilityDecisionProvider(provider=provider)

    decision = decision_provider.decide("some ambiguous message")

    assert decision.tool_name == "Echo"


def test_markdown_code_fence_is_tolerated_before_parsing():
    provider = FakeAIProvider(
        reply='```json\n{"capability": "DIRECT", "tool_name": null}\n```'
    )
    decision_provider = GeminiCapabilityDecisionProvider(provider=provider)

    decision = decision_provider.decide("some ambiguous message")

    assert decision == CapabilityDecision(capability=Capability.DIRECT)


def test_tool_name_key_may_be_omitted_entirely_for_non_tool_capabilities():
    provider = FakeAIProvider(reply='{"capability": "DIRECT"}')
    decision_provider = GeminiCapabilityDecisionProvider(provider=provider)

    decision = decision_provider.decide("some ambiguous message")

    assert decision == CapabilityDecision(capability=Capability.DIRECT)


# --- GeminiCapabilityDecisionProvider: strict validation, fails safely --------


def test_invalid_json_raises_capability_decision_error():
    provider = FakeAIProvider(reply="sure, I'd say this is a DIRECT question.")
    decision_provider = GeminiCapabilityDecisionProvider(provider=provider)

    with pytest.raises(CapabilityDecisionError):
        decision_provider.decide("some ambiguous message")


def test_non_object_json_raises_capability_decision_error():
    provider = FakeAIProvider(reply='"DIRECT"')
    decision_provider = GeminiCapabilityDecisionProvider(provider=provider)

    with pytest.raises(CapabilityDecisionError):
        decision_provider.decide("some ambiguous message")


def test_missing_capability_key_raises_capability_decision_error():
    provider = FakeAIProvider(reply='{"tool_name": null}')
    decision_provider = GeminiCapabilityDecisionProvider(provider=provider)

    with pytest.raises(CapabilityDecisionError):
        decision_provider.decide("some ambiguous message")


def test_invalid_capability_value_raises_capability_decision_error():
    provider = FakeAIProvider(reply='{"capability": "AUTOMATION", "tool_name": null}')
    decision_provider = GeminiCapabilityDecisionProvider(provider=provider)

    with pytest.raises(CapabilityDecisionError):
        decision_provider.decide("some ambiguous message")


def test_tool_without_tool_name_raises_capability_decision_error():
    provider = FakeAIProvider(reply='{"capability": "TOOL", "tool_name": null}')
    decision_provider = GeminiCapabilityDecisionProvider(provider=provider)

    with pytest.raises(CapabilityDecisionError):
        decision_provider.decide("some ambiguous message")


def test_tool_with_missing_tool_name_key_raises_capability_decision_error():
    provider = FakeAIProvider(reply='{"capability": "TOOL"}')
    decision_provider = GeminiCapabilityDecisionProvider(provider=provider)

    with pytest.raises(CapabilityDecisionError):
        decision_provider.decide("some ambiguous message")


def test_tool_with_blank_tool_name_raises_capability_decision_error():
    provider = FakeAIProvider(reply='{"capability": "TOOL", "tool_name": "   "}')
    decision_provider = GeminiCapabilityDecisionProvider(provider=provider)

    with pytest.raises(CapabilityDecisionError):
        decision_provider.decide("some ambiguous message")


def test_direct_with_a_non_null_tool_name_raises_capability_decision_error():
    provider = FakeAIProvider(reply='{"capability": "DIRECT", "tool_name": "echo"}')
    decision_provider = GeminiCapabilityDecisionProvider(provider=provider)

    with pytest.raises(CapabilityDecisionError):
        decision_provider.decide("some ambiguous message")


def test_unexpected_extra_field_raises_capability_decision_error():
    provider = FakeAIProvider(
        reply='{"capability": "DIRECT", "tool_name": null, "confidence": 0.9}'
    )
    decision_provider = GeminiCapabilityDecisionProvider(provider=provider)

    with pytest.raises(CapabilityDecisionError):
        decision_provider.decide("some ambiguous message")


# --- GeminiCapabilityDecisionProvider: provider failure -----------------------


def test_underlying_provider_failure_raises_capability_decision_error():
    provider = FakeAIProvider(error=ProviderError("Gemini is down"))
    decision_provider = GeminiCapabilityDecisionProvider(provider=provider)

    with pytest.raises(CapabilityDecisionError, match="Gemini is down"):
        decision_provider.decide("some ambiguous message")


# --- Dependency injection / defaults -------------------------------------------


def test_default_provider_is_gemini_provider():
    from app.providers.gemini import GeminiProvider

    decision_provider = GeminiCapabilityDecisionProvider()

    assert isinstance(decision_provider.provider, GeminiProvider)


def test_gemini_capability_decision_provider_is_a_capability_decision_provider():
    assert isinstance(GeminiCapabilityDecisionProvider(), CapabilityDecisionProvider)


# --- TASK detection (M15) -------------------------------------------------


def test_explicit_task_invocation_is_task_with_captured_goal():
    router = DeterministicCapabilityRouter()

    decision = router.route("create a goal: prepare everything for tomorrow's meeting")

    assert decision.capability is Capability.TASK
    assert decision.goal == "prepare everything for tomorrow's meeting"


@pytest.mark.parametrize(
    "phrase",
    [
        "create a task: organize my downloads",
        "create task: organize my downloads",
        "start a job: organize my downloads",
        "start job: organize my downloads",
        "begin a goal: organize my downloads",
    ]
)
def test_explicit_task_invocation_accepts_goal_job_and_task_wording(phrase):
    router = DeterministicCapabilityRouter()

    decision = router.route(phrase)

    assert decision.capability is Capability.TASK
    assert decision.goal == "organize my downloads"


@pytest.mark.parametrize(
    "message",
    [
        "Prepare everything I need for tomorrow's robotics meeting.",
        "Organize my Downloads folder.",
        "Research robotics competitions and prepare a shortlist.",
        "Plan a birthday party for next week.",
        "Arrange a call with the team.",
        "Set up a new project folder structure.",
        "Put together a summary of this quarter's sales.",
    ]
)
def test_goal_verb_messages_are_task_with_the_whole_message_as_goal(message):
    router = DeterministicCapabilityRouter()

    decision = router.route(message)

    assert decision.capability is Capability.TASK
    assert decision.goal == message


def test_goal_verb_question_is_not_misrouted_to_task():
    router = DeterministicCapabilityRouter()

    decision = router.route("Prepare? What do you mean by that?")

    assert decision.capability is not Capability.TASK


def test_tool_invocation_takes_priority_over_goal_verb():
    router = DeterministicCapabilityRouter()

    decision = router.route("use tool prepare_report")

    assert decision.capability is Capability.TOOL
    assert decision.tool_name == "prepare_report"


def test_goal_verb_mentioned_mid_message_is_not_task():
    router = DeterministicCapabilityRouter()

    decision = router.route("let's discuss how to prepare for finals")

    assert decision.capability is not Capability.TASK


def test_ordinary_direct_and_search_messages_are_unaffected_by_task_detection():
    router = DeterministicCapabilityRouter()

    assert router.route("what is quantum entanglement?").capability is Capability.DIRECT
    assert router.route("what's the latest news on AI?").capability is Capability.SEARCH


# --- TASK_STATUS / CONFIRMATION detection -----------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "what's the status of my task?",
        "what is the status of my job?",
        "status of my task",
        "what are you doing?",
        "what are you currently doing",
        "is that finished?",
        "is it done?",
        "is my task finished?",
        "are you done?",
        "are you finished?",
        "show me my running tasks",
        "list me my active tasks",
        "my running tasks",
        "my in-progress tasks",
    ]
)
def test_status_query_phrasings_route_to_task_status(message):
    router = DeterministicCapabilityRouter()

    decision = router.route(message)

    assert decision.capability is Capability.TASK_STATUS


@pytest.mark.parametrize("message", ["my tasks", "show me my tasks", "list my tasks"])
def test_bare_my_tasks_does_not_route_to_task_status(message):
    """Deliberately excluded - ambiguous with app.tasks (Rhasspy Tasks,
    M9's unrelated todo list); only phrasings implying an in-progress
    GOAL (status/doing/finished/running/active/in-progress) trigger
    TASK_STATUS, never a bare 'my tasks' with no such qualifier."""
    router = DeterministicCapabilityRouter()

    decision = router.route(message)

    assert decision.capability is not Capability.TASK_STATUS


@pytest.mark.parametrize(
    "message,expected_approve",
    [
        ("yes", True),
        ("Yes.", True),
        ("yep", True),
        ("yeah!", True),
        ("sure", True),
        ("ok", True),
        ("okay", True),
        ("confirm", True),
        ("confirmed", True),
        ("approve", True),
        ("approved", True),
        ("go ahead", True),
        ("send it", True),
        ("do it", True),
        ("proceed", True),
        ("no", False),
        ("nope", False),
        ("nah", False),
        ("don't", False),
        ("cancel that", False),
        ("decline", False),
        ("declined", False),
        ("stop", False),
    ]
)
def test_short_confirmation_replies_route_to_confirmation(message, expected_approve):
    router = DeterministicCapabilityRouter()

    decision = router.route(message)

    assert decision.capability is Capability.CONFIRMATION
    assert decision.approve is expected_approve


def test_confirmation_pattern_matches_the_whole_message_not_a_substring():
    """A longer sentence that merely contains 'yes' must NOT be treated
    as a confirmation reply - only a message that IS essentially just the
    short reply counts."""
    router = DeterministicCapabilityRouter()

    decision = router.route("yes, and also can you tell me about Rome")

    assert decision.capability is not Capability.CONFIRMATION


def test_tool_invocation_takes_priority_over_confirmation_and_status_patterns():
    router = DeterministicCapabilityRouter()

    assert router.route("use tool confirm_job").capability is Capability.TOOL
