"""
Tests for SearchRouter (app/search/router.py): deterministic DIRECT vs
SEARCH routing by keyword/phrase matching only. No AI/Gemini call is ever
made here - SearchRouter has no dependency that could make one.
"""

import pytest

from app.search.router import SearchDecision, SearchRouter


@pytest.fixture
def router() -> SearchRouter:
    return SearchRouter()


# --- Stable/general knowledge stays DIRECT ----------------------------------


@pytest.mark.parametrize(
    "message",
    [
        "what is quantum entanglement?",
        "how does a transistor work?",
        "explain photosynthesis",
        "hello, how are you?",
        "what is your name?",
        "can you help me write a poem about the ocean?",
    ],
)
def test_stable_questions_route_direct(router, message):
    assert router.route(message) is SearchDecision.DIRECT


# --- Explicit search/look-up requests route SEARCH ---------------------------


@pytest.mark.parametrize(
    "message",
    [
        "search for the best pizza place near me",
        "please search the web for python 3.13 release notes",
        "look up the population of France",
        "look this up for me",
        "google it",
        "do a web search for the tallest mountain",
    ],
)
def test_explicit_search_requests_route_search(router, message):
    assert router.route(message) is SearchDecision.SEARCH


# --- Freshness/currentness indicators route SEARCH ---------------------------


@pytest.mark.parametrize(
    "message",
    [
        "what's the latest news on AI?",
        "what is happening currently in the stock market?",
        "what's the weather today?",
        "what happened yesterday in the news?",
        "any recent updates on the project?",
        "what time is it now?",
        "what's on the agenda this week?",
    ],
)
def test_freshness_indicators_route_search(router, message):
    assert router.route(message) is SearchDecision.SEARCH


# --- Live/external information requests route SEARCH -------------------------


@pytest.mark.parametrize(
    "message",
    [
        "what's the weather in London?",
        "what's the weather forecast for tomorrow?",
        "what was the score of the game last night?",
        "what are the election results?",
        "what's the price of bitcoin?",
        "is the restaurant available for a table tonight?",
    ],
)
def test_live_info_requests_route_search(router, message):
    assert router.route(message) is SearchDecision.SEARCH


# --- Explicit search intent overrides a stable-sounding subject --------------


def test_explicit_search_intent_overrides_stable_topic(router):
    assert (
        router.route("search the web and explain quantum entanglement")
        is SearchDecision.SEARCH
    )


def test_explicit_search_intent_overrides_general_chat(router):
    assert router.route("please search for how photosynthesis works") is (
        SearchDecision.SEARCH
    )


# --- Case-insensitivity -------------------------------------------------------


def test_routing_is_case_insensitive(router):
    assert router.route("SEARCH for cat pictures") is SearchDecision.SEARCH
    assert router.route("What's the LATEST on this?") is SearchDecision.SEARCH


# --- Determinism ---------------------------------------------------------------


def test_routing_is_deterministic_across_calls(router):
    message = "what's the latest on the weather today?"

    first = router.route(message)
    second = router.route(message)

    assert first == second == SearchDecision.SEARCH


# --- False-positive check: ordinary messages containing trigger words --------
#
# Keyword matching is inherently imprecise. These tests were added after
# probing the router with plausible real messages. One genuine, high-impact
# false positive was found and fixed (below); the rest are words the
# routing rules explicitly ask for (current/results/available/weather/
# score/price) and are inherently ambiguous in isolation - these are
# documented as accepted, deterministic trade-offs, not bugs, per the
# explicit instruction to keep the router "deterministic, not another AI
# call" rather than adding fragile intent-detection heuristics.


@pytest.mark.parametrize(
    "message",
    [
        "explain how binary search works",
        "what is a linear search algorithm?",
        "how does depth-first search work?",
        "what's the difference between search and sort algorithms?",
    ],
)
def test_search_as_an_algorithm_name_does_not_trigger_search(router, message):
    """Fixed during M6.9 verification: a bare `search` used as a noun/
    algorithm name (very plausible for a programming-context assistant)
    must not be treated as an explicit search request. Only an actual
    request phrasing does - see the next test."""
    assert router.route(message) is SearchDecision.DIRECT


@pytest.mark.parametrize(
    "message",
    [
        "can you search for binary search tutorials?",
        "please search for information on depth-first search",
        "search the web for sorting algorithm comparisons",
    ],
)
def test_explicit_search_phrasing_still_triggers_search_even_about_search_itself(
    router, message
):
    assert router.route(message) is SearchDecision.SEARCH


@pytest.mark.parametrize(
    "message",
    [
        "explain how electric current works in a circuit",
        "what are the results of this calculation?",
        "what functions are available in the math module?",
        "explain how prices are determined in a free market",
        "how do you score higher on a standardized test?",
    ],
)
def test_known_limitation_ambiguous_trigger_words_still_route_search(router, message):
    """These are stable/general questions that happen to contain a
    requested trigger word (current/results/available/prices/score) used
    in a non-temporal, non-live sense. This is an accepted limitation of
    coarse deterministic keyword matching (see README), not something this
    milestone attempts to fix with added heuristics - documented here so a
    future change to this behavior is deliberate, not accidental."""
    assert router.route(message) is SearchDecision.SEARCH
