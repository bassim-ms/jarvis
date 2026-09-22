"""
Tests for the grounded search-answer layer (app/search/answer.py):
GeminiSearchAnswerGenerator built on the existing AIProvider abstraction,
and its deterministic source validation/deduplication. Uses a fake
AIProvider throughout - no real Gemini API call is made.
"""

import pytest

from app.providers.base import AIProvider, ProviderError
from app.providers.gemini import GeminiProvider
from app.providers.manager import AIProviderEntry, AIProviderManager
from app.providers.requirements import LONG_CONTEXT_CHAR_THRESHOLD, TaskType
from app.search.answer import (
    GeminiSearchAnswerGenerator,
    SearchAnswer,
    SearchAnswerError,
    SearchSource,
)
from app.search.base import SearchResult
from app.search.modes import SearchMode
from app.search.response import SearchResponse


class FakeAIProvider(AIProvider):
    """Fake AIProvider for testing GeminiSearchAnswerGenerator without a
    real Gemini call. Records the prompt it was given."""

    def __init__(self, reply: str = "", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.last_message: str | None = None

    def generate_reply(self, message: str) -> str:
        self.last_message = message
        if self.error:
            raise self.error
        return self.reply


def make_response(results=None, query="python", mode=SearchMode.NORMAL) -> SearchResponse:
    return SearchResponse(
        query=query,
        mode=mode,
        queries_used=[query],
        results=results if results is not None else [],
    )


RESULT_A = SearchResult(
    title="Python.org",
    url="https://python.org",
    snippet="Official Python site",
    source="searxng",
    query="python",
)
RESULT_B = SearchResult(
    title="Python Docs",
    url="https://docs.python.org",
    snippet="Python documentation",
    source="searxng",
    query="python",
)


# --- Successful grounded answer generation ----------------------------------


def test_generate_returns_answer_and_matching_source():
    provider = FakeAIProvider(
        reply=(
            "Python is a programming language. See https://python.org for "
            "details."
        )
    )
    generator = GeminiSearchAnswerGenerator(provider=provider)
    response = make_response(results=[RESULT_A])

    answer = generator.generate(response)

    assert isinstance(answer, SearchAnswer)
    assert "python.org" in answer.answer
    assert answer.sources == [
        SearchSource(title="Python.org", url="https://python.org", query="python")
    ]


def test_generate_sends_query_mode_queries_used_and_evidence_in_the_prompt():
    provider = FakeAIProvider(reply="answer text")
    generator = GeminiSearchAnswerGenerator(provider=provider)
    response = SearchResponse(
        query="python",
        mode=SearchMode.RESEARCH,
        queries_used=["python", "python 2026"],
        results=[RESULT_A],
    )

    generator.generate(response)

    prompt = provider.last_message
    assert "python" in prompt
    assert "research" in prompt
    assert "python 2026" in prompt
    assert RESULT_A.title in prompt
    assert RESULT_A.url in prompt
    assert RESULT_A.snippet in prompt


def test_generate_instructs_gemini_not_to_invent_evidence():
    provider = FakeAIProvider(reply="answer text")
    generator = GeminiSearchAnswerGenerator(provider=provider)

    generator.generate(make_response(results=[RESULT_A]))

    prompt = provider.last_message
    assert "do not" in prompt.lower()
    assert "invent" in prompt.lower()
    assert "insufficient" in prompt.lower()


# --- Insufficient evidence handling ------------------------------------------


def test_insufficient_evidence_produces_no_fabricated_sources():
    provider = FakeAIProvider(
        reply="The search results don't contain enough information to answer this."
    )
    generator = GeminiSearchAnswerGenerator(provider=provider)
    response = make_response(results=[RESULT_A, RESULT_B])

    answer = generator.generate(response)

    assert "enough information" in answer.answer
    assert answer.sources == []


def test_empty_search_results_still_produces_an_answer_with_no_sources():
    provider = FakeAIProvider(reply="I have no evidence to answer this question.")
    generator = GeminiSearchAnswerGenerator(provider=provider)

    answer = generator.generate(make_response(results=[]))

    assert answer.answer
    assert answer.sources == []
    assert "no search results were found" in provider.last_message


# --- Source validation --------------------------------------------------------


def test_only_cited_results_become_sources():
    provider = FakeAIProvider(
        reply="Details are at https://python.org, not the other page."
    )
    generator = GeminiSearchAnswerGenerator(provider=provider)
    response = make_response(results=[RESULT_A, RESULT_B])

    answer = generator.generate(response)

    assert answer.sources == [
        SearchSource(title="Python.org", url="https://python.org", query="python")
    ]


def test_sources_preserve_real_title_url_and_query_from_the_result():
    provider = FakeAIProvider(reply=f"See {RESULT_A.url} for more.")
    generator = GeminiSearchAnswerGenerator(provider=provider)
    response = make_response(results=[RESULT_A], query="python", mode=SearchMode.FAST)

    answer = generator.generate(response)

    source = answer.sources[0]
    assert source.title == RESULT_A.title
    assert source.url == RESULT_A.url
    assert source.query == RESULT_A.query


# --- Invented/unknown URLs are rejected or ignored ----------------------------


def test_invented_url_is_never_returned_as_a_source():
    provider = FakeAIProvider(
        reply="More info at https://totally-made-up-site.example/fake-page."
    )
    generator = GeminiSearchAnswerGenerator(provider=provider)
    response = make_response(results=[RESULT_A])

    answer = generator.generate(response)

    assert answer.sources == []


def test_modified_real_url_is_not_matched():
    """A URL that merely resembles a real one (e.g. a typo or an added
    path segment) must not be treated as citing the real result."""
    provider = FakeAIProvider(reply="See https://python.org/extra-page for info.")
    generator = GeminiSearchAnswerGenerator(provider=provider)
    response = make_response(results=[RESULT_A])

    answer = generator.generate(response)

    assert answer.sources == []


# --- Duplicate sources ---------------------------------------------------------


def test_repeated_citation_of_the_same_url_is_deduplicated():
    provider = FakeAIProvider(
        reply=(
            f"First, see {RESULT_A.url}. Later in the answer, {RESULT_A.url} "
            "is mentioned again for emphasis."
        )
    )
    generator = GeminiSearchAnswerGenerator(provider=provider)
    response = make_response(results=[RESULT_A])

    answer = generator.generate(response)

    assert answer.sources == [
        SearchSource(title="Python.org", url="https://python.org", query="python")
    ]


# --- M6.10: conversation_context for reference resolution --------------------


def test_conversation_context_appears_in_the_prompt_when_supplied():
    provider = FakeAIProvider(reply="answer text")
    generator = GeminiSearchAnswerGenerator(provider=provider)
    response = make_response(results=[RESULT_A])

    generator.generate(
        response,
        conversation_context=(
            "user: tell me about TUM\nassistant: TUM is a university in Munich."
        ),
    )

    prompt = provider.last_message
    assert "tell me about TUM" in prompt
    assert "TUM is a university in Munich." in prompt


def test_conversation_context_section_is_clearly_labeled_for_reference_resolution_only():
    provider = FakeAIProvider(reply="answer text")
    generator = GeminiSearchAnswerGenerator(provider=provider)
    response = make_response(results=[RESULT_A])

    generator.generate(response, conversation_context="user: tell me about TUM")

    prompt = provider.last_message
    assert "Recent conversation context" in prompt
    assert "for understanding references only" in prompt
    assert "Do NOT treat it as external evidence" in prompt
    assert "never invent facts" in prompt.lower()


def test_conversation_context_section_is_absent_when_empty():
    provider = FakeAIProvider(reply="answer text")
    generator = GeminiSearchAnswerGenerator(provider=provider)
    response = make_response(results=[RESULT_A])

    generator.generate(response, conversation_context="")

    assert "Recent conversation context" not in provider.last_message


def test_generate_without_the_context_argument_is_backward_compatible():
    """Omitting conversation_context entirely must produce the exact same
    prompt as explicitly passing an empty one - existing callers (and
    existing tests written before M6.10) keep working unchanged."""
    provider_no_arg = FakeAIProvider(reply="answer text")
    provider_empty_arg = FakeAIProvider(reply="answer text")
    response = make_response(results=[RESULT_A])

    GeminiSearchAnswerGenerator(provider=provider_no_arg).generate(response)
    GeminiSearchAnswerGenerator(provider=provider_empty_arg).generate(
        response, conversation_context=""
    )

    assert provider_no_arg.last_message == provider_empty_arg.last_message


def test_urls_mentioned_only_in_conversation_context_are_not_treated_as_sources():
    """Grounding guarantee (M6.8) holds even with conversation context
    present: a URL that only appears in the conversation_context (not in
    the actual search evidence) never becomes a SearchSource, even if
    Gemini's reply happens to repeat it."""
    provider = FakeAIProvider(
        reply=(
            f"Following up on https://not-a-search-result.example and "
            f"{RESULT_A.url}, here is the answer."
        )
    )
    generator = GeminiSearchAnswerGenerator(provider=provider)
    response = make_response(results=[RESULT_A])

    context = "user: check https://not-a-search-result.example\nassistant: ok"
    answer = generator.generate(response, conversation_context=context)

    assert answer.sources == [
        SearchSource(title="Python.org", url="https://python.org", query="python")
    ]


# --- Provider/API failure handling --------------------------------------------


def test_provider_failure_raises_search_answer_error():
    provider = FakeAIProvider(error=ProviderError("Gemini is down"))
    generator = GeminiSearchAnswerGenerator(provider=provider)

    with pytest.raises(SearchAnswerError, match="Gemini is down"):
        generator.generate(make_response(results=[RESULT_A]))


# --- Dependency injection / structured objects --------------------------------


def test_default_provider_is_gemini_provider():
    generator = GeminiSearchAnswerGenerator()

    assert isinstance(generator.provider, GeminiProvider)


def test_search_answer_and_source_are_plain_comparable_dataclasses():
    source = SearchSource(title="t", url="https://x.com", query="q")
    answer = SearchAnswer(answer="a", sources=[source])

    assert answer == SearchAnswer(
        answer="a", sources=[SearchSource(title="t", url="https://x.com", query="q")]
    )


# --- M14: RequestRequirements wiring -------------------------------------------
#
# generate() now builds a RequestRequirements and calls the provider through
# call_with_requirements (app/providers/manager.py) instead of generate_reply
# directly. Every test above this section uses a plain FakeAIProvider and
# already proves that's unaffected (call_with_requirements is a no-op for a
# non-AIProviderManager provider); these tests cover the new behavior itself,
# using a real AIProviderManager to observe what was actually built.


def test_generate_routes_through_a_real_manager_and_tags_search_synthesis():
    manager = AIProviderManager(
        [AIProviderEntry(name="only", provider=FakeAIProvider(reply="answer"))]
    )
    generator = GeminiSearchAnswerGenerator(provider=manager)

    generator.generate(make_response(results=[RESULT_A]))

    decision = manager.last_routing_decision
    assert decision is not None
    assert decision.requirements.task_type is TaskType.SEARCH_SYNTHESIS
    assert decision.requirements.search_synthesis is True
    # A soft preference, never a hard gate (see app/providers/router.py) -
    # the single registered provider is still eligible, never excluded.
    assert decision.excluded == []


def test_generate_long_context_flag_reflects_the_assembled_prompt_length():
    manager = AIProviderManager(
        [AIProviderEntry(name="only", provider=FakeAIProvider(reply="answer"))]
    )
    generator = GeminiSearchAnswerGenerator(provider=manager)
    # A single huge result snippet pushes the assembled evidence prompt
    # well past LONG_CONTEXT_CHAR_THRESHOLD - real execution context (the
    # actual payload size), not a keyword match.
    huge_result = SearchResult(
        title="Big",
        url="https://big.example",
        snippet="x" * (LONG_CONTEXT_CHAR_THRESHOLD + 1),
        source="searxng",
        query="python",
    )

    generator.generate(make_response(results=[huge_result]))

    assert manager.last_routing_decision.requirements.long_context is True


def test_generate_short_response_is_not_flagged_long_context():
    manager = AIProviderManager(
        [AIProviderEntry(name="only", provider=FakeAIProvider(reply="answer"))]
    )
    generator = GeminiSearchAnswerGenerator(provider=manager)

    generator.generate(make_response(results=[RESULT_A]))

    assert manager.last_routing_decision.requirements.long_context is False


def test_generate_stays_backward_compatible_with_a_plain_provider():
    # Every test above this section already exercises this implicitly
    # (they all use FakeAIProvider, never a manager) - this pins the
    # contract explicitly: a plain AIProvider must never receive a
    # `requirements` keyword argument it doesn't understand.
    provider = FakeAIProvider(reply="a plain answer")
    generator = GeminiSearchAnswerGenerator(provider=provider)

    answer = generator.generate(make_response(results=[RESULT_A]))

    assert answer.answer == "a plain answer"


def test_generate_never_leaks_the_query_or_evidence_into_the_routing_decision():
    manager = AIProviderManager(
        [AIProviderEntry(name="only", provider=FakeAIProvider(reply="answer"))]
    )
    generator = GeminiSearchAnswerGenerator(provider=manager)
    secret_query = "sk-super-secret-query-do-not-leak"

    generator.generate(make_response(results=[RESULT_A], query=secret_query))

    decision = manager.last_routing_decision
    assert not hasattr(decision.requirements, "message")
    assert not hasattr(decision.requirements, "query")
    assert secret_query not in decision.reason
    assert all(secret_query not in c.reason for c in decision.excluded)
