"""
Tests for the grounded search-answer layer (app/search/answer.py):
GeminiSearchAnswerGenerator built on the existing AIProvider abstraction,
and its deterministic source validation/deduplication. Uses a fake
AIProvider throughout - no real Gemini API call is made.
"""

import pytest

from app.providers.base import AIProvider, ProviderError
from app.providers.gemini import GeminiProvider
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
