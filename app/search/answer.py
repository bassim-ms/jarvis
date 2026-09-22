"""
Grounded answer generation over a SearchResponse.

SearchAnswerGenerator turns the structured search context M6.7 already
produces (app/search/response.py) into a free-text answer plus a
deterministic, validated list of sources - never the other way around.
Provider-agnostic at the interface level: GeminiSearchAnswerGenerator is
the only implementation, built entirely on the existing AIProvider
abstraction (app/providers/base.py) - no separate API key, HTTP client, or
AI integration of its own. JarvisCore's SEARCH path (app/core.py) is the
caller, since M6.9.

`generate()` also accepts optional `conversation_context` (M6.10): plain
text built from recent conversation history/facts, used only so a
follow-up question ("what's the latest deadline?" after "tell me about
TUM") can be understood - never as a second source of facts to answer
from, and never fed into the search query itself (that stays exactly the
raw user message - see app/search/router.py and app/tools/web_search.py).

Since M14, `generate()` builds a RequestRequirements
(app/providers/requirements.py) unconditionally tagged search_synthesis
(this class only ever performs search-answer synthesis - there's no
ambiguity to classify), with `long_context` derived from the assembled
prompt's actual length, and calls the provider through
call_with_requirements (app/providers/manager.py) instead of
generate_reply directly. That's a no-op whenever `self.provider` isn't an
AIProviderManager (this class's own default GeminiProvider(), every
existing test's FakeAIProvider, ...) - see call_with_requirements's own
docstring - so this is purely additive for the one caller
(JarvisCore._handle_search, via Core's shared self.provider) that already
wraps an AIProviderManager.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.providers.base import AIProvider, ProviderError
from app.providers.gemini import GeminiProvider
from app.providers.manager import call_with_requirements
from app.providers.requirements import (
    LONG_CONTEXT_CHAR_THRESHOLD,
    RequestRequirements,
    TaskType,
)
from app.search.base import SearchResult
from app.search.response import SearchResponse


class SearchAnswerError(RuntimeError):
    """Raised when a SearchAnswerGenerator can't produce an answer."""


@dataclass
class SearchSource:
    """One citation backing a SearchAnswer.

    `title`, `url`, and `query` are always copied verbatim from a real
    SearchResult in the SearchResponse that was answered - never written
    or modified by the AI provider.
    """

    title: str
    url: str
    query: str


@dataclass
class SearchAnswer:
    """A grounded answer to one SearchResponse: free-text `answer` plus
    the `sources` actually cited, each guaranteed to exist in the
    SearchResponse.results that produced it."""

    answer: str
    sources: list[SearchSource]


class SearchAnswerGenerator(ABC):
    """Turns a SearchResponse into a grounded SearchAnswer."""

    @abstractmethod
    def generate(
        self, response: SearchResponse, conversation_context: str = ""
    ) -> SearchAnswer:
        """Answer the SearchResponse's original query using only its
        results.

        `conversation_context`, if given, is recent prior conversation
        (history/facts, formatted as plain text) - used only to resolve
        references in the question (e.g. "it", "the deadline"), never as
        evidence to answer from. The search query itself is never derived
        from this context (see app/search/router.py and
        app/tools/web_search.py) - it stays exactly the raw message the
        user sent.
        """


class GeminiSearchAnswerGenerator(SearchAnswerGenerator):
    """SearchAnswerGenerator backed by the existing AIProvider/Gemini
    integration."""

    def __init__(self, provider: AIProvider | None = None):
        self.provider = provider or GeminiProvider()

    def generate(
        self, response: SearchResponse, conversation_context: str = ""
    ) -> SearchAnswer:
        prompt = self._build_prompt(response, conversation_context)
        requirements = RequestRequirements(
            task_type=TaskType.SEARCH_SYNTHESIS,
            search_synthesis=True,
            long_context=len(prompt) > LONG_CONTEXT_CHAR_THRESHOLD,
        )

        try:
            raw_answer = call_with_requirements(self.provider, prompt, requirements)
        except ProviderError as exc:
            raise SearchAnswerError(f"Failed to generate an answer: {exc}") from exc

        sources = self._extract_cited_sources(raw_answer, response.results)
        return SearchAnswer(answer=raw_answer.strip(), sources=sources)

    @staticmethod
    def _build_prompt(response: SearchResponse, conversation_context: str = "") -> str:
        if response.results:
            evidence = "\n\n".join(
                f"[{i}] {result.title}\nURL: {result.url}\n{result.snippet}"
                for i, result in enumerate(response.results, start=1)
            )
        else:
            evidence = "(no search results were found)"

        # Empty when no context is given, so the prompt is byte-for-byte
        # identical to the pre-M6.10 prompt in that case.
        context_section = ""
        if conversation_context:
            context_section = (
                "Recent conversation context (for understanding references "
                "only):\n"
                f"{conversation_context}\n\n"
                "Use this context ONLY to understand what the user's "
                "question refers to (e.g. a pronoun or an implied subject "
                "from an earlier message). Do NOT treat it as external "
                "evidence, and do NOT answer any factual or current claim "
                "from it - factual/current claims must come only from the "
                "search evidence below. Never invent facts, sources, or "
                "URLs from the conversation context either.\n\n"
            )

        return (
            "You are answering a user's question using ONLY the search "
            "evidence below. Do not use outside knowledge, and do not "
            "invent facts, sources, URLs, or citations that are not present "
            "in the evidence. If the evidence is insufficient to answer "
            "confidently, say so explicitly instead of guessing.\n\n"
            f"{context_section}"
            f"User question: {response.query}\n"
            f"Search mode: {response.mode.value}\n"
            f"Queries searched: {', '.join(response.queries_used)}\n\n"
            f"Evidence:\n{evidence}\n\n"
            "When you reference a source, include its exact URL from the "
            "evidence above, written exactly as given - never a URL that is "
            "not listed above."
        )

    # Trailing characters to strip from a whitespace-split token before
    # comparing it to a real URL, so "...https://x.com." (end of sentence)
    # still matches, without loosening the match to substring containment
    # (which would let "https://x.com/extra-page" falsely match "https://x.com").
    _TRAILING_PUNCTUATION = ".,;:!?)]}\"'"

    @classmethod
    def _extract_cited_sources(
        cls, raw_answer: str, results: list[SearchResult]
    ) -> list[SearchSource]:
        """Only a real SearchResult's URL appearing as an exact whitespace-
        delimited token in the answer text counts as a citation. This is
        what makes source handling deterministic: an invented URL, or one
        merely derived from a real URL (typo, extra path segment), can
        never match and is silently ignored, and each real URL is
        considered at most once (results are already deduplicated - see
        app/search/quality.py - so this also dedupes the returned sources).
        """
        cited_tokens = {
            token.strip(cls._TRAILING_PUNCTUATION) for token in raw_answer.split()
        }

        sources: list[SearchSource] = []
        seen_urls: set[str] = set()
        for result in results:
            if result.url in seen_urls:
                continue
            if result.url in cited_tokens:
                sources.append(
                    SearchSource(title=result.title, url=result.url, query=result.query)
                )
                seen_urls.add(result.url)
        return sources
