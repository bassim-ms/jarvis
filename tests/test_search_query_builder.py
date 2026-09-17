"""
Tests for deterministic conversational query enrichment
(app/search/query_builder.py): DeterministicSearchQueryBuilder narrowly
enriches a context-dependent SEARCH follow-up and leaves everything else -
self-contained queries, context-free messages, empty/minimal context -
byte-for-byte unchanged. No network/AI call is involved anywhere in this
module, so these tests need no fakes/mocks.
"""

from app.search.query_builder import (
    DeterministicSearchQueryBuilder,
    SearchQueryBuilder,
)


def make_context(*turns: tuple[str, str]) -> str:
    """Build a plain-text context string in the same shape
    JarvisCore._format_context produces: "Conversation so far:\n" followed
    by "role: content" lines."""
    lines = "\n".join(f"{role}: {content}" for role, content in turns)
    return f"Conversation so far:\n{lines}"


# --- Context-dependent follow-ups are enriched -------------------------------


def test_context_dependent_follow_up_is_enriched_with_the_previous_user_message():
    builder = DeterministicSearchQueryBuilder()
    context = make_context(
        ("user", "what's the latest news about SpaceX?"),
        ("assistant", "Here's the latest on SpaceX..."),
    )

    query = builder.build("is there anything more recent on that?", context)

    assert "SpaceX" in query
    assert "is there anything more recent on that?" in query


def test_what_about_follow_up_is_enriched():
    builder = DeterministicSearchQueryBuilder()
    context = make_context(
        ("user", "what's the latest news about SpaceX?"),
        ("assistant", "Here's the latest on SpaceX..."),
    )

    query = builder.build("what about NASA's response?", context)

    assert "SpaceX" in query
    assert "what about NASA's response?" in query


def test_the_second_one_follow_up_is_enriched():
    builder = DeterministicSearchQueryBuilder()
    context = make_context(
        ("user", "what's the latest news about SpaceX?"),
        ("assistant", "Here's the latest on SpaceX..."),
    )

    query = builder.build("tell me more about the second one", context)

    assert "SpaceX" in query


# --- Self-contained SEARCH queries are unchanged -----------------------------


def test_self_contained_query_remains_byte_for_byte_unchanged():
    builder = DeterministicSearchQueryBuilder()
    context = make_context(
        ("user", "what's the latest news about SpaceX?"),
        ("assistant", "Here's the latest on SpaceX..."),
    )

    query = builder.build("what's the latest news about NASA?", context)

    assert query == "what's the latest news about NASA?"


def test_self_contained_query_unchanged_even_with_no_context_at_all():
    builder = DeterministicSearchQueryBuilder()

    query = builder.build("what's the latest news about NASA?", "")

    assert query == "what's the latest news about NASA?"


# --- Unrelated/context-free messages are unchanged ---------------------------


def test_unrelated_message_with_context_present_is_unchanged():
    builder = DeterministicSearchQueryBuilder()
    context = make_context(
        ("user", "what's the latest news about SpaceX?"),
        ("assistant", "Here's the latest on SpaceX..."),
    )

    query = builder.build("what's the weather in Berlin?", context)

    assert query == "what's the weather in Berlin?"


# --- Empty/minimal context falls back safely ---------------------------------


def test_context_dependent_phrasing_with_empty_context_falls_back_to_original():
    builder = DeterministicSearchQueryBuilder()

    query = builder.build("is there anything more recent on that?", "")

    assert query == "is there anything more recent on that?"


def test_context_dependent_phrasing_with_no_prior_user_message_falls_back():
    """Context that only contains facts (no "user: " line at all - e.g. the
    conversation has only ever had assistant-only content, which can't
    happen in practice but is a safe defensive case) - falls back to the
    raw message instead of guessing."""
    builder = DeterministicSearchQueryBuilder()
    context = "Known facts about the user:\n- name: Bassim"

    query = builder.build("is there anything more recent on that?", context)

    assert query == "is there anything more recent on that?"


def test_context_dependent_phrasing_that_matches_its_own_prior_message_falls_back():
    """If the only 'prior user message' recoverable from context is
    identical to the current message (e.g. a repeated message), enrichment
    would add nothing useful, so the original is returned unchanged."""
    builder = DeterministicSearchQueryBuilder()
    context = make_context(("user", "what about that?"))

    query = builder.build("what about that?", context)

    assert query == "what about that?"


# --- Uses the LAST prior user message, not an earlier one --------------------


def test_uses_the_most_recent_prior_user_message_not_an_earlier_one():
    builder = DeterministicSearchQueryBuilder()
    context = make_context(
        ("user", "what's the latest news about SpaceX?"),
        ("assistant", "Here's the latest on SpaceX..."),
        ("user", "what's the latest news about Tesla?"),
        ("assistant", "Here's the latest on Tesla..."),
    )

    query = builder.build("what about that?", context)

    assert "Tesla" in query
    assert "SpaceX" not in query


# --- Abstract interface -------------------------------------------------------


def test_deterministic_builder_is_a_search_query_builder():
    assert isinstance(DeterministicSearchQueryBuilder(), SearchQueryBuilder)
