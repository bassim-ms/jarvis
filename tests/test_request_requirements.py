"""
Tests for the M13 request-side routing metadata
(app/providers/requirements.py): RequestRequirements defaults and
classify_request()'s deterministic classification. No network calls, no AI
calls - classify_request is plain keyword/pattern matching.
"""

from app.capability import Capability
from app.providers.requirements import (
    LONG_CONTEXT_CHAR_THRESHOLD,
    LatencyPreference,
    RequestRequirements,
    TaskType,
    classify_request,
)


def test_default_requirements_are_least_demanding():
    req = RequestRequirements()

    assert req.task_type is TaskType.GENERAL
    assert req.reasoning is False
    assert req.latency_preference is LatencyPreference.NORMAL
    assert req.tool_use is False
    assert req.structured_output is False
    assert req.long_context is False
    assert req.search_synthesis is False


def test_requirements_are_frozen():
    req = RequestRequirements()
    try:
        req.reasoning = True
        assert False, "RequestRequirements must be immutable"
    except AttributeError:
        pass


def test_classify_plain_message_is_general():
    req = classify_request("what's the weather like where you are")

    assert req.task_type is TaskType.GENERAL
    assert req.reasoning is False
    assert req.tool_use is False
    assert req.structured_output is False
    assert req.search_synthesis is False


def test_classify_detects_coding_keywords():
    req = classify_request("can you help me debug this function")

    assert req.task_type is TaskType.CODING
    assert req.reasoning is True


def test_classify_detects_reasoning_keywords():
    req = classify_request("explain why the sky is blue")

    assert req.task_type is TaskType.REASONING
    assert req.reasoning is True


def test_classify_search_capability_sets_search_synthesis():
    req = classify_request("what's happening in the news today", capability=Capability.SEARCH)

    assert req.task_type is TaskType.SEARCH_SYNTHESIS
    assert req.search_synthesis is True
    assert req.tool_use is False


def test_classify_tool_capability_sets_tool_use():
    req = classify_request("use tool echo hello", capability=Capability.TOOL)

    assert req.task_type is TaskType.TOOL_USE
    assert req.tool_use is True
    assert req.search_synthesis is False


def test_classify_direct_capability_falls_through_to_keyword_matching():
    req = classify_request("write a function to reverse a string", capability=Capability.DIRECT)

    assert req.task_type is TaskType.CODING


def test_classify_long_context_threshold():
    below = classify_request("hi", context_char_count=LONG_CONTEXT_CHAR_THRESHOLD)
    above = classify_request("hi", context_char_count=LONG_CONTEXT_CHAR_THRESHOLD + 1)

    assert below.long_context is False
    assert above.long_context is True


def test_classify_long_context_applies_to_search_and_tool_too():
    search_req = classify_request(
        "news", capability=Capability.SEARCH, context_char_count=LONG_CONTEXT_CHAR_THRESHOLD + 1
    )
    tool_req = classify_request(
        "use tool echo", capability=Capability.TOOL, context_char_count=LONG_CONTEXT_CHAR_THRESHOLD + 1
    )

    assert search_req.long_context is True
    assert tool_req.long_context is True


def test_classify_never_stores_the_raw_message():
    req = classify_request("my api key is sk-super-secret-do-not-log-this")

    assert not any("sk-super-secret" in str(value) for value in vars(req).values())
