"""
Request-side routing metadata for M13 hybrid provider routing.

RequestRequirements is a small, extensible representation of what a
request needs from an AI provider - never the request's own text.
ProviderRouter (app/providers/router.py) uses it to score/filter
registered providers; AIProviderManager.generate_reply
(app/providers/manager.py) accepts one as an optional `requirements`
keyword argument and does nothing new when it's omitted, matching M12's
existing behavior exactly.

M13 is additive infrastructure only - JarvisCore, GeminiSearchAnswerGenerator,
and tool-call construction are deliberately not wired to build/pass a
RequestRequirements yet (see app/providers/router.py's module docstring).
"""

from dataclasses import dataclass
from enum import Enum

from app.capability import Capability


class TaskType(str, Enum):
    """A coarse label for what kind of work a request is - informational/
    traceability only (surfaced in a RoutingDecision's `reason`). The
    routing signals ProviderRouter actually acts on are the boolean flags
    below and `latency_preference`, not this label by itself."""

    GENERAL = "general"
    REASONING = "reasoning"
    CODING = "coding"
    TOOL_USE = "tool_use"
    STRUCTURED_OUTPUT = "structured_output"
    SEARCH_SYNTHESIS = "search_synthesis"


class LatencyPreference(str, Enum):
    NORMAL = "normal"
    LOW = "low"


@dataclass(frozen=True)
class RequestRequirements:
    """What a request needs from a provider - never the request's own
    text/content, so this (and any RoutingDecision built from it) is
    always safe to log. Every field defaults to the least-demanding value,
    matching plain DIRECT conversation: an unclassified request has zero
    hard requirements and zero soft-score bonuses, so it routes exactly
    like plain M12 always has."""

    task_type: TaskType = TaskType.GENERAL
    reasoning: bool = False
    latency_preference: LatencyPreference = LatencyPreference.NORMAL
    tool_use: bool = False
    structured_output: bool = False
    long_context: bool = False
    search_synthesis: bool = False


# Deterministic keyword signals for a DIRECT-style message with no other
# signal available - small and fixed, the same discipline as
# app/search/router.py and app/capability.py's own patterns. Never an AI
# call, never fuzzy matching.
_CODING_KEYWORDS = (
    "code", "function", "bug", "debug", "compile", "refactor", "script",
    "class ", "def ", "stack trace", "traceback",
)
_REASONING_KEYWORDS = (
    "why", "explain", "reason", "prove", "step by step", "step-by-step",
    "analyze", "compare", "trade-off", "tradeoff",
)

# A fixed structural threshold, not derived from any provider's actual
# context window (no adapter in this project does token counting) - just a
# signal that a prompt has grown large enough that context capacity is
# worth preferring. Deliberately generous so ordinary conversations never
# trip it.
LONG_CONTEXT_CHAR_THRESHOLD = 6000


def classify_request(
    message: str,
    *,
    capability: Capability | None = None,
    context_char_count: int = 0,
) -> RequestRequirements:
    """Deterministically derive RequestRequirements from a message and
    whatever Core's CapabilityRouter has already decided, if anything -
    never an AI call, never randomness.

    `capability`, when given, is authoritative: SEARCH always means
    search_synthesis, TOOL always means tool_use - matching exactly how
    this project's existing SEARCH/TOOL handling already only ever uses
    Gemini for those (see app/search/answer.py, app/tools/decision.py).
    This function only makes that fact expressible as routing metadata; it
    does not change what anything does with it (nothing consumes it yet).

    `context_char_count` is a length the caller already computed (e.g.
    JarvisCore._format_context's output) - this function never inspects
    conversation history itself, only the number passed in.

    `message` is used only to derive `task_type` for a plain DIRECT-style
    request (light keyword matching for CODING/REASONING) and is never
    stored on the returned RequestRequirements.
    """
    long_context = context_char_count > LONG_CONTEXT_CHAR_THRESHOLD

    if capability is Capability.SEARCH:
        return RequestRequirements(
            task_type=TaskType.SEARCH_SYNTHESIS,
            search_synthesis=True,
            long_context=long_context,
        )
    if capability is Capability.TOOL:
        return RequestRequirements(
            task_type=TaskType.TOOL_USE,
            tool_use=True,
            long_context=long_context,
        )

    lowered = message.lower()
    if any(keyword in lowered for keyword in _CODING_KEYWORDS):
        task_type = TaskType.CODING
        reasoning = True
    elif any(keyword in lowered for keyword in _REASONING_KEYWORDS):
        task_type = TaskType.REASONING
        reasoning = True
    else:
        task_type = TaskType.GENERAL
        reasoning = False

    return RequestRequirements(
        task_type=task_type,
        reasoning=reasoning,
        long_context=long_context,
    )
