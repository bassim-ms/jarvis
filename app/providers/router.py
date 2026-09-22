"""
ProviderRouter: deterministic hybrid routing for M13.

Decides which registered provider AIProviderManager
(app/providers/manager.py) should attempt FIRST for a request, given that
request's RequestRequirements (app/providers/requirements.py), each
provider's static capabilities (app/providers/capabilities.py), and the
SAME health/cooldown state M12 already tracks on each AIProviderEntry -
never a second health system. `now` is always supplied by the caller
(AIProviderManager's own clock); this module owns no clock of its own.

M13 is purely additive: AIProviderManager.generate_reply only consults
this router when a caller explicitly passes `requirements=` - every
existing call site (JarvisCore, GeminiSearchAnswerGenerator, every M12
test) never does, so nothing here changes any currently running behavior.
Wiring real request classification into JarvisCore/search/tool-call
construction is a deliberate, separate, later step - not part of M13.

Routing NEVER reduces how many providers AIProviderManager is willing to
attempt: RoutingDecision.ordered_names is always a permutation of every
entry passed in - eligible, best-scored-first, then excluded ones appended
afterward in their original order. AIProviderManager still independently
re-checks each entry's real enabled/cooldown state as it iterates (its own
existing, unchanged code) - this router's "excluded" list is
routing/debugging information, never an authority the manager trusts
blindly, and never a way to make AllProvidersFailedError fire with fewer
attempts than M12 alone would have made. If nothing is eligible (e.g.
every candidate cooling down, or a hard requirement nothing declares),
ordered_names degrades to exactly the input order - i.e. plain M12
behavior - rather than raising or dropping anything here.

Retries, cooldown *management* (setting it), error classification,
fallback execution, and AllProvidersFailedError all remain entirely M12's
responsibility (app/providers/manager.py) - this module only ever computes
an order and a safe-to-log decision record; it never calls generate_reply
on anything itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.providers.capabilities import (
    LatencyTier,
    ProviderCapability,
    capabilities_for,
    latency_tier_for,
)
from app.providers.requirements import LatencyPreference, RequestRequirements

if TYPE_CHECKING:
    from app.providers.manager import AIProviderEntry

# Requirement flag name -> the ProviderCapability it corresponds to.
_CAPABILITY_FLAGS: tuple[tuple[str, ProviderCapability], ...] = (
    ("reasoning", ProviderCapability.REASONING),
    ("tool_use", ProviderCapability.TOOL_USE),
    ("structured_output", ProviderCapability.STRUCTURED_OUTPUT),
    ("search_synthesis", ProviderCapability.SEARCH_SYNTHESIS),
    ("long_context", ProviderCapability.LONG_CONTEXT),
)

# Which of the flags above are HARD requirements (a provider missing the
# matching capability is excluded) rather than soft scoring preferences.
#
# Only tool_use and structured_output are hard gates. Both require a
# response that reliably parses as strict, schema-conforming JSON
# (GeminiCapabilityDecisionProvider/GeminiToolCallProvider validate this
# rigorously and reject anything that doesn't conform - see
# app/capability.py, app/tools/decision.py) - something this project has
# only ever engineered a prompt *and* a validator for against Gemini.
# Groq/OpenRouter/Cloudflare's adapters are architecturally identical
# plain-text completions (see capabilities.py's docstring) and could
# plausibly be prompted for JSON too, but nothing in this codebase has
# ever validated that they do so reliably - so a HARD requirement (used
# only where a caller genuinely can't tolerate an unparseable response)
# conservatively sticks to what's actually proven here.
#
# search_synthesis is deliberately NOT a hard gate (see capabilities.py's
# docstring for the full reasoning): turning already-retrieved search
# evidence into a grounded, cited answer needs no structured output at
# all - GeminiSearchAnswerGenerator's own citation extraction
# (app/search/answer.py:_extract_cited_sources) is plain Python string
# matching over free text, not JSON parsing - so it's architecturally
# indistinguishable from an ordinary GENERAL_CONVERSATION completion. It
# stays a SOFT preference (Gemini still gets a score bonus below) purely
# because this project has actually exercised and tuned that exact
# prompt/citation pairing against Gemini, not because any other provider
# is presumed unable to attempt it.
#
# reasoning/long_context stay soft-only too, since every provider can
# legitimately attempt a general/reasoning prompt, and this project does
# no token-budget enforcement that would make a long-context miss an
# outright failure.
_HARD_REQUIREMENT_FLAGS = frozenset({"tool_use", "structured_output"})

_CAPABILITY_SCORE = 2
_LATENCY_SCORE = 1


@dataclass(frozen=True)
class ExcludedCandidate:
    """One provider the router put at the back of the order, and why.
    `reason` is always one of a small, fixed set of safe, non-secret
    strings - never derived from a request's message or any credential."""

    name: str
    reason: str


@dataclass(frozen=True)
class RoutingDecision:
    """A safe-to-log record of one routing decision. Never holds an API
    key, token, header, request URL, or the request's own message text -
    RequestRequirements itself never stores the message (see
    requirements.py), so nothing here can leak it transitively either."""

    requirements: RequestRequirements
    ordered_names: list[str]
    eligible: list[str]
    excluded: list[ExcludedCandidate]
    scores: dict[str, int]
    selected: str | None
    reason: str


def _exclusion_reason(
    entry: "AIProviderEntry", requirements: RequestRequirements, now: float
) -> str | None:
    """None if `entry` is eligible; otherwise a short, safe reason it was
    excluded from primary consideration (still appended to the final
    order - see module docstring)."""
    if not entry.health.enabled:
        return "disabled"

    if entry.health.in_cooldown(now):
        remaining = entry.health.cooldown_until - now
        return f"cooling down ({remaining:.0f}s left)"

    caps = capabilities_for(entry.name)
    for flag_name, capability in _CAPABILITY_FLAGS:
        if flag_name not in _HARD_REQUIREMENT_FLAGS:
            continue
        if getattr(requirements, flag_name) and capability not in caps:
            return f"missing capability: {flag_name}"

    return None


def _score(entry: "AIProviderEntry", requirements: RequestRequirements) -> int:
    caps = capabilities_for(entry.name)
    score = 0
    for flag_name, capability in _CAPABILITY_FLAGS:
        if getattr(requirements, flag_name) and capability in caps:
            score += _CAPABILITY_SCORE

    if (
        requirements.latency_preference is LatencyPreference.LOW
        and latency_tier_for(entry.name) is LatencyTier.LOW
    ):
        score += _LATENCY_SCORE

    return score


class ProviderRouter:
    """Stateless, deterministic router - see module docstring. Every call
    to `route` with the same arguments (including `now`) returns an
    identical RoutingDecision; nothing here is randomized or
    time-varying beyond the caller-supplied `now`."""

    def route(
        self,
        requirements: RequestRequirements,
        entries: list["AIProviderEntry"],
        *,
        now: float,
    ) -> RoutingDecision:
        eligible_entries: list["AIProviderEntry"] = []
        excluded: list[ExcludedCandidate] = []

        for entry in entries:
            reason = _exclusion_reason(entry, requirements, now)
            if reason is None:
                eligible_entries.append(entry)
            else:
                excluded.append(ExcludedCandidate(name=entry.name, reason=reason))

        scores = {entry.name: _score(entry, requirements) for entry in eligible_entries}
        order_index = {entry.name: i for i, entry in enumerate(entries)}

        # Deterministic tie-break: equal score falls back to the entry's
        # position in the caller-supplied order (i.e. settings.ai_provider_order)
        # - never dict/set iteration order, never random.
        eligible_entries.sort(key=lambda e: (-scores[e.name], order_index[e.name]))

        ordered_names = [e.name for e in eligible_entries] + [
            e.name for e in excluded
        ]
        selected = ordered_names[0] if ordered_names else None

        if not eligible_entries and excluded:
            reason = "no eligible provider - falling back to configured order"
        elif selected is None:
            reason = "no providers registered"
        elif scores.get(selected, 0) > 0:
            reason = f"{selected} selected: routing score {scores[selected]}"
        else:
            reason = f"{selected} selected: first eligible in configured order"

        return RoutingDecision(
            requirements=requirements,
            ordered_names=ordered_names,
            eligible=[e.name for e in eligible_entries],
            excluded=excluded,
            scores=scores,
            selected=selected,
            reason=reason,
        )
