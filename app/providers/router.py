"""
ProviderRouter: the ONE place a request's provider order is decided - and the order is always the configured one.

Fixed-order rule (the user's decision): Jarvis tries providers in exactly the order of `settings.ai_provider_order`
(Groq -> OpenRouter -> Gemini -> Cloudflare -> FreeLLMAPI by default), for EVERY kind of request - a plain chat reply, a search
answer, context reasoning/planning, a tool call, a Job plan. Nothing here promotes a provider because it is thought to be better
at a task. Before this rule the router scored providers per request (a long prompt or a strict-JSON request put Gemini first,
a low-latency preference put Groq first), so the configured order was silently overridden on most paths.

What the router still does:
  * returns `ordered_names` - every entry, in the configured order, never reordered, never fewer;
  * records which entries are currently unavailable (disabled / cooling down) as `excluded` - information only: the manager
    re-checks each entry's real state itself as it iterates and simply skips those;
  * records, as ADVISORY data only, how each provider's declared capabilities (app/providers/capabilities.py) match the
    request: `scores` and `unvalidated` (providers this project has never validated for a strict-JSON / tool-call request).
    Advisory means exactly that: it is for logs and debugging and never changes the order. Whether a reply is acceptable is
    decided by the caller's own validator (see `call_structured` in app/providers/manager.py), and a reply that fails it
    sends the request on to the NEXT provider in the same fixed order.

Retries, cooldowns, error classification, fallback execution and AllProvidersFailedError stay entirely in
app/providers/manager.py - this module never calls a provider. `now` is supplied by the caller; there is no clock here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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

# The flags that mean "the reply must parse as strict JSON". A provider lacking the matching capability is NOT excluded and
# NOT moved: it is only listed in RoutingDecision.unvalidated, and its reply is checked by the caller's validator like any other.
_STRICT_OUTPUT_FLAGS = frozenset({"tool_use", "structured_output"})

_CAPABILITY_SCORE = 2
_LATENCY_SCORE = 1


@dataclass(frozen=True)
class ExcludedCandidate:
    """One provider that is currently unavailable, and why. `reason` is one of a small, fixed set of safe strings -
    never derived from a request's message or any credential."""

    name: str
    reason: str


@dataclass(frozen=True)
class RoutingDecision:
    """A safe-to-log record of one routing decision. Never holds an API key, token, header, request URL, or the request's
    own message text (RequestRequirements never stores the message)."""

    requirements: RequestRequirements
    #: Every entry, in the CONFIGURED order - the order the manager tries them in.
    ordered_names: list[str]
    #: The entries that are enabled and not cooling down, in the configured order.
    eligible: list[str]
    #: Disabled / cooling-down entries (information only; they keep their place in `ordered_names`).
    excluded: list[ExcludedCandidate]
    #: ADVISORY capability match per eligible provider. Never used to order anything.
    scores: dict[str, int]
    #: The first eligible entry in the configured order (the first entry of all if none is eligible; None with no entries).
    selected: str | None
    reason: str
    #: ADVISORY: eligible providers this project has not validated for the strict-JSON kind of request being made.
    unvalidated: list[str] = field(default_factory=list)


def _unavailable_reason(entry: "AIProviderEntry", now: float) -> str | None:
    if not entry.health.enabled:
        return "disabled"
    if entry.health.in_cooldown(now):
        return f"cooling down ({entry.health.cooldown_until - now:.0f}s left)"
    return None


def _advisory_score(entry: "AIProviderEntry", requirements: RequestRequirements) -> int:
    caps = capabilities_for(entry.name)
    score = sum(_CAPABILITY_SCORE for flag, capability in _CAPABILITY_FLAGS if getattr(requirements, flag) and capability in caps)
    if requirements.latency_preference is LatencyPreference.LOW and latency_tier_for(entry.name) is LatencyTier.LOW:
        score += _LATENCY_SCORE
    return score


def _is_unvalidated(entry: "AIProviderEntry", requirements: RequestRequirements) -> bool:
    caps = capabilities_for(entry.name)
    return any(
        getattr(requirements, flag) and capability not in caps
        for flag, capability in _CAPABILITY_FLAGS
        if flag in _STRICT_OUTPUT_FLAGS
    )


class ProviderRouter:
    """Stateless and deterministic - see the module docstring. The same arguments always give the same RoutingDecision."""

    def route(
        self,
        requirements: RequestRequirements,
        entries: list["AIProviderEntry"],
        *,
        now: float,
    ) -> RoutingDecision:
        ordered_names = [entry.name for entry in entries]  # the configured order, exactly
        eligible: list["AIProviderEntry"] = []
        excluded: list[ExcludedCandidate] = []
        for entry in entries:
            reason = _unavailable_reason(entry, now)
            if reason is None:
                eligible.append(entry)
            else:
                excluded.append(ExcludedCandidate(name=entry.name, reason=reason))

        if eligible:
            selected: str | None = eligible[0].name
            reason = f"{selected} selected: first available in the configured order"
        elif ordered_names:
            selected = ordered_names[0]
            reason = "no provider available - trying the configured order"
        else:
            selected = None
            reason = "no providers registered"

        return RoutingDecision(
            requirements=requirements,
            ordered_names=ordered_names,
            eligible=[entry.name for entry in eligible],
            excluded=excluded,
            scores={entry.name: _advisory_score(entry, requirements) for entry in eligible},
            selected=selected,
            reason=reason,
            unvalidated=[entry.name for entry in eligible if _is_unvalidated(entry, requirements)],
        )
