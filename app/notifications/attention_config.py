"""
Attention configuration from settings - the one place the environment reaches the
attention policy. A malformed setting falls back to the built-in default for that
one setting (a typo must never make Jarvis interrupt more, or go silent, by accident).

    JARVIS_ATTENTION_QUIET_HOURS            "22:00-07:00"  (empty: none)
    JARVIS_ATTENTION_SUPPRESSING_CONTEXTS   comma list: contexts that hold back everything short of urgent
    JARVIS_ATTENTION_URGENT_WINDOW_MINUTES  how close a deadline must be to count as time-critical
    JARVIS_ATTENTION_ENABLED_CHANNELS       channels the user allows at all (default: desktop)
    JARVIS_ATTENTION_URGENT_CHANNELS        channels allowed to carry an URGENT interruption (default: desktop)

Urgency never implies a channel: a channel (earbuds, a watch, glasses...) is an urgent
channel only if it is listed here - and it must also really exist on the router.
"""

from __future__ import annotations

from datetime import time, timedelta

from app.notifications.attention import (
    DEFAULT_FREE_CONTEXTS,
    DEFAULT_SUPPRESSING_CONTEXTS,
    DEFAULT_URGENT_WINDOW,
    AttentionConfig,
    ChannelKind,
    ChannelProfile,
    QuietHours,
)


def _names(raw: str) -> frozenset[str]:
    return frozenset(part.strip().lower() for part in str(raw).split(",") if part.strip())


def parse_quiet_hours(raw: str) -> QuietHours | None:
    try:
        start, end = str(raw).strip().split("-")
        (sh, sm), (eh, em) = (map(int, start.strip().split(":")), map(int, end.strip().split(":")))
        return QuietHours(time(sh, sm), time(eh, em))
    except (ValueError, TypeError):
        return None


def channel_profiles(enabled: frozenset[str], urgent: frozenset[str]) -> tuple[ChannelProfile, ...]:
    """One profile per anticipated kind. `available` stays False here: only the gate, from what
    is registered on the router, can make a channel available."""
    return tuple(
        ChannelProfile(
            name=kind.value,
            kind=kind,
            available=False,
            enabled=kind.value in enabled,
            notify_ok=True,
            urgent_ok=kind.value in urgent,
        )
        for kind in ChannelKind
    )


def attention_config_from_settings() -> AttentionConfig:
    from app.config import settings

    quiet = parse_quiet_hours(settings.attention_quiet_hours) if settings.attention_quiet_hours.strip() else None
    suppressing = _names(settings.attention_suppressing_contexts) or DEFAULT_SUPPRESSING_CONTEXTS
    try:
        minutes = float(settings.attention_urgent_window_minutes)
        window = timedelta(minutes=minutes) if 0 < minutes <= 24 * 60 else DEFAULT_URGENT_WINDOW
    except (TypeError, ValueError):
        window = DEFAULT_URGENT_WINDOW
    return AttentionConfig(
        quiet_hours=quiet,
        suppressing_contexts=suppressing,
        free_contexts=DEFAULT_FREE_CONTEXTS,
        urgent_window=window,
        channels=channel_profiles(_names(settings.attention_enabled_channels), _names(settings.attention_urgent_channels)),
    )
