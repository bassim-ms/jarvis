"""
The attention policy: "should Jarvis interrupt the user right now, or should this wait?"

    EVENT (+ trusted structured metadata)
      -> CONTEXT (what the user is doing, quiet hours, which channels exist)
      -> ATTENTION POLICY            <- this module: pure, deterministic, no I/O, no AI
      -> ATTENTION DECISION          (SILENT | QUEUE | NOTIFY | URGENT, and which channels)
      -> NOTIFICATION ENGINE         (the existing NotificationRouter - delivery only)
      -> CHANNEL / DEVICE

The policy decides whether and how strongly to interrupt. It never delivers, never
stores, never resolves a confirmation, never grants a permission and never touches a
tool: an URGENT "approve this action" is a louder message, not an approval. (This
package may not import tools, orchestration or automation - a test pins it.)

OUTCOMES
  SILENT   nothing interrupts; the information stays stored (a LOW-importance event,
           an expired one, a duplicate).
  QUEUE    useful, but it can wait: kept for later presentation (the user is busy, it
           is quiet hours, or no channel could carry it right now).
  NOTIFY   important enough to interrupt now, on the allowed channels.
  URGENT   genuinely time-critical: may bypass ordinary suppression - but ONLY on a
           channel configured to carry urgent interruptions, and only when the urgency
           came from trusted code, not from a model or external text. With no such
           channel it is NOT sent anywhere else: it is held for escalation.

URGENCY IS NOT A CLAIM. An event is URGENT only if ALL of: a trusted source (SYSTEM or
USER - never MODEL or EXTERNAL) flagged it urgent; its importance is HIGH; and it is
time-critical (flagged so by that source, or its deadline is within the urgent window
and not yet past). Anything less is handled as its importance alone would be, and the
reason says why urgency was not accepted.

CONTEXT. Any string, matched against a small configured set. A context the config
knows to be suppressing (class, studying, meeting, sleeping, focused_work by default)
holds back everything short of URGENT. "free"/"normal" and any context the config does
not know ("unknown") follow the ordinary rules: not knowing what the user is doing is
NEVER a reason to interrupt harder - or to stay quiet.

CHANNELS are described, not assumed. A ChannelProfile says whether a channel actually
exists (`available`, filled in by the gate from the router), whether the user enabled
it (`enabled`), and - separately - whether it may carry an ordinary notification
(`notify_ok`) and an URGENT one (`urgent_ok`). Urgency never implies a channel: earbuds
(or a watch, or glasses) are only ever an emergency channel if configured so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum, IntEnum


class Importance(IntEnum):
    """How much the event matters (numerically aligned with NotificationPriority)."""

    LOW = 10
    NORMAL = 20
    HIGH = 30


class Attention(IntEnum):
    SILENT = 0
    QUEUE = 1
    NOTIFY = 2
    URGENT = 3


class SignalSource(Enum):
    """Who classified the event. Only SYSTEM (Jarvis's own trusted rules) and USER
    (the user's own configuration/request) may make an event URGENT."""

    SYSTEM = "system"
    USER = "user"
    MODEL = "model"
    EXTERNAL = "external"


TRUSTED_SOURCES = frozenset({SignalSource.SYSTEM, SignalSource.USER})


class ChannelKind(Enum):
    """The kinds of channel the design anticipates. Most do not exist yet; a kind
    is only a label until a real channel is registered under it."""

    DESKTOP = "desktop"
    IN_APP = "in_app"
    PHONE = "phone"
    SMARTWATCH = "smartwatch"
    GLASSES = "glasses"
    EARBUDS = "earbuds"
    EMAIL = "email"


@dataclass(frozen=True)
class ChannelProfile:
    """What the policy may know about one channel. `name` matches the router
    channel's own `name`. `available`: a real channel is registered (set by the
    gate, never by configuration). `enabled`: the user allows it at all.
    `notify_ok` / `urgent_ok`: what it may carry - independent of each other and of
    how urgent anything is."""

    name: str
    kind: ChannelKind
    available: bool = False
    enabled: bool = True
    notify_ok: bool = True
    urgent_ok: bool = False


@dataclass(frozen=True)
class QuietHours:
    """A daily window (local wall-clock times of the `now` it is asked about); it may
    wrap midnight (22:00-07:00). start == end means no window."""

    start: time
    end: time

    def active(self, now: datetime) -> bool:
        t = now.time().replace(tzinfo=None)
        if self.start == self.end:
            return False
        if self.start < self.end:
            return self.start <= t < self.end
        return t >= self.start or t < self.end


DEFAULT_SUPPRESSING_CONTEXTS = frozenset({"class", "studying", "meeting", "sleeping", "focused_work"})
DEFAULT_FREE_CONTEXTS = frozenset({"free", "normal"})
UNKNOWN_CONTEXT = "unknown"
DEFAULT_URGENT_WINDOW = timedelta(minutes=15)


@dataclass(frozen=True)
class AttentionConfig:
    """The whole configuration: small and extensible - a new activity is one more
    string in `suppressing_contexts`, not new code."""

    quiet_hours: QuietHours | None = None
    suppressing_contexts: frozenset[str] = DEFAULT_SUPPRESSING_CONTEXTS
    free_contexts: frozenset[str] = DEFAULT_FREE_CONTEXTS
    urgent_window: timedelta = DEFAULT_URGENT_WINDOW
    channels: tuple[ChannelProfile, ...] = ()

    def normalize_context(self, context: object) -> str:
        """A context name the config knows, else "unknown"."""
        if not isinstance(context, str):
            return UNKNOWN_CONTEXT
        name = "_".join(context.strip().lower().replace("-", " ").split())
        return name if name in self.suppressing_contexts | self.free_contexts else UNKNOWN_CONTEXT


@dataclass(frozen=True)
class AttentionSignal:
    """The structured classification of one event, supplied by trusted code (never
    inferred here, never an opinion of a model). `urgent` is a CLAIM that only counts
    when `source` is trusted, importance is HIGH and the event is time-critical (see
    the module docstring). `deadline` / `expires_at` are aware datetimes."""

    importance: Importance
    source: SignalSource = SignalSource.SYSTEM
    urgent: bool = False
    time_critical: bool = False
    deadline: datetime | None = None
    expires_at: datetime | None = None


@dataclass(frozen=True)
class AttentionInput:
    signal: AttentionSignal
    now: datetime
    context: str = UNKNOWN_CONTEXT
    suppressed: bool = False  # an explicit "do not disturb", from whatever source the caller has
    channels: tuple[ChannelProfile, ...] = ()
    already_notified: bool = False
    duplicate: bool = False


@dataclass(frozen=True)
class AttentionDecision:
    """`channels` are the channel NAMES the event may be delivered on (empty unless
    the level is NOTIFY or URGENT, and for an URGENT that no allowed channel could
    carry). `suppressed_by` names what would have held it back (and, for URGENT, was
    bypassed). `escalate`: an URGENT event with no allowed channel - it must be held
    and surfaced again, never sent somewhere it is not allowed."""

    level: Attention
    code: str
    reason: str
    channels: tuple[str, ...] = field(default_factory=tuple)
    suppressed_by: str | None = None
    escalate: bool = False


class AttentionPolicy:
    """See the module docstring. Stateless; the same input always gives the same decision."""

    def __init__(self, config: AttentionConfig | None = None):
        self.config = config or AttentionConfig()

    def decide(self, inp: AttentionInput) -> AttentionDecision:
        cfg, sig, now = self.config, inp.signal, inp.now

        if sig.expires_at is not None and now >= sig.expires_at:
            return AttentionDecision(Attention.SILENT, "expired", "it has expired")
        if inp.already_notified:
            return AttentionDecision(Attention.SILENT, "already_notified", "the user was already notified of this")
        if inp.duplicate:
            return AttentionDecision(Attention.SILENT, "duplicate", "this event was already seen")

        suppressed_by = self._suppressed_by(inp)
        urgent, urgency_note = self._urgent(sig, now)

        if urgent:
            allowed = self._eligible(inp.channels, urgent=True)
            if allowed:
                return AttentionDecision(
                    Attention.URGENT, "urgent", "urgent and time-critical", allowed, suppressed_by
                )
            return AttentionDecision(
                Attention.URGENT,
                "urgent_no_channel",
                "urgent, but no channel is allowed to carry an urgent notification - held for escalation",
                (),
                suppressed_by,
                escalate=True,
            )

        note = f" ({urgency_note})" if urgency_note else ""
        if sig.importance <= Importance.LOW:
            return AttentionDecision(Attention.SILENT, "low_importance", "low importance: stored, not interrupting" + note)
        if suppressed_by is not None:
            return AttentionDecision(
                Attention.QUEUE, "suppressed", f"held back by {suppressed_by}" + note, suppressed_by=suppressed_by
            )
        allowed = self._eligible(inp.channels, urgent=False)
        if not allowed:
            return AttentionDecision(Attention.QUEUE, "no_channel", "no enabled channel is available right now" + note)
        return AttentionDecision(Attention.NOTIFY, "notify", "worth telling the user now" + note, allowed)

    # -- the parts --------------------------------------------------------------------------------------------------

    def _suppressed_by(self, inp: AttentionInput) -> str | None:
        cfg = self.config
        if inp.suppressed:
            return "do_not_disturb"
        if cfg.quiet_hours is not None and cfg.quiet_hours.active(inp.now):
            return "quiet_hours"
        context = cfg.normalize_context(inp.context)
        if context in cfg.suppressing_contexts:
            return f"context:{context}"
        return None  # free, normal, or unknown: the ordinary rules

    def _urgent(self, sig: AttentionSignal, now: datetime) -> tuple[bool, str]:
        """(accepted, why not). Urgency needs a trusted source, HIGH importance and time-criticality."""
        if not sig.urgent:
            return False, ""
        if sig.source not in TRUSTED_SOURCES:
            return False, f"urgency from a {sig.source.value} source is not accepted"
        if sig.importance < Importance.HIGH:
            return False, "urgency needs high importance"
        near_deadline = (
            sig.deadline is not None and timedelta(0) <= sig.deadline - now <= self.config.urgent_window
        )
        if not (sig.time_critical or near_deadline):
            return False, "not time-critical"
        return True, ""

    @staticmethod
    def _eligible(channels: tuple[ChannelProfile, ...], *, urgent: bool) -> tuple[str, ...]:
        return tuple(
            c.name
            for c in channels
            if c.available and c.enabled and (c.urgent_ok if urgent else c.notify_ok)
        )
