"""
AttentionGate: the thin front door that puts the attention policy in front of the
existing NotificationRouter.

    gate.submit(user, event, signal, context=...)
        -> AttentionPolicy.decide(...)           (pure; see attention.py)
        -> SILENT / QUEUE : the event is stored in the AttentionQueue, nothing interrupts
        -> NOTIFY / URGENT: handed to the EXISTING router, restricted to the channels the
                            policy allowed (`router.notify(..., only=...)`)

The router stays the notification engine: it still claims the key (at-most-once
delivery), applies each channel's own `min_priority`, and delivers on its own
threads. The gate never delivers anything itself and never decides delivery
details; it only decides WHETHER and ON WHICH of the router's channels, then hands
over. A channel that is not registered on the router is never eligible, so nothing
is ever reported as delivered through a device that does not exist.

Held-back events are not lost: `release_queued` re-decides every waiting event under
the CURRENT context and time - delivering those now allowed, expiring those whose
expiry passed, and leaving the rest waiting. Nothing schedules it: a caller (a later
slice) chooses when.

EXISTING PRODUCERS. `GatedNotificationRouter` is a drop-in for the router the Job engine,
the interaction lifecycle and the reminder scheduler already call (`notify`, `already_sent`,
`sent_count`): they keep emitting the same structured events and learn nothing about
attention. What an event MEANS to attention is decided here, in one table, from its
structured `kind` (see `IMPORTANCE_BY_KIND`) - and it is URGENT only if trusted code marked
it `time_critical` (the text is never read).

Nothing here resolves a confirmation, calls a tool or changes a Job: delivering a
louder notification does not approve anything.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from app.notifications.attention import (
    UNKNOWN_CONTEXT,
    Attention,
    AttentionConfig,
    AttentionDecision,
    AttentionInput,
    AttentionPolicy,
    AttentionSignal,
    ChannelProfile,
    Importance,
    SignalSource,
)
from app.notifications.attention_queue import AttentionQueue, InMemoryAttentionQueue, QueuedItem
from app.notifications.base import DeliveryResult, NotificationEvent, NotificationKind, NotificationRouter


@dataclass(frozen=True)
class AttentionOutcome:
    """What the gate did. `handed_off_to`: the channel names the router was given the
    event for (delivery is asynchronous and recorded in the router's ledger); empty
    when nothing was sent. `stored`: the event is now (or already was) in the queue."""

    key: str
    decision: AttentionDecision
    handed_off_to: tuple[str, ...] = ()
    stored: bool = False


#: What each kind of existing event means to attention. Producers never see this table:
#: a Job that failed or is waiting on the user matters more than one that merely finished.
IMPORTANCE_BY_KIND: dict[NotificationKind, Importance] = {
    NotificationKind.COMPLETED: Importance.NORMAL,
    NotificationKind.FAILED: Importance.HIGH,
    NotificationKind.WAITING: Importance.HIGH,  # needs the user's input or confirmation
    NotificationKind.EXPIRY_WARNING: Importance.HIGH,  # something is about to lapse
    NotificationKind.EXPIRED: Importance.NORMAL,
    NotificationKind.REMINDER: Importance.NORMAL,
}


def signal_from_event(event: NotificationEvent) -> AttentionSignal:
    """The signal for an event whose producer supplied none. Importance comes from the event's
    structured KIND (never from its text); the source is Jarvis's own rules; and it is urgent
    only if trusted code set `time_critical` - which nothing does by default. So an existing
    producer's event can never become URGENT by accident, and text saying "URGENT" changes nothing."""
    importance = IMPORTANCE_BY_KIND.get(event.kind) or Importance(int(event.priority))
    flagged = bool(event.time_critical)
    return AttentionSignal(
        importance=importance, source=SignalSource.SYSTEM, urgent=flagged, time_critical=flagged
    )


class AttentionGate:
    def __init__(
        self,
        router: NotificationRouter,
        policy: AttentionPolicy | None = None,
        queue: AttentionQueue | None = None,
        *,
        context_provider: Callable[[], str] | None = None,
        relevance_check: Callable[[NotificationEvent], bool] | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        self.router = router
        self.policy = policy or AttentionPolicy()
        self.queue = queue or InMemoryAttentionQueue()
        # Zero-arg callable returning the user's current context name ("class", "free"...) from
        # whatever source exists. None / a failure / an unrecognised name all mean "unknown",
        # which follows the ordinary rules (never "urgent", never "silent").
        self.context_provider = context_provider
        # "Is this still worth telling the user?" (e.g. the question it announces is still open). Asked
        # only when a HELD event is reconsidered: False expires it (it is never delivered late); a check
        # that fails is treated as "yes" (unsure: keep it, do not drop it).
        self.relevance_check = relevance_check
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    # -- inputs ----------------------------------------------------------------------------------------------

    def _profiles(self) -> tuple[ChannelProfile, ...]:
        """The configured profiles, each marked available only if a channel of that name is
        actually registered on the router."""
        registered = {c.name for c in self.router.channels}
        return tuple(dataclasses.replace(p, available=p.name in registered) for p in self.policy.config.channels)

    def _context(self, context: str | None) -> str:
        if context is not None:
            return context
        if self.context_provider is None:
            return UNKNOWN_CONTEXT
        try:
            return self.context_provider()
        except Exception:  # noqa: BLE001 - a context source that fails means "unknown"
            return UNKNOWN_CONTEXT

    def _input(
        self, event: NotificationEvent, signal: AttentionSignal, now: datetime, context: str, suppressed: bool,
        *, duplicate: bool,
    ) -> AttentionInput:
        return AttentionInput(
            signal=signal,
            now=now,
            context=context,
            suppressed=suppressed,
            channels=self._profiles(),
            already_notified=self.router.already_sent(event.key),
            duplicate=duplicate,
        )

    # -- the front door -----------------------------------------------------------------------------------------

    def submit(
        self,
        user: str,
        event: NotificationEvent,
        signal: AttentionSignal | None = None,
        *,
        context: str | None = None,
        suppressed: bool = False,
    ) -> AttentionOutcome:
        """Decide and act on one new event. Never raises for a policy outcome."""
        signal = signal or signal_from_event(event)
        now = self._clock()
        existing = self.queue.get(event.key)
        decision = self.policy.decide(
            self._input(event, signal, now, self._context(context), suppressed, duplicate=existing is not None)
        )

        if existing is not None:  # the same logical event again: nothing new, nothing interrupts
            return AttentionOutcome(event.key, decision, stored=True)
        return self._act(user, event, signal, decision, now, is_new=True)

    def release_queued(
        self, user: str, *, context: str | None = None, suppressed: bool = False
    ) -> list[AttentionOutcome]:
        """Re-decide everything still waiting under the current context and time. Delivers what
        is now allowed, expires what has expired, leaves the rest queued. Returns the outcomes
        that CHANGED something (a hand-off or an expiry)."""
        now = self._clock()
        current = self._context(context)
        changed: list[AttentionOutcome] = []
        for item in self.queue.waiting():
            if not self._still_relevant(item.event):
                gone = AttentionDecision(Attention.SILENT, "no_longer_relevant", "it is no longer relevant")
                self._store(item.event, item.signal, "expired", gone, (), now, is_new=False)
                changed.append(AttentionOutcome(item.key, gone, stored=True))
                continue
            decision = self.policy.decide(self._input(item.event, item.signal, now, current, suppressed, duplicate=False))
            if (
                decision.level is Attention.QUEUE
                and item.status == "queued"
                and item.reason == decision.reason
            ):
                continue  # still held for the same reason: nothing to do, nothing to write
            outcome = self._act(user, item.event, item.signal, decision, now, is_new=False, existing=item)
            if outcome.handed_off_to or decision.code in ("expired", "already_notified"):
                changed.append(outcome)
        return changed

    def _still_relevant(self, event: NotificationEvent) -> bool:
        if self.relevance_check is None:
            return True
        try:
            return bool(self.relevance_check(event))
        except Exception:  # noqa: BLE001 - when unsure, do not deliver something that may be stale... nor drop it
            return True

    def take_escalations(self, limit: int = 3) -> list[QueuedItem]:
        """URGENT events that could not be delivered anywhere (no allowed, available channel), for the
        chat to show the user in its next reply - the one surface the user is demonstrably looking at.
        Each returned item is marked `presented_in_chat` so it is shown once; a stale one is expired
        instead of shown. Nothing is delivered through any channel here and no restriction is lifted:
        the user is simply told, in a reply to their own message."""
        now = self._clock()
        shown: list[QueuedItem] = []
        for item in self.queue.waiting():
            if item.status != "escalation":
                continue
            if not self._still_relevant(item.event):
                gone = AttentionDecision(Attention.SILENT, "no_longer_relevant", "it is no longer relevant")
                self._store(item.event, item.signal, "expired", gone, (), now, is_new=False)
                continue
            if len(shown) >= limit:
                break
            done = AttentionDecision(Attention.URGENT, "shown_in_chat", "shown in the chat: no channel could carry it")
            self._store(item.event, item.signal, "presented_in_chat", done, (), now, is_new=False)
            shown.append(item)
        return shown

    # -- acting on a decision ---------------------------------------------------------------------------------------

    def _act(
        self,
        user: str,
        event: NotificationEvent,
        signal: AttentionSignal,
        decision: AttentionDecision,
        now: datetime,
        *,
        is_new: bool,
        existing: QueuedItem | None = None,
    ) -> AttentionOutcome:
        level = decision.level

        if level in (Attention.NOTIFY, Attention.URGENT) and decision.channels:
            names = self._channels_that_accept(event, decision.channels)
            if names:
                result = self.router.notify(user, event, only=names)
                if result.accepted:
                    self._store(event, signal, "handed_off", decision, result.channels, now, is_new)
                    return AttentionOutcome(event.key, decision, tuple(result.channels), stored=True)
                if result.duplicate:  # the router already claimed this key: it was delivered before
                    self._store(event, signal, "handed_off", decision, (), now, is_new)
                    return AttentionOutcome(event.key, decision, stored=True)
            # nothing could carry it: it waits (an URGENT one is held for escalation), never lost
            held = dataclasses.replace(
                decision,
                level=Attention.URGENT if level is Attention.URGENT else Attention.QUEUE,
                code="no_channel_accepts",
                reason="no allowed channel would accept it right now",
                channels=(),
                escalate=level is Attention.URGENT,
            )
            self._store(event, signal, "escalation" if held.escalate else "queued", held, (), now, is_new)
            return AttentionOutcome(event.key, held, stored=True)

        if level is Attention.URGENT:  # allowed to interrupt, but no permitted channel exists: held
            self._store(event, signal, "escalation", decision, (), now, is_new)
            return AttentionOutcome(event.key, decision, stored=True)
        if level is Attention.QUEUE:
            self._store(event, signal, "queued", decision, (), now, is_new)
            return AttentionOutcome(event.key, decision, stored=True)

        # SILENT: stored where that is the point (low importance); expiry / already-notified update a waiting item
        if decision.code == "expired":
            self._store(event, signal, "expired", decision, (), now, is_new)
        elif decision.code == "already_notified":
            self._store(event, signal, "handed_off", decision, (), now, is_new)
        else:
            self._store(event, signal, "silent", decision, (), now, is_new)
        return AttentionOutcome(event.key, decision, stored=True)

    def _channels_that_accept(self, event: NotificationEvent, allowed: tuple[str, ...]) -> tuple[str, ...]:
        """Only channels that are really registered on the router AND accept this event
        (their own `min_priority`) - so a hand-off is only ever to a channel that can take it."""
        return tuple(c.name for c in self.router.channels if c.name in allowed and c.accepts(event))

    def _store(
        self,
        event: NotificationEvent,
        signal: AttentionSignal,
        status: str,
        decision: AttentionDecision,
        channels: tuple[str, ...],
        now: datetime,
        is_new: bool,
    ) -> None:
        try:
            if is_new:
                stamp = now.isoformat()
                self.queue.add(
                    QueuedItem(
                        event=event, signal=signal, status=status, level=decision.level.name.lower(),
                        reason=decision.reason, channels=tuple(channels), created_at=stamp, updated_at=stamp,
                    )
                )
            else:
                self.queue.update(
                    event.key, status=status, level=decision.level.name.lower(), reason=decision.reason,
                    channels=tuple(channels), now=now,
                )
        except Exception:  # noqa: BLE001 - bookkeeping never breaks delivery
            pass


class GatedNotificationRouter:
    """The router the existing producers call, with attention in front of it.

    Same three methods they already use on a NotificationRouter (`notify`, `already_sent`,
    `sent_count`) - so a Job engine, an interaction lifecycle or a reminder scheduler is given
    THIS instead of the router and changes nowhere else. `notify` submits the event to the gate
    (importance from its kind, urgency only from its `time_critical` metadata, context from the
    gate's provider). Its `accepted` means "taken responsibility for": handed to a channel OR held
    in the queue; `channels` names only what was really handed to the router. Producers therefore
    never see - or duplicate - an attention decision."""

    def __init__(self, gate: AttentionGate):
        self.gate = gate

    @property
    def channels(self):
        return self.gate.router.channels

    def add_channel(self, channel) -> None:
        self.gate.router.add_channel(channel)

    def notify(self, user: str, event: NotificationEvent) -> DeliveryResult:
        try:
            outcome = self.gate.submit(user, event)
        except Exception:  # noqa: BLE001 - like the router: notifying never raises into a producer
            return DeliveryResult(accepted=False)
        duplicate = outcome.decision.code in ("duplicate", "already_notified")
        return DeliveryResult(
            accepted=bool(outcome.handed_off_to) or (outcome.stored and not duplicate),
            duplicate=duplicate,
            channels=tuple(outcome.handed_off_to),
        )

    def already_sent(self, key: str) -> bool:
        """Sent OR held: either way the producer must not send it again."""
        try:
            return self.gate.router.already_sent(key) or self.gate.queue.get(key) is not None
        except Exception:  # noqa: BLE001
            return False

    def sent_count(self, prefix: str) -> int:
        try:
            return max(self.gate.router.sent_count(prefix), self.gate.queue.count(prefix))
        except Exception:  # noqa: BLE001
            return 0
