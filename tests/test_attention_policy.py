"""
The attention policy: "should Jarvis interrupt the user right now, or should this wait?"

    EVENT -> CONTEXT -> AttentionPolicy -> AttentionDecision -> NotificationRouter -> CHANNEL

The policy is pure and is tested on its own (a table of inputs). The gate is tested against
the REAL router, ledger and SQLite queue with recorder channels - never a real device. Times
are fixed; the "user context" is simply a string handed in (no class detector exists yet).
"""

import dataclasses
import pathlib
import re
from datetime import datetime, time, timedelta, timezone

import pytest

import app.notifications.attention as attention_module
import app.notifications.attention_gate as gate_module
import app.notifications.attention_queue as queue_module
from app.notifications.attention import (
    Attention,
    AttentionConfig,
    AttentionDecision,
    AttentionInput,
    AttentionPolicy,
    AttentionSignal,
    ChannelKind,
    ChannelProfile,
    Importance,
    QuietHours,
    SignalSource,
)
from app.notifications.attention_config import (
    attention_config_from_settings,
    channel_profiles,
    parse_quiet_hours,
)
from app.notifications.attention_gate import AttentionGate, signal_from_event
from app.notifications.attention_queue import InMemoryAttentionQueue, SQLiteAttentionQueue
from app.notifications.base import (
    InMemoryNotificationLedger,
    NotificationChannel,
    NotificationEvent,
    NotificationKind,
    NotificationPriority,
    NotificationRouter,
)
from app.orchestration.sqlite_store import SQLiteJobStore

NOW = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)  # a Tuesday, midday
LATER = NOW + timedelta(hours=3)
TRUSTED = SignalSource.SYSTEM

DESKTOP = ChannelProfile("desktop", ChannelKind.DESKTOP, available=True, enabled=True, urgent_ok=True)
EARBUDS_ON = ChannelProfile("earbuds", ChannelKind.EARBUDS, available=True, enabled=True, urgent_ok=False)
EARBUDS_OFF = ChannelProfile("earbuds", ChannelKind.EARBUDS, available=True, enabled=False, urgent_ok=False)


def signal(importance=Importance.NORMAL, **kw):
    return AttentionSignal(importance=importance, source=kw.pop("source", TRUSTED), **kw)


def urgent(**kw):
    return signal(Importance.HIGH, urgent=True, time_critical=True, **kw)


def decide(sig=None, *, context="unknown", channels=(DESKTOP,), now=NOW, config=None, **kw):
    inp = AttentionInput(sig or signal(), now, context=context, channels=channels, **kw)
    return AttentionPolicy(config).decide(inp)


# --- the policy: importance x context ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("context", ["free", "normal", "FREE", " Normal "])
def test_normal_importance_in_a_free_context_notifies(context):
    d = decide(signal(Importance.NORMAL), context=context)
    assert d.level is Attention.NOTIFY and d.channels == ("desktop",) and d.suppressed_by is None


@pytest.mark.parametrize("context", ["class", "studying", "meeting", "sleeping", "focused_work", "Focused Work", "focused-work"])
def test_normal_importance_is_queued_in_a_suppressing_context(context):
    d = decide(signal(Importance.NORMAL), context=context)
    assert d.level is Attention.QUEUE and d.code == "suppressed" and d.channels == ()
    assert d.suppressed_by == f"context:{context.lower().replace(' ', '_').replace('-', '_')}"


def test_normal_importance_during_class_and_during_studying_is_queued_not_delivered():
    assert decide(context="class").level is Attention.QUEUE
    assert decide(context="studying").level is Attention.QUEUE


def test_high_importance_notifies_when_free():
    assert decide(signal(Importance.HIGH), context="free").level is Attention.NOTIFY


def test_high_importance_is_normally_queued_in_class():
    d = decide(signal(Importance.HIGH), context="class")
    assert d.level is Attention.QUEUE and d.suppressed_by == "context:class"


def test_an_unknown_context_falls_back_to_the_ordinary_rules_never_to_urgent_or_silence():
    for context in ("unknown", "", "hanging out", "   ", None, 5, "clas"):
        normal = decide(signal(Importance.NORMAL), context=context)
        high = decide(signal(Importance.HIGH), context=context)
        assert normal.level is Attention.NOTIFY and high.level is Attention.NOTIFY, context
        assert normal.suppressed_by is None
    # ...and it never makes anything urgent that was not
    assert decide(signal(Importance.HIGH), context="unknown").level is not Attention.URGENT
    assert decide(signal(Importance.HIGH), context="unknown") == decide(signal(Importance.HIGH), context="free")


def test_low_importance_never_interrupts_wherever_the_user_is():
    for context in ("free", "class", "unknown"):
        d = decide(signal(Importance.LOW), context=context)
        assert d.level is Attention.SILENT and d.code == "low_importance" and d.channels == ()


def test_an_expired_event_is_silent_and_a_duplicate_or_already_notified_one_does_not_interrupt_again():
    assert decide(signal(expires_at=NOW), context="free").code == "expired"
    assert decide(signal(expires_at=NOW + timedelta(seconds=1)), context="free").level is Attention.NOTIFY
    assert decide(context="free", already_notified=True).code == "already_notified"
    assert decide(context="free", duplicate=True).code == "duplicate"
    # not even an urgent one interrupts twice
    assert decide(urgent(), context="class", already_notified=True).level is Attention.SILENT
    assert decide(urgent(), context="class", duplicate=True).level is Attention.SILENT
    assert decide(urgent(expires_at=NOW - timedelta(minutes=1)), context="class").level is Attention.SILENT


def test_the_decision_is_deterministic_and_the_policy_holds_no_state():
    policy = AttentionPolicy()
    inp = AttentionInput(signal(Importance.HIGH), NOW, context="class", channels=(DESKTOP,))
    assert policy.decide(inp) == policy.decide(inp) == AttentionPolicy().decide(inp)


# --- the policy: quiet hours and do-not-disturb -----------------------------------------------------------------------------------------


def _quiet(start, end):
    return AttentionConfig(quiet_hours=QuietHours(start, end))


@pytest.mark.parametrize(
    "hour,minute,active",
    [(22, 0, True), (23, 30, True), (0, 0, True), (6, 59, True), (7, 0, False), (12, 0, False), (21, 59, False)],
)
def test_quiet_hours_may_wrap_midnight(hour, minute, active):
    quiet = QuietHours(time(22, 0), time(7, 0))
    assert quiet.active(NOW.replace(hour=hour, minute=minute)) is active


def test_quiet_hours_within_a_day_and_the_empty_window():
    assert QuietHours(time(13, 0), time(14, 0)).active(NOW.replace(hour=13, minute=30))
    assert not QuietHours(time(13, 0), time(14, 0)).active(NOW)
    assert not QuietHours(time(9, 0), time(9, 0)).active(NOW)


def test_quiet_hours_suppress_normal_and_high_notifications():
    config = _quiet(time(11, 0), time(13, 0))  # NOW is inside it
    for importance in (Importance.NORMAL, Importance.HIGH):
        d = decide(signal(importance), context="free", config=config)
        assert d.level is Attention.QUEUE and d.suppressed_by == "quiet_hours"
    assert decide(signal(), context="free", config=_quiet(time(13, 0), time(14, 0))).level is Attention.NOTIFY


def test_an_explicit_do_not_disturb_suppresses_like_any_other_suppression():
    d = decide(signal(), context="free", suppressed=True)
    assert d.level is Attention.QUEUE and d.suppressed_by == "do_not_disturb"


def test_a_new_suppressing_context_is_configuration_not_code():
    config = AttentionConfig(suppressing_contexts=frozenset({"class", "prayer"}))
    assert decide(context="prayer", config=config).level is Attention.QUEUE
    assert decide(context="meeting", config=config).level is Attention.NOTIFY  # not in THIS config: "unknown"


# --- the policy: urgency is not a claim ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("context", ["class", "studying", "sleeping", "free", "unknown"])
def test_a_genuinely_urgent_event_bypasses_suppression_on_an_allowed_channel(context):
    d = decide(urgent(), context=context)
    assert d.level is Attention.URGENT and d.channels == ("desktop",) and not d.escalate


def test_the_bypass_is_recorded_it_covers_quiet_hours_and_do_not_disturb_too():
    d = decide(urgent(), context="class", suppressed=True, config=_quiet(time(11, 0), time(13, 0)))
    assert d.level is Attention.URGENT and d.suppressed_by == "do_not_disturb"
    assert decide(urgent(), context="free", config=_quiet(time(11, 0), time(13, 0))).suppressed_by == "quiet_hours"


def test_urgency_can_come_from_a_deadline_that_is_close_and_not_past():
    soon = signal(Importance.HIGH, urgent=True, deadline=NOW + timedelta(minutes=10))
    edge = signal(Importance.HIGH, urgent=True, deadline=NOW + timedelta(minutes=15))
    assert decide(soon, context="class").level is Attention.URGENT
    assert decide(edge, context="class").level is Attention.URGENT
    for late in (
        signal(Importance.HIGH, urgent=True, deadline=NOW + timedelta(minutes=16)),  # not close yet
        signal(Importance.HIGH, urgent=True, deadline=NOW - timedelta(seconds=1)),  # already past
    ):
        d = decide(late, context="class")
        assert d.level is Attention.QUEUE and "not time-critical" in d.reason


@pytest.mark.parametrize("source", [SignalSource.MODEL, SignalSource.EXTERNAL])
def test_urgency_declared_by_a_model_or_external_text_is_never_accepted(source):
    d = decide(urgent(source=source), context="class")
    assert d.level is Attention.QUEUE and "not accepted" in d.reason
    d = decide(urgent(source=source), context="free")
    assert d.level is Attention.NOTIFY  # it is still important; it is just not urgent


@pytest.mark.parametrize("source", [SignalSource.SYSTEM, SignalSource.USER])
def test_urgency_from_trusted_code_or_the_user_is_accepted(source):
    assert decide(urgent(source=source), context="class").level is Attention.URGENT


def test_urgency_needs_high_importance_and_time_criticality_and_the_claim_itself():
    assert decide(signal(Importance.NORMAL, urgent=True, time_critical=True), context="class").level is Attention.QUEUE
    assert decide(signal(Importance.HIGH, urgent=True), context="class").level is Attention.QUEUE  # not time-critical
    # high + time-critical but nobody claimed urgent (even near a deadline): ordinary handling
    assert decide(signal(Importance.HIGH, time_critical=True), context="class").level is Attention.QUEUE
    assert decide(signal(Importance.HIGH, deadline=NOW + timedelta(minutes=1)), context="class").level is Attention.QUEUE
    assert decide(signal(Importance.LOW, urgent=True, time_critical=True), context="class").level is Attention.SILENT


def test_the_urgent_window_is_configurable():
    config = AttentionConfig(urgent_window=timedelta(minutes=60))
    far = signal(Importance.HIGH, urgent=True, deadline=NOW + timedelta(minutes=45))
    assert decide(far, context="class", config=config).level is Attention.URGENT
    assert decide(far, context="class").level is Attention.QUEUE


# --- the policy: channels are separate from urgency ---------------------------------------------------------------------------------------


def test_urgent_does_not_use_a_disabled_channel_it_uses_another_allowed_one():
    d = decide(urgent(), context="class", channels=(EARBUDS_OFF, DESKTOP))
    assert d.level is Attention.URGENT and d.channels == ("desktop",)


def test_earbuds_are_not_an_emergency_channel_unless_configured_to_be():
    assert decide(urgent(), channels=(EARBUDS_ON, DESKTOP)).channels == ("desktop",)  # enabled, but not urgent_ok
    on = dataclasses.replace(EARBUDS_ON, urgent_ok=True)
    assert decide(urgent(), channels=(on, DESKTOP)).channels == ("earbuds", "desktop")  # only when the user said so


def test_with_no_allowed_channel_an_urgent_event_is_held_for_escalation_never_sent_elsewhere():
    for channels in ((), (EARBUDS_OFF,), (EARBUDS_ON,), (dataclasses.replace(DESKTOP, enabled=False),)):
        d = decide(urgent(), context="class", channels=channels)
        assert d.level is Attention.URGENT and d.channels == () and d.escalate and d.code == "urgent_no_channel"


def test_a_channel_that_does_not_exist_is_never_eligible_however_it_is_configured():
    ghost = ChannelProfile("smartwatch", ChannelKind.SMARTWATCH, available=False, enabled=True, notify_ok=True, urgent_ok=True)
    assert decide(urgent(), channels=(ghost,)).channels == ()
    assert decide(signal(), context="free", channels=(ghost,)).level is Attention.QUEUE  # "no channel", not "delivered"


def test_a_notify_with_no_enabled_channel_waits_rather_than_pretending():
    d = decide(signal(), context="free", channels=(EARBUDS_OFF,))
    assert d.level is Attention.QUEUE and d.code == "no_channel"


def test_a_channel_may_be_allowed_to_notify_but_not_carry_urgent_and_the_reverse():
    notify_only = ChannelProfile("glasses", ChannelKind.GLASSES, available=True, enabled=True, notify_ok=True, urgent_ok=False)
    urgent_only = ChannelProfile("phone", ChannelKind.PHONE, available=True, enabled=True, notify_ok=False, urgent_ok=True)
    assert decide(signal(), context="free", channels=(notify_only, urgent_only)).channels == ("glasses",)
    assert decide(urgent(), channels=(notify_only, urgent_only)).channels == ("phone",)


# --- the gate on the real router ----------------------------------------------------------------------------------------------------------


class Recorder(NotificationChannel):
    def __init__(self, name, min_priority=NotificationPriority.LOW):
        self.name = name
        self.min_priority = min_priority
        self.delivered = []

    def deliver(self, user, event):
        self.delivered.append(event)


def event(key="k1", priority=NotificationPriority.NORMAL, text="Sir, a reminder.", **kw):
    return NotificationEvent(key=key, kind=NotificationKind.REMINDER, priority=priority, text=text, **kw)


class Rig:
    """The real router (synchronous dispatch, fixed clock) and gate; recorder channels."""

    def __init__(self, tmp_path=None, channels=("desktop",), profiles=(DESKTOP,), config=None, now=NOW, queue=None):
        self.now = now
        self.recorders = {name: Recorder(name) for name in channels}
        self.router = NotificationRouter(
            list(self.recorders.values()), InMemoryNotificationLedger(), dispatcher=lambda work: work(), clock=lambda: self.now
        )
        cfg = dataclasses.replace(config or AttentionConfig(), channels=tuple(profiles))
        self.queue = queue or InMemoryAttentionQueue()
        self.contexts = ["unknown"]
        self.gate = AttentionGate(
            self.router, AttentionPolicy(cfg), self.queue, context_provider=lambda: self.contexts[-1], clock=lambda: self.now
        )

    def submit(self, ev=None, sig=None, **kw):
        return self.gate.submit("owner", ev or event(), sig, **kw)

    def delivered(self, name="desktop"):
        return self.recorders[name].delivered


def test_a_normal_notification_in_a_free_context_is_handed_to_the_router_and_delivered():
    rig = Rig()

    outcome = rig.submit(context="free")

    assert outcome.decision.level is Attention.NOTIFY and outcome.handed_off_to == ("desktop",)
    assert [e.key for e in rig.delivered()] == ["k1"]
    assert rig.router.ledger.get("k1").delivered == ("desktop",)
    assert rig.queue.get("k1").status == "handed_off"


def test_a_normal_notification_during_class_is_queued_and_not_delivered():
    rig = Rig()

    outcome = rig.submit(context="class")

    assert outcome.decision.level is Attention.QUEUE and outcome.handed_off_to == () and outcome.stored
    assert rig.delivered() == [] and not rig.router.already_sent("k1")  # the router never even claimed it


def test_a_queued_notification_keeps_everything_and_is_not_lost():
    rig = Rig()
    ev = event("k1", NotificationPriority.HIGH, "Sir, the plan is ready.", job_id="job-1", conversation_id="c1", interaction_code="abc234")
    sig = signal(Importance.HIGH, expires_at=LATER, deadline=LATER)

    rig.submit(ev, sig, context="class")

    (item,) = rig.queue.waiting()
    assert item.event == ev  # the ORIGINAL event, whole
    assert item.signal == sig  # importance, urgency, expiry
    assert item.status == "queued" and item.level == "queue"
    assert "context:class" in item.reason
    assert item.created_at == NOW.isoformat() and item.signal.expires_at == LATER


def test_a_queued_notification_survives_a_restart_and_is_presented_later_when_the_user_is_free(tmp_path):
    jobs = SQLiteJobStore(str(tmp_path / "jobs.db"))
    rig = Rig(queue=SQLiteAttentionQueue.for_job_store(jobs))
    rig.submit(event("k1", text="Sir, remember this."), signal(expires_at=LATER), context="class")
    rig.submit(event("k2", text="Sir, and this."), context="meeting")
    assert rig.delivered() == []

    jobs2 = SQLiteJobStore(str(tmp_path / "jobs.db"))  # a restart: new connection, same file
    later = Rig(queue=SQLiteAttentionQueue.for_job_store(jobs2), now=NOW + timedelta(hours=1))
    assert [i.key for i in later.queue.waiting()] == ["k1", "k2"]
    assert later.queue.get("k1").event.text == "Sir, remember this." and later.queue.get("k1").signal.expires_at == LATER

    changed = later.gate.release_queued("owner", context="free")

    assert sorted(o.key for o in changed) == ["k1", "k2"] and all(o.handed_off_to == ("desktop",) for o in changed)
    assert sorted(e.key for e in later.delivered()) == ["k1", "k2"]
    assert later.queue.waiting() == [] and later.queue.get("k1").status == "handed_off"


def test_releasing_while_still_busy_delivers_nothing_and_loses_nothing():
    rig = Rig()
    rig.submit(context="class")

    assert rig.gate.release_queued("owner", context="studying") == []
    assert rig.delivered() == [] and [i.key for i in rig.queue.waiting()] == ["k1"]


def test_a_queued_notification_that_expired_before_the_user_was_free_is_expired_not_delivered():
    rig = Rig()
    rig.submit(event(), signal(expires_at=NOW + timedelta(hours=1)), context="class")
    rig.now = NOW + timedelta(hours=2)

    changed = rig.gate.release_queued("owner", context="free")

    assert [o.decision.code for o in changed] == ["expired"] and rig.delivered() == []
    assert rig.queue.get("k1").status == "expired" and rig.queue.waiting() == []
    assert rig.queue.get("k1").event.text == "Sir, a reminder."  # still on record


def test_the_context_comes_from_the_provider_and_an_explicit_context_overrides_it():
    rig = Rig()
    rig.contexts.append("class")

    assert rig.submit(event("a")).decision.level is Attention.QUEUE  # from the provider
    assert rig.submit(event("b"), context="free").decision.level is Attention.NOTIFY  # explicit wins
    rig.contexts.append("free")
    assert rig.gate.release_queued("owner")[0].key == "a"


def test_no_provider_or_a_failing_one_means_unknown_which_is_the_ordinary_rules():
    def broken():
        raise RuntimeError("bug")

    rig = Rig()
    rig.gate.context_provider = broken
    assert rig.submit(event("a")).decision.level is Attention.NOTIFY
    rig.gate.context_provider = None
    assert rig.submit(event("b")).decision.level is Attention.NOTIFY


def test_a_repeated_event_does_not_interrupt_again():
    rig = Rig()

    first = rig.submit(context="free")
    second = rig.submit(context="free")
    third = rig.submit(context="free")

    assert first.handed_off_to == ("desktop",)
    assert second.decision.level is Attention.SILENT and third.decision.code in ("duplicate", "already_notified")
    assert len(rig.delivered()) == 1 and len(rig.queue.all()) == 1


def test_a_repeated_suppressed_event_is_queued_once_and_delivered_once_later():
    rig = Rig()
    for _ in range(4):
        rig.submit(context="class")

    assert len(rig.queue.waiting()) == 1
    rig.gate.release_queued("owner", context="free")
    rig.gate.release_queued("owner", context="free")
    assert len(rig.delivered()) == 1


def test_an_event_the_router_already_claimed_elsewhere_is_not_delivered_again():
    rig = Rig()
    rig.router.notify("owner", event())  # an existing producer sent it directly

    outcome = rig.submit(context="free")

    assert outcome.decision.code == "already_notified" and len(rig.delivered()) == 1


def test_an_urgent_event_during_class_is_delivered_on_the_allowed_channel_only():
    rig = Rig(channels=("desktop", "earbuds"), profiles=(EARBUDS_ON, DESKTOP))

    outcome = rig.submit(event("u1", NotificationPriority.HIGH, "Sir, act now."), urgent(), context="class")

    assert outcome.decision.level is Attention.URGENT and outcome.handed_off_to == ("desktop",)
    assert rig.delivered("desktop") and rig.delivered("earbuds") == []
    assert outcome.decision.suppressed_by == "context:class"  # the bypass is on the record


def test_an_urgent_event_with_earbuds_disabled_never_uses_earbuds():
    rig = Rig(channels=("desktop", "earbuds"), profiles=(EARBUDS_OFF, DESKTOP))
    rig.submit(event("u1", NotificationPriority.HIGH), urgent(), context="class")
    assert rig.delivered("earbuds") == [] and len(rig.delivered("desktop")) == 1

    only_earbuds = Rig(channels=("earbuds",), profiles=(EARBUDS_OFF,))
    outcome = only_earbuds.submit(event("u2", NotificationPriority.HIGH), urgent(), context="class")
    assert outcome.handed_off_to == () and only_earbuds.delivered("earbuds") == []
    assert only_earbuds.queue.get("u2").status == "escalation"  # held for escalation, not lost, not sent


def test_an_urgent_event_with_only_a_non_urgent_earbud_channel_is_held_not_sent():
    rig = Rig(channels=("earbuds",), profiles=(EARBUDS_ON,))

    outcome = rig.submit(event("u1", NotificationPriority.HIGH), urgent(), context="free")

    assert outcome.handed_off_to == () and rig.delivered("earbuds") == []
    assert outcome.decision.escalate and rig.queue.get("u1").status == "escalation"
    assert not rig.router.already_sent("u1")  # nothing claimed: nothing was sent


def test_an_escalated_event_is_delivered_once_an_allowed_channel_exists():
    rig = Rig(channels=(), profiles=(DESKTOP,))
    rig.submit(event("u1", NotificationPriority.HIGH), urgent(), context="class")
    assert rig.queue.get("u1").status == "escalation"

    rig.recorders["desktop"] = Recorder("desktop")
    rig.router.add_channel(rig.recorders["desktop"])
    changed = rig.gate.release_queued("owner", context="class")

    assert [o.key for o in changed] == ["u1"] and len(rig.delivered()) == 1
    assert rig.queue.get("u1").status == "handed_off"


def test_a_channel_that_is_configured_but_not_registered_is_never_treated_as_delivered():
    watch = ChannelProfile("smartwatch", ChannelKind.SMARTWATCH, enabled=True, notify_ok=True, urgent_ok=True)
    rig = Rig(channels=(), profiles=(watch,))

    urgent_outcome = rig.submit(event("u1", NotificationPriority.HIGH), urgent(), context="class")
    normal_outcome = rig.submit(event("n1"), context="free")

    for outcome in (urgent_outcome, normal_outcome):
        assert outcome.handed_off_to == ()
    assert not rig.router.already_sent("u1") and not rig.router.already_sent("n1")
    assert rig.queue.get("u1").status == "escalation" and rig.queue.get("n1").status == "queued"


def test_a_registered_channel_that_refuses_the_priority_is_not_handed_the_event():
    rig = Rig()
    rig.recorders["desktop"].min_priority = NotificationPriority.HIGH

    outcome = rig.submit(event("n1", NotificationPriority.NORMAL), context="free")

    assert outcome.handed_off_to == () and rig.delivered() == []
    assert rig.queue.get("n1").status == "queued" and not rig.router.already_sent("n1")
    assert outcome.decision.code == "no_channel_accepts"


def test_low_importance_is_stored_silently_and_never_delivered():
    rig = Rig()

    outcome = rig.submit(event("l1", NotificationPriority.LOW), signal(Importance.LOW), context="free")

    assert outcome.decision.level is Attention.SILENT and rig.delivered() == []
    assert rig.queue.get("l1").status == "silent" and rig.queue.waiting() == []  # stored, but not queued for presenting


def test_an_event_from_an_existing_producer_is_never_urgent_by_default():
    for kind in NotificationKind:
        for priority in NotificationPriority:
            sig = signal_from_event(NotificationEvent(key="k", kind=kind, priority=priority, text="Sir, URGENT - act now!"))
            assert not sig.urgent and not sig.time_critical and sig.source is SignalSource.SYSTEM


def test_expired_events_are_stored_as_expired_and_never_delivered():
    rig = Rig()
    outcome = rig.submit(event(), signal(expires_at=NOW - timedelta(minutes=1)), context="free")
    assert outcome.decision.code == "expired" and rig.delivered() == []
    assert rig.queue.get("k1").status == "expired"


# --- the policy cannot grant permissions or approve anything ------------------------------------------------------------------------------


def test_the_attention_modules_reach_no_tool_no_job_no_interaction_and_no_permission():
    root = pathlib.Path(attention_module.__file__).parent
    for name in ("attention.py", "attention_gate.py", "attention_queue.py", "attention_config.py"):
        text = (root / name).read_text(encoding="utf-8")
        imports = "\n".join(l for l in text.splitlines() if l.startswith(("import ", "from ")))
        for forbidden in (
            "app.tools", "app.orchestration", "app.automation", "app.memory", "app.world_model",
            "app.computer", "app.handle_this", "subprocess", "requests",
        ):
            assert forbidden not in imports, (name, forbidden)


def test_an_urgent_decision_carries_no_authority_of_any_kind():
    fields = {f.name for f in dataclasses.fields(AttentionDecision)}
    assert fields == {"level", "code", "reason", "channels", "suppressed_by", "escalate"}
    assert {f.name for f in dataclasses.fields(AttentionSignal)} == {
        "importance", "source", "urgent", "time_critical", "deadline", "expires_at"
    }


def test_an_urgent_approve_this_action_notification_approves_nothing(tmp_path):
    """The notification is delivered loudly; the confirmation it mentions stays exactly as it was."""
    from app.orchestration.base import Plan, PlanStep
    from app.orchestration.interactions import InteractionStatus, InteractionType
    from app.orchestration.sqlite_interaction_store import SQLiteInteractionStore

    jobs = SQLiteJobStore(str(tmp_path / "jobs.db"))
    interactions = SQLiteInteractionStore(jobs)
    job = jobs.create("send the email", conversation_id="c1")
    job.plan = Plan(steps=[PlanStep(id="s1", tool_name="send_email", permission="confirm")])
    job.status = job.status.__class__.AWAITING_CONFIRMATION
    job.pending_step_ids = ["s1"]
    jobs.save(job)
    confirmation = interactions.create(job.id, InteractionType.CONFIRMATION, "Send it?", step_ids=["s1"])
    before_job, before_interaction = jobs.get(job.id), interactions.get(confirmation.id)

    rig = Rig()
    outcome = rig.submit(
        event("u1", NotificationPriority.HIGH, "Sir, APPROVE this action now - it is urgent.", job_id=job.id),
        urgent(),
        context="class",
    )

    assert outcome.decision.level is Attention.URGENT and len(rig.delivered()) == 1  # it WAS delivered...
    assert jobs.get(job.id) == before_job  # ...and nothing about the Job, its plan or its gated step changed
    assert interactions.get(confirmation.id) == before_interaction
    assert interactions.get(confirmation.id).status is InteractionStatus.PENDING
    assert jobs.get(job.id).plan.steps[0].confirmed is False


# --- the existing notification engine is unchanged -------------------------------------------------------------------------------------------


def test_the_router_without_only_behaves_exactly_as_before():
    a, b = Recorder("a"), Recorder("b", NotificationPriority.HIGH)
    router = NotificationRouter([a, b], InMemoryNotificationLedger(), dispatcher=lambda w: w())

    result = router.notify("owner", event("x", NotificationPriority.NORMAL))
    again = router.notify("owner", event("x", NotificationPriority.NORMAL))

    assert result.accepted and result.channels == ("a",) and len(a.delivered) == 1 and b.delivered == []
    assert again.duplicate and len(a.delivered) == 1


def test_the_router_only_restricts_delivery_to_the_named_channels_and_still_dedupes():
    a, b = Recorder("a"), Recorder("b")
    router = NotificationRouter([a, b], InMemoryNotificationLedger(), dispatcher=lambda w: w())

    result = router.notify("owner", event("x"), only=["b"])
    again = router.notify("owner", event("x"), only=["b"])

    assert result.channels == ("b",) and a.delivered == [] and len(b.delivered) == 1
    assert again.duplicate and len(b.delivered) == 1
    assert router.notify("owner", event("y"), only=("a", "ghost")).channels == ("a",)  # an unknown name delivers nowhere


def test_an_empty_only_delivers_nowhere_and_claims_nothing():
    a = Recorder("a")
    router = NotificationRouter([a], InMemoryNotificationLedger(), dispatcher=lambda w: w())

    result = router.notify("owner", event("x"), only=[])

    assert not result.accepted and not result.duplicate and a.delivered == [] and not router.already_sent("x")
    assert router.notify("owner", event("x")).accepted  # the key was never burned


# --- configuration -------------------------------------------------------------------------------------------------------------------------


def test_the_default_configuration_makes_desktop_the_only_enabled_and_urgent_channel(monkeypatch):
    from app.config import settings

    for name, value in (
        ("attention_quiet_hours", ""), ("attention_suppressing_contexts", "class,studying,meeting,sleeping,focused_work"),
        ("attention_urgent_window_minutes", 15.0), ("attention_enabled_channels", "desktop"), ("attention_urgent_channels", "desktop"),
    ):
        monkeypatch.setattr(settings, name, value)

    config = attention_config_from_settings()

    profiles = {p.name: p for p in config.channels}
    assert set(profiles) == {k.value for k in ChannelKind}
    assert profiles["desktop"].enabled and profiles["desktop"].urgent_ok
    for name in ("earbuds", "smartwatch", "glasses", "phone", "email", "in_app"):
        assert not profiles[name].enabled and not profiles[name].urgent_ok and not profiles[name].available
    assert config.quiet_hours is None and config.urgent_window == timedelta(minutes=15)
    assert config.suppressing_contexts == frozenset({"class", "studying", "meeting", "sleeping", "focused_work"})


def test_earbuds_can_be_enabled_without_becoming_an_emergency_channel():
    profiles = {p.name: p for p in channel_profiles(frozenset({"desktop", "earbuds"}), frozenset({"desktop"}))}
    assert profiles["earbuds"].enabled and not profiles["earbuds"].urgent_ok
    profiles = {p.name: p for p in channel_profiles(frozenset({"desktop", "earbuds"}), frozenset({"desktop", "earbuds"}))}
    assert profiles["earbuds"].urgent_ok  # only because the user said so


@pytest.mark.parametrize(
    "raw,expected",
    [("22:00-07:00", QuietHours(time(22, 0), time(7, 0))), (" 08:30 - 09:15 ", QuietHours(time(8, 30), time(9, 15))), ("", None), ("late", None), ("25:00-07:00", None), ("22:00", None)],
)
def test_quiet_hours_are_parsed_and_a_malformed_setting_means_none(raw, expected):
    assert parse_quiet_hours(raw) == expected


def test_a_malformed_setting_falls_back_to_the_default_for_that_setting_only(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "attention_quiet_hours", "nonsense")
    monkeypatch.setattr(settings, "attention_urgent_window_minutes", "soon")
    monkeypatch.setattr(settings, "attention_suppressing_contexts", " , ")
    config = attention_config_from_settings()
    assert config.quiet_hours is None and config.urgent_window == timedelta(minutes=15)
    assert "class" in config.suppressing_contexts

    monkeypatch.setattr(settings, "attention_urgent_window_minutes", 99999)
    assert attention_config_from_settings().urgent_window == timedelta(minutes=15)
    monkeypatch.setattr(settings, "attention_quiet_hours", "23:00-06:30")
    assert attention_config_from_settings().quiet_hours == QuietHours(time(23, 0), time(6, 30))



# --- boundaries pinned by mutation testing -------------------------------------------------------------------------------------------------


def test_a_same_day_quiet_window_includes_its_start_and_excludes_its_end():
    window = QuietHours(time(13, 0), time(14, 0))
    assert window.active(NOW.replace(hour=13, minute=0)) and window.active(NOW.replace(hour=13, minute=59))
    assert not window.active(NOW.replace(hour=14, minute=0)) and not window.active(NOW.replace(hour=12, minute=59))


def test_a_configured_channel_that_is_not_registered_is_reported_as_no_channel_not_as_a_try():
    watch = ChannelProfile("smartwatch", ChannelKind.SMARTWATCH, enabled=True, notify_ok=True, urgent_ok=True)
    rig = Rig(channels=(), profiles=(watch,))

    urgent_outcome = rig.submit(event("u1", NotificationPriority.HIGH), urgent(), context="class")
    normal_outcome = rig.submit(event("n1"), context="free")

    assert urgent_outcome.decision.code == "urgent_no_channel" and urgent_outcome.decision.channels == ()
    assert normal_outcome.decision.code == "no_channel" and normal_outcome.decision.channels == ()


def test_an_urgent_event_that_its_allowed_channel_refuses_is_held_for_escalation():
    rig = Rig()
    rig.recorders["desktop"].min_priority = NotificationPriority.HIGH

    outcome = rig.submit(event("u1", NotificationPriority.NORMAL), urgent(), context="class")  # the event itself is NORMAL

    assert outcome.decision.level is Attention.URGENT and outcome.decision.escalate
    assert outcome.decision.code == "no_channel_accepts" and outcome.handed_off_to == ()
    assert rig.queue.get("u1").status == "escalation" and rig.delivered() == []


@pytest.mark.parametrize("make_queue", [InMemoryAttentionQueue, lambda: SQLiteAttentionQueue.for_job_store(SQLiteJobStore(":memory:"))])
def test_the_queue_stores_one_row_per_key_and_says_so(make_queue):
    from app.notifications.attention_queue import QueuedItem

    queue = make_queue()
    item = QueuedItem(event(), signal(), "queued", "queue", "why", (), NOW.isoformat(), NOW.isoformat())

    assert queue.add(item) is True
    assert queue.add(dataclasses.replace(item, reason="a different reason")) is False
    assert queue.get("k1").reason == "why" and len(queue.all()) == 1
    queue.update("k1", status="handed_off", level="notify", reason="sent", channels=("desktop",), now=LATER)
    assert queue.get("k1").status == "handed_off" and queue.get("k1").channels == ("desktop",) and queue.waiting() == []
    assert queue.get("missing") is None
