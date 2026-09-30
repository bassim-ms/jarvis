"""
The notification layer: channel-neutral events, deterministic text, priority
routing, at-most-once delivery, and isolation of failing or hanging channels.

Nothing here touches a Job: the router is output only.
"""

import pathlib
import re
import threading
import time
from datetime import datetime, timezone

import pytest

import app.notifications.base as notifications_base
from app.notifications.base import (
    DEFAULT_USER,
    InMemoryNotificationLedger,
    MAX_KEY_CHARS,
    MAX_TEXT_CHARS,
    NotificationChannel,
    NotificationEvent,
    NotificationKind,
    NotificationLedger,
    NotificationPriority,
    NotificationRouter,
    SQLiteNotificationLedger,
)
from app.notifications.channels import DesktopToastChannel
from app.notifications.text import address_for, clean_text, humanize_remaining, sentence
from app.orchestration.sqlite_store import SQLiteJobStore
from app.reminders.notifier import Notifier

NOW = datetime(2026, 9, 29, 12, 0, tzinfo=timezone.utc)


def sync(work):
    work()


def make_event(
    key="k1", priority=NotificationPriority.NORMAL, text="Sir, hello.", kind=NotificationKind.WAITING, **kwargs
):
    return NotificationEvent(key=key, kind=kind, priority=priority, text=text, **kwargs)


class Recorder(NotificationChannel):
    def __init__(self, name="rec", min_priority=NotificationPriority.LOW):
        self.name = name
        self.min_priority = min_priority
        self.received: list[tuple[str, NotificationEvent]] = []

    def deliver(self, user, event):
        self.received.append((user, event))


class Exploding(NotificationChannel):
    name = "boom"

    def deliver(self, user, event):
        raise RuntimeError("channel is broken")


def router(*channels, ledger=None):
    return NotificationRouter(channels, ledger or InMemoryNotificationLedger(), dispatcher=sync, clock=lambda: NOW)


# --- events ------------------------------------------------------------------------------------------------------


def test_an_event_carries_key_kind_priority_text_and_routing_metadata():
    event = make_event(conversation_id="c1", job_id="j1", interaction_code="abc234")
    assert (event.key, event.kind, event.priority, event.text) == (
        "k1", NotificationKind.WAITING, NotificationPriority.NORMAL, "Sir, hello."
    )
    assert (event.conversation_id, event.job_id, event.interaction_code) == ("c1", "j1", "abc234")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"key": ""},
        {"key": "x" * (MAX_KEY_CHARS + 1)},
        {"key": None},
        {"text": ""},
        {"text": "   "},
        {"text": "x" * (MAX_TEXT_CHARS + 1)},
        {"text": "two\nlines"},
        {"text": "tab\there"},
        {"text": "escape\x1b[31m"},
        {"text": "zero​width"},
        {"priority": 20},
        {"kind": "waiting"},
    ],
)
def test_malformed_events_are_rejected(kwargs):
    with pytest.raises(ValueError):
        make_event(**kwargs)


def test_events_are_immutable():
    event = make_event()
    with pytest.raises(Exception):
        event.text = "changed"


def test_priorities_are_ordered_and_leave_room_for_a_future_critical_level():
    assert NotificationPriority.LOW < NotificationPriority.NORMAL < NotificationPriority.HIGH
    assert {p.name for p in NotificationPriority} == {"LOW", "NORMAL", "HIGH"}  # no CRITICAL yet
    values = sorted(int(p) for p in NotificationPriority)
    assert all(b - a >= 10 for a, b in zip(values, values[1:]))  # gaps: a level fits between/above


# --- text helpers ---------------------------------------------------------------------------------------------------


def test_the_form_of_address_is_only_ever_sir_or_boss_and_is_deterministic():
    seen = {address_for(f"key-{i}") for i in range(200)}
    assert seen == {"Sir", "Boss"}
    assert address_for("stable") == address_for("stable")


def test_clean_text_makes_untrusted_text_single_line_bounded_and_control_free():
    dirty = "Line one\n\tline\x1b[31m two​" + "z" * 500
    cleaned = clean_text(dirty, 60)
    assert "\n" not in cleaned and "\x1b" not in cleaned and "​" not in cleaned
    assert len(cleaned) <= 60 and cleaned.endswith("...")
    assert clean_text(None, 10) == ""


def test_sentence_only_adds_punctuation_when_missing():
    assert sentence("Which browser?") == "Which browser?"
    assert sentence("Send it") == "Send it."


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (30, "in about a minute"),
        (45 * 60, "in about 45 minutes"),
        (3600, "in about an hour"),
        (4 * 3600, "in about 4 hours"),
        (24 * 3600, "tomorrow"),
        (72 * 3600, "in about 3 days"),
        (-5, "in about a minute"),
    ],
)
def test_remaining_time_is_said_the_way_a_person_would_say_it(seconds, expected):
    assert humanize_remaining(seconds) == expected


# --- routing ---------------------------------------------------------------------------------------------------------


def test_an_event_reaches_every_accepting_channel_with_the_user_and_the_same_event():
    a, b = Recorder("a"), Recorder("b")
    result = router(a, b).notify(DEFAULT_USER, make_event())

    assert result.accepted and not result.duplicate and result.channels == ("a", "b")
    assert a.received == b.received == [(DEFAULT_USER, make_event())]


def test_a_channel_only_receives_events_at_or_above_its_minimum_priority():
    low_ok, normal_only, high_only = (
        Recorder("low"), Recorder("normal", NotificationPriority.NORMAL), Recorder("high", NotificationPriority.HIGH)
    )
    r = router(low_ok, normal_only, high_only)

    r.notify("owner", make_event("e-low", NotificationPriority.LOW))
    r.notify("owner", make_event("e-normal", NotificationPriority.NORMAL))
    r.notify("owner", make_event("e-high", NotificationPriority.HIGH))

    assert [e.key for _, e in low_ok.received] == ["e-low", "e-normal", "e-high"]
    assert [e.key for _, e in normal_only.received] == ["e-normal", "e-high"]
    assert [e.key for _, e in high_only.received] == ["e-high"]


def test_a_future_channel_plugs_in_without_changing_the_router():
    r = router()
    r.notify("owner", make_event("before"))
    late = Recorder("late")
    r.add_channel(late)
    r.notify("owner", make_event("after"))

    assert [e.key for _, e in late.received] == ["after"]


def test_the_same_logical_event_is_delivered_at_most_once():
    channel = Recorder()
    r = router(channel)

    first = r.notify("owner", make_event("dup"))
    second = r.notify("owner", make_event("dup", text="Boss, reworded."))
    third = r.notify("owner", make_event("dup"))

    assert first.accepted and second.duplicate and third.duplicate
    assert len(channel.received) == 1 and channel.received[0][1].text == "Sir, hello."


def test_different_events_are_each_delivered():
    channel = Recorder()
    r = router(channel)
    r.notify("owner", make_event("one"))
    r.notify("owner", make_event("two"))
    assert [e.key for _, e in channel.received] == ["one", "two"]


def test_a_failing_channel_never_stops_the_others_or_raises():
    good = Recorder("good")
    r = router(Exploding(), good)

    result = r.notify("owner", make_event())  # must not raise

    assert result.accepted and len(good.received) == 1
    entry = r.ledger.get("k1")
    assert entry.failed == ("boom",) and entry.delivered == ("good",)


def test_a_hung_channel_never_blocks_the_caller_or_other_channels():
    release = threading.Event()
    good = Recorder("good")

    class Hangs(NotificationChannel):
        name = "hang"

        def deliver(self, user, event):
            release.wait(10)

    r = NotificationRouter([Hangs(), good], InMemoryNotificationLedger())  # the real (threaded) dispatcher
    started = time.monotonic()
    result = r.notify("owner", make_event())
    elapsed = time.monotonic() - started
    try:
        assert result.accepted and elapsed < 2
        deadline = time.monotonic() + 2
        while not good.received and time.monotonic() < deadline:
            time.sleep(0.01)
        assert len(good.received) == 1  # the healthy channel still delivered
    finally:
        release.set()


def test_notify_never_raises_whatever_it_is_given_or_whatever_is_broken():
    r = router(Recorder())
    for user, event in [(None, make_event()), ("", make_event()), ("owner", "not an event"), ("owner", None)]:
        assert r.notify(user, event).accepted is False

    class BrokenLedger(NotificationLedger):
        def claim(self, event, now):
            raise RuntimeError("disk gone")

        def record_delivery(self, key, channel, ok):
            raise RuntimeError("disk gone")

        def get(self, key):
            raise RuntimeError("disk gone")

        def count(self, prefix):
            raise RuntimeError("disk gone")

    channel = Recorder()
    broken = router(channel, ledger=BrokenLedger())
    assert broken.notify("owner", make_event()).accepted is False
    assert channel.received == []  # cannot be de-duplicated, so nothing is sent
    assert broken.already_sent("k1") is False and broken.sent_count("k") == 0


def test_a_broken_dispatcher_is_recorded_not_raised():
    def bad_dispatch(work):
        raise RuntimeError("no threads")

    r = NotificationRouter([Recorder()], InMemoryNotificationLedger(), dispatcher=bad_dispatch, clock=lambda: NOW)
    assert r.notify("owner", make_event()).accepted
    assert r.ledger.get("k1").failed == ("rec",)


def test_a_channel_that_raises_from_accepts_is_isolated():
    class BadAccepts(Recorder):
        def accepts(self, event):
            raise RuntimeError("bug")

    good = Recorder("good")
    assert router(BadAccepts("bad"), good).notify("owner", make_event()).accepted
    assert len(good.received) == 1


# --- the ledger -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("make_ledger", ["memory", "sqlite"])
def test_ledger_claims_once_records_outcomes_and_counts_by_prefix(make_ledger, tmp_path):
    ledger = (
        InMemoryNotificationLedger()
        if make_ledger == "memory"
        else SQLiteNotificationLedger.for_job_store(SQLiteJobStore(str(tmp_path / "n.db")))
    )
    assert ledger.claim(make_event("interaction:1:reminder:1", job_id="j"), NOW) is True
    assert ledger.claim(make_event("interaction:1:reminder:1"), NOW) is False
    ledger.claim(make_event("interaction:1:reminder:2"), NOW)
    ledger.claim(make_event("interaction:2:reminder:1"), NOW)
    ledger.record_delivery("interaction:1:reminder:1", "desktop", True)
    ledger.record_delivery("interaction:1:reminder:1", "phone", False)
    ledger.record_delivery("missing", "desktop", True)  # unknown key: ignored

    entry = ledger.get("interaction:1:reminder:1")
    assert (entry.delivered, entry.failed, entry.job_id) == (("desktop",), ("phone",), "j")
    assert ledger.get("nope") is None
    assert ledger.count("interaction:1:reminder:") == 2
    assert ledger.count("interaction:") == 3
    assert ledger.count("interaction:%") == 0  # LIKE wildcards in a prefix are literal


def test_the_sqlite_ledger_survives_a_restart(tmp_path):
    path = str(tmp_path / "n.db")
    first = SQLiteNotificationLedger.for_job_store(SQLiteJobStore(path))
    assert first.claim(make_event("persisted"), NOW) is True

    reopened = SQLiteNotificationLedger.for_job_store(SQLiteJobStore(path))  # a new process, in effect

    assert reopened.claim(make_event("persisted"), NOW) is False
    channel = Recorder()
    assert NotificationRouter([channel], reopened, dispatcher=sync).notify("owner", make_event("persisted")).duplicate
    assert channel.received == []


def test_concurrent_claims_of_one_key_have_exactly_one_winner(tmp_path):
    ledger = SQLiteNotificationLedger.for_job_store(SQLiteJobStore(str(tmp_path / "n.db")))
    winners = []
    barrier = threading.Barrier(8)

    def race():
        barrier.wait()
        winners.append(ledger.claim(make_event("contended"), NOW))

    threads = [threading.Thread(target=race) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]

    assert winners.count(True) == 1


# --- the desktop channel ----------------------------------------------------------------------------------------------


def test_the_desktop_channel_shows_the_events_own_text():
    shown = []

    class Spy(Notifier):
        def notify(self, title, message):
            shown.append((title, message))

    channel = DesktopToastChannel(Spy())
    channel.deliver("owner", make_event(text="Boss, the timer is set."))

    assert shown == [("Jarvis", "Boss, the timer is set.")]


def test_the_desktop_channel_can_keep_low_priority_events_off_the_desktop():
    class Silent(Notifier):
        def notify(self, title, message):
            pass

    channel = DesktopToastChannel(Silent(), min_priority=NotificationPriority.NORMAL)
    assert not channel.accepts(make_event(priority=NotificationPriority.LOW))
    assert channel.accepts(make_event(priority=NotificationPriority.NORMAL))
    assert channel.accepts(make_event(priority=NotificationPriority.HIGH))


# --- output only ----------------------------------------------------------------------------------------------------------


def test_the_notification_package_can_only_output():
    """No tools, no Job/interaction access, no OS or network reach."""
    root = pathlib.Path(notifications_base.__file__).parent
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        imports = "\n".join(l for l in text.splitlines() if l.startswith(("import ", "from ")))
        for forbidden in ("app.tools", "app.orchestration", "app.automation", "subprocess", "requests", "shutil", "app.computer"):
            assert forbidden not in imports, (path.name, forbidden)
        assert not re.search(r"^import os\b|^from os\b", imports, re.MULTILINE), path.name
