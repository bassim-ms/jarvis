"""
In-flight `type_text`: typing is SENT in bounded chunks and the admission is rechecked before every chunk, so a revoked or expired grant, or a switch turned off, stops the
typing at the next chunk boundary instead of after up to 30 seconds.

What is promised (and what is not):
  * ONE gated action: the whole text is one grant entry and one rate-limit slot, matched against the grant by the exact text, MAX_TEXT_LENGTH unchanged. Rechecking consumes nothing.
  * After the grant is revoked (`revoke` / `revoke_all`) or ended by a mismatch, after its lifetime has passed, or after the switch is turned off, NO FURTHER CHUNK is sent.
  * It is observed at the NEXT recheck. Characters already sent to the operating system cannot be recalled, and the chunk in progress completes.
  * A failure part-way (an OS error) ends the typing too; the admission stays spent, as for any admitted action.

Everything runs on the fake input library (tests/conftest.py): "sent to the OS" below means "recorded by the fake", and the hooks fire from inside the fake's `write`, i.e. exactly
between two chunks. No real input, no real waiting.
"""

import ast
import logging
import math
import pathlib
import threading

import pytest

import fake_input_library as fake
from app.computer.base import MAX_TEXT_LENGTH, ComputerControlError
from app.computer.gate import AdmissionTicket, InputGate
from app.computer.windows_backend import TYPING_CHUNK_CHARACTERS, TYPING_INTERVAL_SECONDS, WindowsComputerBackend
from grant_support import admit_with, authorize, without_grant_ids

APP_DIR = pathlib.Path(__file__).resolve().parent.parent / "app"
SIZE = TYPING_CHUNK_CHARACTERS


def text_of(length: int) -> str:
    """Distinct-looking text, so a chunk that is dropped, repeated or reordered cannot go unnoticed."""
    return "".join(chr(ord("a") + (i * 7 + i // 26) % 26) for i in range(length))


TEXT = text_of(2 * SIZE + 20)  # three chunks: SIZE, SIZE, 20


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class OsFailure(BaseException):
    """Not an Exception: even this must leave the admission closed."""


@pytest.fixture(autouse=True)
def isolation_holds():
    assert fake.isolation_report()["all_ok"], "the OS-input isolation is not in place; refusing to run"


def rig(*, text=TEXT, ttl=60.0, limit=10, extra_steps=()):
    clock, state = Clock(), {"on": True, "limit": limit}
    gate = InputGate(lambda: state["on"], lambda: state["limit"], clock=clock)
    grant = authorize(gate, ("type_text", text), *extra_steps, ttl_seconds=ttl)
    return gate, grant, WindowsComputerBackend(gate=gate, authority=grant), clock, state


def chunks_sent():
    return [call[1][0] for call in fake.FAKE.calls if call[0] == "write"]


def hook_writes(monkeypatch, after: dict):
    """Replace the fake's `write` with one that records the chunk like the real fake and THEN runs `after[n]()` once the n-th chunk has been 'sent'."""
    count = {"n": 0}

    def write(chunk, **kwargs):
        count["n"] += 1
        fake.FAKE.calls.append(("write", (chunk,), kwargs))
        if count["n"] in after:
            after[count["n"]]()

    monkeypatch.setattr(fake.FAKE, "write", write)


def assert_spent_and_closed(gate, grant, *, entries_left=0):
    assert len(gate._events) == 1  # one slot, whatever happened to the typing
    assert gate.unspent(grant) == entries_left
    assert gate._open == set()  # and the admission was closed


# ================================================ typing in chunks: the same text, one action ===============================================================


@pytest.mark.parametrize("length", [1, SIZE - 1, SIZE, SIZE + 1, 2 * SIZE, 2 * SIZE + 1, MAX_TEXT_LENGTH])
def test_the_text_is_sent_in_order_in_chunks_of_at_most_the_chunk_size(length):
    text = text_of(length)
    gate, grant, backend, _, _ = rig(text=text)

    backend.type_text(text)

    sent = chunks_sent()
    assert "".join(sent) == text and len(sent) == math.ceil(length / SIZE)
    assert all(0 < len(chunk) <= SIZE for chunk in sent)
    assert all(call[2] == {"interval": TYPING_INTERVAL_SECONDS} for call in fake.FAKE.calls)  # the pacing is unchanged
    assert_spent_and_closed(gate, grant)


def test_the_whole_text_is_one_grant_entry_and_one_slot_and_the_rechecks_consume_nothing(monkeypatch):
    text = text_of(MAX_TEXT_LENGTH)
    gate, grant, backend, _, _ = rig(text=text, extra_steps=[("press_key", ("enter",))])
    seen = []
    hook_writes(monkeypatch, {n: (lambda: seen.append((len(gate._events), gate.unspent(grant)))) for n in range(1, 21)})

    backend.type_text(text)

    assert len(chunks_sent()) == 20 and seen == [(1, 1)] * 20  # twenty chunks later: still one slot used, and the NEXT entry of the grant untouched
    assert gate.unspent(grant) == 1 and gate._open == set()


def test_the_chunk_size_bounds_the_time_between_rechecks():
    assert TYPING_CHUNK_CHARACTERS * TYPING_INTERVAL_SECONDS <= 2.0  # a stop is seen within about two seconds of typing, not thirty
    assert math.ceil(MAX_TEXT_LENGTH / TYPING_CHUNK_CHARACTERS) <= 40


def test_max_text_length_is_unchanged_and_an_over_long_or_empty_text_is_refused_before_the_gate_and_costs_nothing():
    assert MAX_TEXT_LENGTH == 1000
    gate, grant, backend, _, _ = rig(text="x" * 1001)  # even a grant that names it cannot make it valid

    for bad in ("x" * 1001, ""):
        with pytest.raises(ComputerControlError):
            backend.type_text(bad)

    assert chunks_sent() == [] and len(gate._events) == 0 and gate.unspent(grant) == 1 and gate._open == set()


def test_the_grant_still_matches_the_exact_text_and_a_different_one_is_refused_and_ends_the_grant():
    gate, grant, backend, _, _ = rig()

    with pytest.raises(ComputerControlError, match="approved authorization"):
        backend.type_text(TEXT[:-1] + "!")

    assert chunks_sent() == [] and len(gate._events) == 0 and gate.unspent(grant) == 0 and gate._open == set()


def test_without_a_grant_nothing_is_typed_and_no_admission_is_left_open():
    gate = InputGate(lambda: True, lambda: 10)

    with pytest.raises(ComputerControlError, match="approved authorization"):
        WindowsComputerBackend(gate=gate).type_text(TEXT)

    assert chunks_sent() == [] and gate._open == set()


# ================================================ revocation: no later chunk ================================================================================


@pytest.mark.parametrize("revoke_after", [1, 2])
@pytest.mark.parametrize("more_entries", [False, True], ids=["last-entry", "more-entries"])
@pytest.mark.parametrize("how", ["revoke", "revoke_all"])
def test_a_grant_revoked_mid_typing_sends_no_further_chunk(monkeypatch, how, more_entries, revoke_after):
    gate, grant, backend, _, _ = rig(extra_steps=[("press_key", ("enter",))] if more_entries else [])
    revoke = (lambda: gate.revoke(grant)) if how == "revoke" else gate.revoke_all
    hook_writes(monkeypatch, {revoke_after: revoke})

    with pytest.raises(ComputerControlError, match="part-way"):
        backend.type_text(TEXT)

    assert "".join(chunks_sent()) == TEXT[: revoke_after * SIZE]  # the chunk in progress completed; nothing after it was sent
    assert_spent_and_closed(gate, grant, entries_left=0)  # (a revoked grant has no usable entry left, and the slot stays spent)


def test_revocation_is_observed_before_the_first_chunk_too_and_then_nothing_was_typed():
    gate, grant, backend, _, _ = rig()
    real_admit = gate.admit

    def admit_then_revoke(*args, **kwargs):
        ticket = real_admit(*args, **kwargs)
        gate.revoke(grant)  # between the admission and the first chunk
        return ticket

    gate.admit = admit_then_revoke

    with pytest.raises(ComputerControlError, match="approved authorization"):
        backend.type_text(TEXT)

    assert chunks_sent() == [] and len(gate._events) == 1 and gate._open == set()


def test_after_a_stopped_typing_the_grant_cannot_be_used_again(monkeypatch):
    gate, grant, backend, _, _ = rig(extra_steps=[("press_key", ("enter",))])
    hook_writes(monkeypatch, {1: lambda: gate.revoke(grant)})
    with pytest.raises(ComputerControlError, match="part-way"):
        backend.type_text(TEXT)
    sent_before = len(chunks_sent())

    with pytest.raises(ComputerControlError, match="approved authorization"):
        backend.type_text(TEXT)

    assert len(chunks_sent()) == sent_before == 1


def test_a_different_grant_being_revoked_does_not_stop_this_typing(monkeypatch):
    gate, grant, backend, _, _ = rig()
    other = authorize(gate, "scroll")
    hook_writes(monkeypatch, {1: lambda: gate.revoke(other)})

    backend.type_text(TEXT)

    assert "".join(chunks_sent()) == TEXT
    assert_spent_and_closed(gate, grant)


def test_a_mismatch_on_the_same_grant_ends_it_and_stops_the_typing_that_is_still_going(monkeypatch):
    gate, grant, backend, _, _ = rig(extra_steps=[("press_key", ("enter",))])

    def wrong_next_action():
        with pytest.raises(ComputerControlError):
            admit_with(gate, grant, "scroll")  # the grant's next entry is press_key: the whole grant ends

    hook_writes(monkeypatch, {1: wrong_next_action})

    with pytest.raises(ComputerControlError, match="part-way"):
        backend.type_text(TEXT)

    assert "".join(chunks_sent()) == TEXT[:SIZE] and gate._open == set()


# ================================================ the switch ================================================================================================


@pytest.mark.parametrize("off_after", [1, 2])
def test_switching_input_off_mid_typing_sends_no_further_chunk_and_ends_every_grant(monkeypatch, off_after):
    gate, grant, backend, _, state = rig()
    other = authorize(gate, "scroll")
    hook_writes(monkeypatch, {off_after: lambda: state.update(on=False)})

    with pytest.raises(ComputerControlError, match="part-way"):
        backend.type_text(TEXT)

    assert "".join(chunks_sent()) == TEXT[: off_after * SIZE]
    assert gate.unspent(other) == 0  # as at admission: a switch seen off ends every grant
    assert_spent_and_closed(gate, grant)


def test_a_switch_seen_off_by_another_admission_ends_the_typing_even_if_it_is_back_on_by_the_next_chunk(monkeypatch):
    gate, grant, backend, _, state = rig()
    other = authorize(gate, "scroll")

    def off_noticed_then_on_again():
        state["on"] = False
        with pytest.raises(ComputerControlError, match="switched off"):
            admit_with(gate, other, "scroll")  # some other admission sees the switch off: as always, no grant outlives that ...
        state["on"] = True  # ... and flipping it back on does not bring anything back

    hook_writes(monkeypatch, {1: off_noticed_then_on_again})

    with pytest.raises(ComputerControlError, match="part-way"):
        backend.type_text(TEXT)

    assert "".join(chunks_sent()) == TEXT[:SIZE]
    assert gate.unspent(other) == 0 and gate._open == set()


def test_the_admission_is_rechecked_before_every_chunk_the_first_included():
    gate, grant, backend, _, state = rig()
    original_check = gate._recheck
    seen = []

    def recheck(ticket):
        seen.append(state["on"])
        return original_check(ticket)

    gate._recheck = recheck
    fake.FAKE.calls.clear()
    backend.type_text(TEXT)

    assert seen == [True, True, True] and "".join(chunks_sent()) == TEXT  # one recheck before EVERY chunk, the first included


def test_an_unreadable_switch_fails_closed_between_chunks(monkeypatch):
    gate, grant, backend, _, _ = rig()

    def unreadable():
        raise RuntimeError("config unavailable")

    hook_writes(monkeypatch, {1: lambda: setattr(gate, "_enabled", unreadable)})

    with pytest.raises(ComputerControlError, match="part-way"):
        backend.type_text(TEXT)

    assert "".join(chunks_sent()) == TEXT[:SIZE]


# ================================================ expiry ====================================================================================================


@pytest.mark.parametrize("expires_after", [1, 2])
def test_a_grant_that_expires_mid_typing_sends_no_further_chunk(monkeypatch, expires_after):
    gate, grant, backend, clock, _ = rig(ttl=10.0)
    hook_writes(monkeypatch, {expires_after: lambda: setattr(clock, "now", clock.now + 10.0)})  # exactly the end of its lifetime

    with pytest.raises(ComputerControlError, match="part-way"):
        backend.type_text(TEXT)

    assert "".join(chunks_sent()) == TEXT[: expires_after * SIZE]
    assert_spent_and_closed(gate, grant)


def test_one_instant_before_expiry_the_typing_carries_on_to_the_end(monkeypatch):
    gate, grant, backend, clock, _ = rig(ttl=10.0)
    hook_writes(monkeypatch, {1: lambda: setattr(clock, "now", clock.now + 9.999)})

    backend.type_text(TEXT)

    assert "".join(chunks_sent()) == TEXT


def test_a_clock_that_fails_between_chunks_stops_the_typing(monkeypatch):
    gate, grant, backend, clock, _ = rig()

    def broken():
        raise RuntimeError("clock unavailable")

    hook_writes(monkeypatch, {1: lambda: setattr(gate, "_clock", broken)})

    with pytest.raises(RuntimeError, match="clock unavailable"):
        backend.type_text(TEXT)

    assert "".join(chunks_sent()) == TEXT[:SIZE] and gate._open == set()


# ================================================ failure part-way ==========================================================================================


@pytest.mark.parametrize("failure", [RuntimeError("the input library crashed"), ComputerControlError("the OS refused"), OsFailure("interrupted")], ids=["crash", "control-error", "base-exception"])
@pytest.mark.parametrize("fail_on", [1, 2])
def test_a_failure_part_way_sends_no_further_chunk_and_leaves_the_admission_spent_and_closed(monkeypatch, failure, fail_on):
    gate, grant, backend, _, _ = rig()
    sent = []

    def write(chunk, **kwargs):
        if len(sent) + 1 == fail_on:
            raise failure
        sent.append(chunk)
        fake.FAKE.calls.append(("write", (chunk,), kwargs))

    monkeypatch.setattr(fake.FAKE, "write", write)

    with pytest.raises(type(failure)):
        backend.type_text(TEXT)

    assert "".join(sent) == TEXT[: (fail_on - 1) * SIZE]  # nothing after the failing chunk
    assert_spent_and_closed(gate, grant)
    with pytest.raises(ComputerControlError, match="approved authorization"):
        backend.type_text(TEXT)  # a failed typing is not a free retry: the entry is gone


def test_a_failure_after_a_revocation_changes_nothing_the_typing_already_stopped(monkeypatch):
    gate, grant, backend, _, _ = rig()
    hook_writes(monkeypatch, {1: lambda: gate.revoke(grant)})

    with pytest.raises(ComputerControlError):
        backend.type_text(TEXT)
    fake.FAKE.calls.clear()
    monkeypatch.setattr(fake.FAKE, "write", lambda chunk, **kw: (_ for _ in ()).throw(RuntimeError("must never be called")))
    with pytest.raises(ComputerControlError):
        backend.type_text(TEXT)

    assert chunks_sent() == []


# ================================================ what must NOT stop the typing =============================================================================


def test_the_action_being_the_last_entry_of_its_grant_or_filling_the_rate_limit_does_not_stop_it():
    gate, grant, backend, _, _ = rig(limit=1)  # one slot: this very action fills the window; the grant has one entry: this very action spends it

    backend.type_text(TEXT)

    assert "".join(chunks_sent()) == TEXT
    assert_spent_and_closed(gate, grant)


def test_lowering_the_limit_mid_typing_does_not_stop_it_one_action_is_one_slot(monkeypatch):
    gate, grant, backend, _, state = rig()
    hook_writes(monkeypatch, {1: lambda: state.update(limit=0)})

    backend.type_text(TEXT)

    assert "".join(chunks_sent()) == TEXT


# ================================================ the audit trail ===========================================================================================


def test_a_stopped_typing_leaves_one_reasoned_warning_and_no_text_count_or_length(monkeypatch, caplog):
    secret = "TOP-SECRET-" + text_of(2 * SIZE)
    gate, grant, backend, _, _ = rig(text=secret)
    hook_writes(monkeypatch, {1: lambda: gate.revoke(grant)})
    caplog.set_level(logging.DEBUG)

    with pytest.raises(ComputerControlError):
        backend.type_text(secret)
    text = without_grant_ids("\n".join(f"{r.getMessage()} | {r.args!r} | {r.exc_text or ''}" for r in caplog.records))

    assert "computer_input refused action=type_text reason=revoked stage=continuation" in [r.getMessage() for r in caplog.records]
    assert "TOP-SECRET" not in text and secret[:20] not in text
    assert not any(str(number) in text for number in (len(secret), SIZE, len(secret) - SIZE))  # no length, chunk size or remainder anywhere


# ================================================ concurrency ===============================================================================================


def test_a_revocation_from_another_thread_stops_the_typing_at_the_next_chunk(monkeypatch):
    gate, grant, backend, _, _ = rig()
    first_chunk_sent, revoked = threading.Event(), threading.Event()

    def write(chunk, **kwargs):
        fake.FAKE.calls.append(("write", (chunk,), kwargs))
        if len(chunks_sent()) == 1:
            first_chunk_sent.set()
            assert revoked.wait(10), "the revoking thread never ran"

    monkeypatch.setattr(fake.FAKE, "write", write)
    outcome = []

    def typing():
        try:
            backend.type_text(TEXT)
            outcome.append("finished")
        except ComputerControlError:
            outcome.append("stopped")

    worker = threading.Thread(target=typing)
    worker.start()
    assert first_chunk_sent.wait(10)
    gate.revoke(grant)  # from this thread, while the first chunk is 'being typed'
    revoked.set()
    worker.join(10)

    assert not worker.is_alive() and outcome == ["stopped"]
    assert "".join(chunks_sent()) == TEXT[:SIZE]


# ================================================ the ticket itself =========================================================================================


def test_admit_returns_a_ticket_only_when_asked_and_a_ticket_admits_nothing():
    gate = InputGate(lambda: True, lambda: 10)
    grant = authorize(gate, "scroll", "scroll")
    from app.computer.authority import canonical_input

    assert gate.admit("scroll", authority=grant, canonical=canonical_input("scroll", 5)) is None
    ticket = gate.admit("scroll", authority=grant, canonical=canonical_input("scroll", 5), keep_open=True)

    assert isinstance(ticket, AdmissionTicket) and len(gate._events) == 2
    for _ in range(50):
        ticket.still_authorized()
    assert len(gate._events) == 2  # fifty rechecks, nothing more spent or admitted
    with pytest.raises(ComputerControlError):
        admit_with(gate, grant, "scroll")  # and the grant, spent by the two real admissions, admits no third


def test_a_closed_ticket_is_refused_and_closing_twice_is_harmless():
    gate = InputGate(lambda: True, lambda: 10)
    grant = authorize(gate, "scroll")
    from app.computer.authority import canonical_input

    ticket = gate.admit("scroll", authority=grant, canonical=canonical_input("scroll", 5), keep_open=True)
    ticket.close()
    ticket.close()

    with pytest.raises(ComputerControlError):
        ticket.still_authorized()
    assert gate._open == set()


def test_a_ticket_is_only_good_at_the_gate_that_issued_it():
    first, second = InputGate(lambda: True, lambda: 10), InputGate(lambda: True, lambda: 10)
    grant = authorize(first, "scroll")
    from app.computer.authority import canonical_input

    ticket = first.admit("scroll", authority=grant, canonical=canonical_input("scroll", 5), keep_open=True)

    with pytest.raises(ComputerControlError):
        second._recheck(ticket)
    with pytest.raises(ComputerControlError):
        second._recheck(object())
    ticket.still_authorized()


def test_a_ticket_past_its_grants_lifetime_is_forgotten_and_reads_as_expired():
    clock = Clock()
    gate = InputGate(lambda: True, lambda: 10, clock=clock)
    grant = authorize(gate, "scroll", ttl_seconds=5.0)
    from app.computer.authority import canonical_input

    ticket = gate.admit("scroll", authority=grant, canonical=canonical_input("scroll", 5), keep_open=True)
    clock.now += 5.0
    gate.remaining()  # any read-only question prunes

    assert gate._open == set()
    with pytest.raises(ComputerControlError):
        ticket.still_authorized()


def test_revoke_still_reports_whether_the_grant_was_usable():
    gate = InputGate(lambda: True, lambda: 10)
    live, spent = authorize(gate, "scroll", "scroll"), authorize(gate, "scroll")
    from app.computer.authority import canonical_input

    gate.admit("scroll", authority=spent, canonical=canonical_input("scroll", 5), keep_open=True)

    assert gate.revoke(live) is True and gate.revoke(live) is False  # unchanged contract
    assert gate.revoke(spent) is False  # already spent: nothing usable to revoke - but its in-flight action is stopped (above)


# ================================================ by reading the code =======================================================================================


def test_the_backend_sends_every_chunk_from_one_loop_that_rechecks_first_and_closes_the_admission_in_a_finally():
    tree = ast.parse((APP_DIR / "computer" / "windows_backend.py").read_text(encoding="utf-8"))
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "WindowsComputerBackend")
    method = next(m for m in cls.body if isinstance(m, ast.FunctionDef) and m.name == "type_text")

    writes = [n for n in ast.walk(method) if isinstance(n, ast.Call) and ast.unparse(n.func) == "pyautogui.write"]
    admissions = [n for n in ast.walk(method) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "admit_input"]
    loops = [n for n in ast.walk(method) if isinstance(n, ast.For)]
    tries = [n for n in ast.walk(method) if isinstance(n, ast.Try) and n.finalbody]

    assert len(writes) == 1 and len(admissions) == 1 and len(loops) == 1
    assert any(kw.arg == "keep_open" and ast.unparse(kw.value) == "True" for kw in admissions[0].keywords)
    rechecks = [n for n in ast.walk(loops[0]) if isinstance(n, ast.Call) and ast.unparse(n.func) == "ticket.still_authorized"]
    assert len(rechecks) == 1 and rechecks[0].lineno < writes[0].lineno  # the recheck comes before the write, inside the same loop
    assert admissions[0].lineno < loops[0].lineno
    assert any(ast.unparse(n.func) == "ticket.close" for t in tries for f in t.finalbody for n in ast.walk(f) if isinstance(n, ast.Call))
    assert ast.unparse(loops[0].iter).startswith("range(0, len(text), TYPING_CHUNK_CHARACTERS)")  # the chunk size is the constant, and the whole text is covered


def test_the_documentation_states_the_limits_honestly():
    gate_source = (APP_DIR / "computer" / "gate.py").read_text(encoding="utf-8")
    backend_source = (APP_DIR / "computer" / "windows_backend.py").read_text(encoding="utf-8")

    for source in (gate_source, backend_source):
        assert "cannot be recalled" in source
    assert "NEXT recheck" in gate_source and "next recheck" in backend_source
