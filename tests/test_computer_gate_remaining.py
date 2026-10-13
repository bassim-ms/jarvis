"""
`InputGate.remaining()`: the READ-ONLY capacity query a multi-step task uses to refuse to START what the rolling budget cannot finish.
It must agree exactly with what `admit` would do, and must never consume, log or change anything.

`remaining(authority=grant)` is the smaller of the rate budget and the grant's unspent entries (0 without a usable grant), so every test here holds a REAL grant with at least
as many entries as the budget it examines, minted by the real issuer (tests/grant_support.py): the number under test is then the budget, as it always was.
"""

import logging

import pytest

from app.computer.base import ComputerControlError
from app.computer.gate import InputGate
from grant_support import admit_with, authorize


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def gate_with(limit=10, enabled=True, clock=None):
    clock = clock or Clock()
    state = {"enabled": enabled, "limit": limit}
    return InputGate(enabled=lambda: state["enabled"], limit=lambda: state["limit"], clock=clock), clock, state


def grant_of(gate, action, entries):
    """A real grant with `entries` copies of one action: more entries than the budget, so the BUDGET is what `remaining` reports."""
    return authorize(gate, *[action] * entries)


def test_a_fresh_gate_reports_its_whole_budget_and_a_switched_off_gate_reports_nothing():
    gate, _, state = gate_with(limit=10)
    grant = grant_of(gate, "press_key", 12)
    assert gate.remaining(authority=grant) == 10
    state["enabled"] = False
    assert gate.remaining(authority=grant) == 0


def test_each_admitted_action_takes_one_slot_and_a_refused_one_takes_none():
    gate, _, state = gate_with(limit=3)
    grant = grant_of(gate, "press_key", 6)
    for expected in (2, 1, 0):
        admit_with(gate, grant, "press_key")
        assert gate.remaining(authority=grant) == expected
    with pytest.raises(ComputerControlError):
        admit_with(gate, grant, "press_key")  # over the limit: refused, nothing consumed
    assert gate.remaining(authority=grant) == 0
    state["enabled"] = False
    with pytest.raises(ComputerControlError):
        admit_with(gate, grant, "press_key")
    state["enabled"] = True
    assert gate.remaining(authority=grant) == 0


def test_asking_never_consumes_a_slot():
    gate, _, _ = gate_with(limit=2)
    grant = grant_of(gate, "type_text", 2)
    for _ in range(50):
        assert gate.remaining(authority=grant) == 2
        gate.check("type_text", authority=grant)
    admit_with(gate, grant, "type_text")
    admit_with(gate, grant, "type_text")  # both slots (and both grant entries) were still there after fifty questions
    assert gate.remaining(authority=grant) == 0


def test_recent_actions_count_until_exactly_sixty_seconds_old():
    gate, clock, _ = gate_with(limit=5)
    grant = grant_of(gate, "press_key", 8)
    admit_with(gate, grant, "press_key")
    clock.now += 30
    admit_with(gate, grant, "press_key")
    admit_with(gate, grant, "press_key")
    assert gate.remaining(authority=grant) == 2
    clock.now += 29.999  # the first action is 59.999 s old: still counts
    assert gate.remaining(authority=grant) == 2
    clock.now += 0.001  # exactly 60 s old: its slot is free again
    assert gate.remaining(authority=grant) == 3
    clock.now += 30  # the other two are now 60 s old as well
    assert gate.remaining(authority=grant) == 5


def test_it_agrees_with_admit_at_every_count():
    gate, clock, _ = gate_with(limit=4)
    grant = grant_of(gate, "click_mouse", 6)
    for _ in range(4):
        before = gate.remaining(authority=grant)
        assert before > 0
        admit_with(gate, grant, "click_mouse")
        assert gate.remaining(authority=grant) == before - 1
    assert gate.remaining(authority=grant) == 0
    with pytest.raises(ComputerControlError, match="limited to 4 actions per minute"):
        admit_with(gate, grant, "click_mouse")


def test_a_limit_lowered_below_what_is_used_reports_zero_not_a_negative_number():
    gate, _, state = gate_with(limit=5)
    grant = grant_of(gate, "press_key", 8)
    for _ in range(4):
        admit_with(gate, grant, "press_key")
    state["limit"] = 2
    assert gate.remaining(authority=grant) == 0
    state["limit"] = 6
    assert gate.remaining(authority=grant) == 2


@pytest.mark.parametrize("bad", [True, False, -1, 2.5, "10", None])
def test_an_unreadable_or_invalid_limit_reports_zero(bad):
    gate, _, state = gate_with(limit=10)
    grant = grant_of(gate, "press_key", 12)
    state["limit"] = bad
    assert gate.remaining(authority=grant) == 0


def test_a_limit_or_switch_that_raises_reports_zero():
    def boom():
        raise RuntimeError("unreadable")

    broken_switch = InputGate(enabled=boom, limit=lambda: 10)
    assert broken_switch.remaining() == 0  # (no grant can exist for a switch that cannot be read)
    broken_limit = InputGate(enabled=lambda: True, limit=boom)
    assert broken_limit.remaining(authority=grant_of(broken_limit, "press_key", 2)) == 0


@pytest.mark.parametrize("truthy", [1, "true", "yes", object()])
def test_only_a_real_true_switch_counts_as_on(truthy):
    gate, _, state = gate_with(limit=10)
    grant = grant_of(gate, "press_key", 12)
    state["enabled"] = truthy
    assert gate.remaining(authority=grant) == 0


def test_asking_leaves_no_trace_in_the_audit_log(caplog):
    gate, _, state = gate_with(limit=3)
    grant = grant_of(gate, "press_key", 4)  # (minting is logged; asking is what must not be)
    caplog.set_level(logging.DEBUG)
    caplog.clear()
    gate.remaining(authority=grant)
    state["enabled"] = False
    gate.remaining(authority=grant)
    assert [r for r in caplog.records if r.name.startswith("jarvis.computer")] == []
