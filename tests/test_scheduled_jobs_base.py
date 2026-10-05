"""
Scheduled Jobs V1 / Slice 1 - the model and the state machine: exactly which status changes exist, and that nothing else does.
"""

import itertools

import pytest

from app.scheduled_jobs.base import (
    ALLOWED_TRANSITIONS,
    JOB_OUTCOMES,
    KNOWN_PROFILES,
    OPEN_STATUSES,
    PROFILE_UNATTENDED_V1,
    RESULT_STATES,
    TERMINAL_STATUSES,
    InvalidTransitionError,
    RunOutcome,
    ScheduledJobError,
    ScheduledJobStatus as S,
    StatusReason as R,
    check_transition,
    valid_via,
)

#: The WHOLE state machine, written out independently of the table under test. (status, next status, reason).
EXPECTED = {
    (S.PROPOSED, S.ACTIVE, None),
    (S.PROPOSED, S.CANCELLED, R.DECLINED),
    (S.PROPOSED, S.EXPIRED, R.PROPOSAL_EXPIRED),
    (S.ACTIVE, S.PAUSED, R.USER),
    (S.ACTIVE, S.PAUSED, R.BREAKER),
    (S.ACTIVE, S.PAUSED, R.DEVICE_REVOKED),
    (S.ACTIVE, S.PAUSED, R.BLOCKED),
    (S.ACTIVE, S.COMPLETED, R.ONE_TIME_DONE),
    (S.ACTIVE, S.COMPLETED, R.ONE_TIME_MISSED),
    (S.ACTIVE, S.COMPLETED, R.LIFETIME_ENDED),
    (S.ACTIVE, S.CANCELLED, R.USER),
    (S.PAUSED, S.ACTIVE, None),
    (S.PAUSED, S.COMPLETED, R.LIFETIME_ENDED),
    (S.PAUSED, S.CANCELLED, R.USER),
}


def test_exactly_the_expected_moves_are_allowed_and_every_other_combination_is_refused():
    reasons = [None, *R]
    for current, target, reason in itertools.product(S, S, reasons):
        if (current, target, reason) in EXPECTED:
            check_transition(current, target, reason)
        else:
            with pytest.raises(InvalidTransitionError):
                check_transition(current, target, reason)


def test_terminal_states_are_final():
    assert TERMINAL_STATUSES == {S.COMPLETED, S.CANCELLED, S.EXPIRED}
    for terminal in TERMINAL_STATUSES:
        assert ALLOWED_TRANSITIONS[terminal] == {}
        for target, reason in itertools.product(S, [None, *R]):
            with pytest.raises(InvalidTransitionError):
                check_transition(terminal, target, reason)


def test_every_open_state_can_be_ended_by_the_user():
    assert OPEN_STATUSES == {S.PROPOSED, S.ACTIVE, S.PAUSED}
    check_transition(S.PROPOSED, S.CANCELLED, R.DECLINED)
    check_transition(S.ACTIVE, S.CANCELLED, R.USER)
    check_transition(S.PAUSED, S.CANCELLED, R.USER)


def test_a_proposal_can_never_complete_or_pause_only_be_approved_declined_or_expire():
    assert set(ALLOWED_TRANSITIONS[S.PROPOSED]) == {S.ACTIVE, S.CANCELLED, S.EXPIRED}


def test_a_paused_one_comes_back_only_by_resuming_never_straight_to_completed_by_a_run():
    with pytest.raises(InvalidTransitionError):
        check_transition(S.PAUSED, S.COMPLETED, R.ONE_TIME_DONE)


def test_the_vocabularies_are_closed():
    assert {s.value for s in S} == {"proposed", "active", "paused", "completed", "cancelled", "expired"}
    assert {o.value for o in RunOutcome} == {
        "created", "created_late", "skipped_missed", "skipped_overlap", "skipped_blocked",
        "skipped_disabled", "skipped_cap", "skipped_expired",  # Slice 3: the other reasons an occurrence is consumed without a Job
    }
    assert JOB_OUTCOMES == {RunOutcome.CREATED, RunOutcome.CREATED_LATE}
    assert RESULT_STATES == {"completed", "failed", "cancelled"}


def test_only_the_one_reduce_only_profile_exists_and_unknown_ones_are_not_a_thing():
    assert KNOWN_PROFILES == {PROFILE_UNATTENDED_V1} == {"unattended_v1"}


@pytest.mark.parametrize("via", ["chat", "scheduler", "system", "device:abc-1", "a.b_c"])
def test_via_accepts_short_identifiers(via):
    assert valid_via(via) == via


@pytest.mark.parametrize("via", ["", "Chat", "has space", "x" * 51, "line\nbreak", None, 5])
def test_via_refuses_anything_else(via):
    with pytest.raises(ScheduledJobError):
        valid_via(via)
