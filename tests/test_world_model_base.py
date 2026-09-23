"""
Tests for the pure parsing/validation functions and transition maps in
app/world_model/base.py - no SQLite, no service layer, just the
deterministic domain rules every higher layer builds on.
"""

import pytest

from app.world_model.base import (
    BlockerSeverity,
    BlockerSeverityError,
    BlockerStatus,
    CommitmentStatus,
    EntitySource,
    EntitySourceError,
    GoalStatus,
    InvalidConfidenceError,
    InvalidTransitionError,
    LinkedEntityKind,
    LinkedEntityKindError,
    Priority,
    PriorityError,
    ProjectStatus,
    ResourceStatus,
    ResourceStatusError,
    ResourceType,
    ResourceTypeError,
    TaskRefType,
    TaskRefTypeError,
    TERMINAL_BLOCKER_STATUSES,
    TERMINAL_COMMITMENT_STATUSES,
    TERMINAL_GOAL_STATUSES,
    TERMINAL_PROJECT_STATUSES,
    VALID_BLOCKER_LINK_KINDS,
    VALID_RESOURCE_LINK_KINDS,
    parse_blocker_severity,
    parse_linked_entity_kind,
    parse_priority,
    parse_resource_status,
    parse_resource_type,
    parse_source,
    parse_task_ref_type,
    validate_blocker_transition,
    validate_commitment_transition,
    validate_confidence,
    validate_goal_transition,
    validate_project_transition,
)


# -- parsers --------------------------------------------------------------


def test_parse_priority_accepts_valid_and_rejects_unknown():
    assert parse_priority("low") is Priority.LOW
    assert parse_priority("high") is Priority.HIGH
    with pytest.raises(PriorityError):
        parse_priority("urgent")


def test_parse_source_accepts_valid_and_rejects_unknown():
    assert parse_source("user") is EntitySource.USER
    assert parse_source("inference") is EntitySource.INFERENCE
    with pytest.raises(EntitySourceError):
        parse_source("guess")


def test_parse_resource_type_accepts_valid_and_rejects_unknown():
    assert parse_resource_type("hardware") is ResourceType.HARDWARE
    with pytest.raises(ResourceTypeError):
        parse_resource_type("spaceship")


def test_parse_resource_status_accepts_valid_and_rejects_unknown():
    assert parse_resource_status("available") is ResourceStatus.AVAILABLE
    with pytest.raises(ResourceStatusError):
        parse_resource_status("broken")


def test_parse_blocker_severity_accepts_valid_and_rejects_unknown():
    assert parse_blocker_severity("critical") is BlockerSeverity.CRITICAL
    with pytest.raises(BlockerSeverityError):
        parse_blocker_severity("meh")


def test_parse_linked_entity_kind_accepts_valid_and_rejects_unknown():
    assert parse_linked_entity_kind("project") is LinkedEntityKind.PROJECT
    with pytest.raises(LinkedEntityKindError):
        parse_linked_entity_kind("robot")


def test_parse_task_ref_type_accepts_valid_and_rejects_unknown():
    assert parse_task_ref_type("job") is TaskRefType.JOB
    with pytest.raises(TaskRefTypeError):
        parse_task_ref_type("reminder")


# -- confidence -------------------------------------------------------------


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0, 0.999])
def test_validate_confidence_accepts_in_range_values(value):
    validate_confidence(value)  # must not raise


@pytest.mark.parametrize("value", [-0.01, 1.01, -1.0, 2.0])
def test_validate_confidence_rejects_out_of_range_values(value):
    with pytest.raises(InvalidConfidenceError):
        validate_confidence(value)


def test_validate_confidence_rejects_non_numeric():
    with pytest.raises(InvalidConfidenceError):
        validate_confidence("high")  # type: ignore[arg-type]


def test_validate_confidence_rejects_bool():
    # bool is technically an int subclass in Python - explicitly excluded
    # so a caller's accidental `confidence=True` doesn't silently pass.
    with pytest.raises(InvalidConfidenceError):
        validate_confidence(True)  # type: ignore[arg-type]


# -- Project transitions ----------------------------------------------------


@pytest.mark.parametrize(
    "current,target",
    [
        (ProjectStatus.ACTIVE, ProjectStatus.PAUSED),
        (ProjectStatus.ACTIVE, ProjectStatus.COMPLETED),
        (ProjectStatus.ACTIVE, ProjectStatus.ARCHIVED),
        (ProjectStatus.PAUSED, ProjectStatus.ACTIVE),
        (ProjectStatus.PAUSED, ProjectStatus.ARCHIVED),
    ],
)
def test_valid_project_transitions(current, target):
    validate_project_transition(current, target)  # must not raise


@pytest.mark.parametrize(
    "current,target",
    [
        (ProjectStatus.PAUSED, ProjectStatus.COMPLETED),
        (ProjectStatus.COMPLETED, ProjectStatus.ACTIVE),
        (ProjectStatus.ARCHIVED, ProjectStatus.ACTIVE),
        (ProjectStatus.COMPLETED, ProjectStatus.ARCHIVED),
        (ProjectStatus.ACTIVE, ProjectStatus.ACTIVE),
    ],
)
def test_invalid_project_transitions(current, target):
    with pytest.raises(InvalidTransitionError):
        validate_project_transition(current, target)


def test_project_terminal_statuses():
    assert TERMINAL_PROJECT_STATUSES == {ProjectStatus.COMPLETED, ProjectStatus.ARCHIVED}


# -- Goal transitions ---------------------------------------------------


@pytest.mark.parametrize(
    "current,target",
    [
        (GoalStatus.ACTIVE, GoalStatus.PAUSED),
        (GoalStatus.ACTIVE, GoalStatus.ACHIEVED),
        (GoalStatus.ACTIVE, GoalStatus.ABANDONED),
        (GoalStatus.PAUSED, GoalStatus.ACTIVE),
        (GoalStatus.PAUSED, GoalStatus.ABANDONED),
    ],
)
def test_valid_goal_transitions(current, target):
    validate_goal_transition(current, target)


@pytest.mark.parametrize(
    "current,target",
    [
        (GoalStatus.PAUSED, GoalStatus.ACHIEVED),
        (GoalStatus.ACHIEVED, GoalStatus.ACTIVE),
        (GoalStatus.ABANDONED, GoalStatus.ACTIVE),
    ],
)
def test_invalid_goal_transitions(current, target):
    with pytest.raises(InvalidTransitionError):
        validate_goal_transition(current, target)


def test_goal_terminal_statuses():
    assert TERMINAL_GOAL_STATUSES == {GoalStatus.ACHIEVED, GoalStatus.ABANDONED}


# -- Commitment transitions -----------------------------------------------


@pytest.mark.parametrize(
    "target",
    [CommitmentStatus.FULFILLED, CommitmentStatus.MISSED, CommitmentStatus.CANCELLED],
)
def test_valid_commitment_transitions_from_open(target):
    validate_commitment_transition(CommitmentStatus.OPEN, target)


@pytest.mark.parametrize(
    "current",
    [CommitmentStatus.FULFILLED, CommitmentStatus.MISSED, CommitmentStatus.CANCELLED],
)
def test_invalid_commitment_transitions_out_of_terminal_states(current):
    with pytest.raises(InvalidTransitionError):
        validate_commitment_transition(current, CommitmentStatus.OPEN)


def test_commitment_terminal_statuses():
    assert TERMINAL_COMMITMENT_STATUSES == {
        CommitmentStatus.FULFILLED,
        CommitmentStatus.MISSED,
        CommitmentStatus.CANCELLED,
    }


# -- Blocker transitions --------------------------------------------------


@pytest.mark.parametrize("target", [BlockerStatus.RESOLVED, BlockerStatus.DISMISSED])
def test_valid_blocker_transitions_from_active(target):
    validate_blocker_transition(BlockerStatus.ACTIVE, target)


@pytest.mark.parametrize("current", [BlockerStatus.RESOLVED, BlockerStatus.DISMISSED])
def test_invalid_blocker_transitions_out_of_terminal_states(current):
    with pytest.raises(InvalidTransitionError):
        validate_blocker_transition(current, BlockerStatus.ACTIVE)


def test_blocker_terminal_statuses():
    assert TERMINAL_BLOCKER_STATUSES == {BlockerStatus.RESOLVED, BlockerStatus.DISMISSED}


# -- relationship-kind constraints ----------------------------------------


def test_valid_blocker_link_kinds_cover_all_four_entity_kinds():
    assert VALID_BLOCKER_LINK_KINDS == {
        LinkedEntityKind.PROJECT,
        LinkedEntityKind.GOAL,
        LinkedEntityKind.TASK,
        LinkedEntityKind.JOB,
    }


def test_valid_resource_link_kinds_exclude_goal():
    assert VALID_RESOURCE_LINK_KINDS == {
        LinkedEntityKind.PROJECT,
        LinkedEntityKind.TASK,
        LinkedEntityKind.JOB,
    }
    assert LinkedEntityKind.GOAL not in VALID_RESOURCE_LINK_KINDS
