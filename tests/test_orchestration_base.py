"""
Tests for the pure parsing/validation functions in
app/orchestration/base.py: parse_autonomy, parse_priority, and (added
after M15's initial pass) parse_autonomous_permissions - the function
that makes AUTONOMOUS's boundary an explicit, per-Job configuration
instead of a hardcoded engine constant, while structurally forbidding
"dangerous" from ever being part of it.
"""

import pytest

from app.orchestration.base import (
    AutonomyError,
    AutonomyLevel,
    AutonomousPermissionError,
    DEFAULT_AUTONOMOUS_PERMISSIONS,
    JobPriority,
    JobPriorityError,
    parse_autonomous_permissions,
    parse_autonomy,
    parse_priority,
)


def test_parse_autonomy_accepts_valid_levels():
    assert parse_autonomy("manual") is AutonomyLevel.MANUAL
    assert parse_autonomy("assisted") is AutonomyLevel.ASSISTED
    assert parse_autonomy("autonomous") is AutonomyLevel.AUTONOMOUS


def test_parse_autonomy_rejects_unknown_level():
    with pytest.raises(AutonomyError):
        parse_autonomy("godmode")


def test_parse_priority_accepts_valid_priorities():
    assert parse_priority("low") is JobPriority.LOW
    assert parse_priority("high") is JobPriority.HIGH


def test_parse_priority_rejects_unknown_priority():
    with pytest.raises(JobPriorityError):
        parse_priority("urgent")


# --- parse_autonomous_permissions -------------------------------------------


def test_parse_autonomous_permissions_none_returns_the_default():
    assert parse_autonomous_permissions(None) == DEFAULT_AUTONOMOUS_PERMISSIONS
    assert DEFAULT_AUTONOMOUS_PERMISSIONS == frozenset({"confirm"})


def test_parse_autonomous_permissions_empty_list_is_a_real_opt_out():
    """An explicit empty list is different from None - it means AUTONOMOUS
    should behave like ASSISTED for this Job, not 'use the default'."""
    result = parse_autonomous_permissions([])
    assert result == frozenset()
    assert result != DEFAULT_AUTONOMOUS_PERMISSIONS


def test_parse_autonomous_permissions_accepts_confirm():
    assert parse_autonomous_permissions(["confirm"]) == frozenset({"confirm"})


def test_parse_autonomous_permissions_rejects_dangerous():
    with pytest.raises(AutonomousPermissionError, match="dangerous"):
        parse_autonomous_permissions(["dangerous"])


def test_parse_autonomous_permissions_rejects_dangerous_even_mixed_with_valid_values():
    with pytest.raises(AutonomousPermissionError):
        parse_autonomous_permissions(["confirm", "dangerous"])


@pytest.mark.parametrize("value", ["read", "safe_action", "not_a_real_permission", ""])
def test_parse_autonomous_permissions_rejects_anything_else(value):
    """READ/SAFE_ACTION are excluded too - they're already unconditionally
    automatic regardless of this setting, so listing them would imply
    control that doesn't exist (see the module's own docstring)."""
    with pytest.raises(AutonomousPermissionError):
        parse_autonomous_permissions([value])
