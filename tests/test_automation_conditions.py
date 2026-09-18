"""
Unit tests for automation Conditions (app/automation/base.py):
FactEqualsCondition, TimeWindowCondition, and condition_from_dict/
to_dict serialization.
"""

from datetime import datetime, timezone

import pytest

from app.automation.base import (
    ConditionContext,
    ConditionError,
    FactEqualsCondition,
    TimeWindowCondition,
    condition_from_dict,
)


def _ctx(hour: int, facts: dict[str, str] | None = None) -> ConditionContext:
    now = datetime(2026, 1, 1, hour, 0, 0, tzinfo=timezone.utc)
    return ConditionContext(now=now, facts=facts or {})


# --- FactEqualsCondition --------------------------------------------------


def test_fact_equals_true_when_fact_matches():
    condition = FactEqualsCondition(key="mode", value="work")

    assert condition.evaluate(_ctx(10, {"mode": "work"})) is True


def test_fact_equals_false_when_fact_differs():
    condition = FactEqualsCondition(key="mode", value="work")

    assert condition.evaluate(_ctx(10, {"mode": "home"})) is False


def test_fact_equals_false_when_fact_missing():
    condition = FactEqualsCondition(key="mode", value="work")

    assert condition.evaluate(_ctx(10, {})) is False


def test_fact_equals_round_trips_through_dict():
    condition = FactEqualsCondition(key="mode", value="work")

    assert condition_from_dict(condition.to_dict()) == condition


# --- TimeWindowCondition ---------------------------------------------------


def test_time_window_true_inside_normal_range():
    condition = TimeWindowCondition(start_hour=9, end_hour=18)

    assert condition.evaluate(_ctx(12)) is True


def test_time_window_false_outside_normal_range():
    condition = TimeWindowCondition(start_hour=9, end_hour=18)

    assert condition.evaluate(_ctx(20)) is False


def test_time_window_boundaries_are_inclusive_start_exclusive_end():
    condition = TimeWindowCondition(start_hour=9, end_hour=18)

    assert condition.evaluate(_ctx(9)) is True
    assert condition.evaluate(_ctx(18)) is False


def test_time_window_wraps_past_midnight():
    condition = TimeWindowCondition(start_hour=22, end_hour=6)

    assert condition.evaluate(_ctx(23)) is True
    assert condition.evaluate(_ctx(2)) is True
    assert condition.evaluate(_ctx(12)) is False


def test_time_window_round_trips_through_dict():
    condition = TimeWindowCondition(start_hour=9, end_hour=18)

    assert condition_from_dict(condition.to_dict()) == condition


# --- condition_from_dict errors --------------------------------------------


def test_condition_from_dict_rejects_unknown_type():
    with pytest.raises(ConditionError):
        condition_from_dict({"type": "mystery"})
