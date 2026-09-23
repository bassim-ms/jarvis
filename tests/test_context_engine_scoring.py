"""
Tests for app/context_engine/scoring.py's pure scoring functions -
relevance weights, explicit-match dominance, deterministic time
relevance (including overdue staying high, never zero), and
deterministic tie-breaking. No World Model/store involved here.
"""

from app.context_engine.scoring import (
    DEFAULT_WEIGHTS,
    matches_hint,
    priority_relevance,
    tie_break_key,
    time_relevance,
)
from app.world_model.base import Priority


def test_matches_hint_is_case_insensitive_substring_either_direction():
    assert matches_hint("robotics", "Robotics Arm Project") is True
    assert matches_hint("Robotics Arm Project", "robotics") is True
    assert matches_hint("ROBOTICS", "robotics arm") is True
    assert matches_hint("physics", "Robotics Arm Project") is False


def test_matches_hint_rejects_blank_inputs():
    assert matches_hint("", "Robotics Arm Project") is False
    assert matches_hint("robotics", "") is False


def test_priority_relevance_orders_high_above_medium_above_low():
    high, _ = priority_relevance(Priority.HIGH)
    medium, _ = priority_relevance(Priority.MEDIUM)
    low, _ = priority_relevance(Priority.LOW)
    assert high > medium > low


def test_explicit_match_dominates_every_other_signal_combined():
    """The core design rule: a candidate that stacks every other signal
    (relationship + active state + high priority + time-now + dependency
    + blocker + resource match) must still score LOWER than a candidate
    whose only signal is an explicit match."""
    w = DEFAULT_WEIGHTS
    max_inferred_stack = (
        w.RELATIONSHIP
        + w.CURRENT_STATE_ACTIVE
        + w.PRIORITY_HIGH
        + w.TIME_NOW
        + w.DEPENDENCY
        + w.BLOCKER_PRESENT
        + w.RESOURCE_MATCH
    )
    assert w.EXPLICIT_MATCH > max_inferred_stack


def test_time_relevance_now_is_highest():
    now = "2026-03-04T12:00:00+00:00"
    due_now = "2026-03-04T12:30:00+00:00"
    score, reason = time_relevance(due_now, now)
    assert reason == "time_now"
    assert score == DEFAULT_WEIGHTS.TIME_NOW


def test_time_relevance_overdue_stays_high_not_zero():
    now = "2026-03-04T12:00:00+00:00"
    overdue = "2026-03-01T00:00:00+00:00"
    score, reason = time_relevance(overdue, now)
    assert reason == "time_overdue"
    assert score == DEFAULT_WEIGHTS.TIME_OVERDUE
    assert score > 0


def test_time_relevance_overdue_scores_higher_than_far_future():
    now = "2026-03-04T12:00:00+00:00"
    overdue_score, _ = time_relevance("2026-01-01T00:00:00+00:00", now)
    far_future_score, _ = time_relevance("2027-01-01T00:00:00+00:00", now)
    assert overdue_score > far_future_score


def test_time_relevance_today_tomorrow_this_week_far_future_descending():
    now = "2026-03-04T00:00:00+00:00"  # Wednesday
    today, _ = time_relevance("2026-03-04T18:00:00+00:00", now)
    tomorrow, _ = time_relevance("2026-03-05T09:00:00+00:00", now)
    this_week, _ = time_relevance("2026-03-08T09:00:00+00:00", now)
    far_future, _ = time_relevance("2027-01-01T00:00:00+00:00", now)
    assert today > tomorrow > this_week > far_future


def test_time_relevance_returns_none_for_missing_timestamp():
    assert time_relevance(None, "2026-03-04T00:00:00+00:00") is None


def test_time_relevance_returns_none_for_unparseable_timestamp():
    assert time_relevance("not-a-date", "2026-03-04T00:00:00+00:00") is None


# -- tie-breaking -----------------------------------------------------


def test_tie_break_prefers_higher_priority_first():
    high_key = tie_break_key(
        priority=Priority.HIGH,
        proximity_timestamp=None,
        updated_at="2026-01-01T00:00:00+00:00",
        entity_id="b",
    )
    low_key = tie_break_key(
        priority=Priority.LOW,
        proximity_timestamp=None,
        updated_at="2026-01-01T00:00:00+00:00",
        entity_id="a",
    )
    assert high_key < low_key  # sorts ascending = "better" first


def test_tie_break_prefers_closer_due_time_when_priority_equal():
    closer = tie_break_key(
        priority=Priority.MEDIUM,
        proximity_timestamp="2026-03-05T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        entity_id="b",
    )
    farther = tie_break_key(
        priority=Priority.MEDIUM,
        proximity_timestamp="2026-06-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        entity_id="a",
    )
    assert closer < farther


def test_tie_break_prefers_a_present_timestamp_over_a_missing_one():
    present = tie_break_key(
        priority=Priority.MEDIUM,
        proximity_timestamp="2026-03-05T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        entity_id="a",
    )
    missing = tie_break_key(
        priority=Priority.MEDIUM,
        proximity_timestamp=None,
        updated_at="2026-01-01T00:00:00+00:00",
        entity_id="a",
    )
    assert present < missing


def test_tie_break_prefers_more_recently_updated_when_priority_and_time_equal():
    recent = tie_break_key(
        priority=Priority.MEDIUM,
        proximity_timestamp=None,
        updated_at="2026-03-01T00:00:00+00:00",
        entity_id="b",
    )
    older = tie_break_key(
        priority=Priority.MEDIUM,
        proximity_timestamp=None,
        updated_at="2026-01-01T00:00:00+00:00",
        entity_id="a",
    )
    assert recent < older


def test_tie_break_falls_back_to_stable_id_ordering():
    a = tie_break_key(
        priority=Priority.MEDIUM,
        proximity_timestamp=None,
        updated_at="2026-01-01T00:00:00+00:00",
        entity_id="aaa",
    )
    b = tie_break_key(
        priority=Priority.MEDIUM,
        proximity_timestamp=None,
        updated_at="2026-01-01T00:00:00+00:00",
        entity_id="bbb",
    )
    assert a < b
