"""
Tests for deterministic RequestContext construction
(app/context_engine/request_builder.py): time-scope parsing, context-
intent classification (including reuse of an existing Capability
decision), and entity-mention extraction. No AI call is ever made here -
these tests assert the exact deterministic outputs for fixed inputs.
"""

from datetime import datetime, timezone

from app.capability import Capability
from app.context_engine.base import ContextIntent, TimeScopeKind
from app.context_engine.request_builder import (
    build_request_context,
    classify_context_intent,
    extract_time_scope,
)

_FIXED_NOW = datetime(2026, 3, 4, 12, 0, 0, tzinfo=timezone.utc)  # a Wednesday


# -- time scope -------------------------------------------------------------


def test_extract_time_scope_defaults_to_none():
    scope = extract_time_scope("what is my robotics project about", now=_FIXED_NOW)
    assert scope.kind is TimeScopeKind.NONE
    assert scope.start is None
    assert scope.end is None


def test_extract_time_scope_today():
    scope = extract_time_scope("what's due today?", now=_FIXED_NOW)
    assert scope.kind is TimeScopeKind.TODAY
    assert scope.start == "2026-03-04T00:00:00+00:00"
    assert scope.end == "2026-03-05T00:00:00+00:00"


def test_extract_time_scope_tomorrow():
    scope = extract_time_scope("what's due tomorrow?", now=_FIXED_NOW)
    assert scope.kind is TimeScopeKind.TOMORROW
    assert scope.start == "2026-03-05T00:00:00+00:00"
    assert scope.end == "2026-03-06T00:00:00+00:00"


def test_extract_time_scope_this_week_is_the_iso_calendar_week():
    scope = extract_time_scope("what's happening this week?", now=_FIXED_NOW)
    assert scope.kind is TimeScopeKind.THIS_WEEK
    # 2026-03-04 is a Wednesday; the ISO week runs Monday 2026-03-02
    # through the following Monday 2026-03-09.
    assert scope.start == "2026-03-02T00:00:00+00:00"
    assert scope.end == "2026-03-09T00:00:00+00:00"


def test_extract_time_scope_overdue():
    scope = extract_time_scope("show me overdue tasks", now=_FIXED_NOW)
    assert scope.kind is TimeScopeKind.OVERDUE
    assert scope.start is None
    assert scope.end == "2026-03-04T12:00:00+00:00"


def test_extract_time_scope_upcoming():
    scope = extract_time_scope("what's upcoming?", now=_FIXED_NOW)
    assert scope.kind is TimeScopeKind.UPCOMING
    assert scope.start == "2026-03-04T12:00:00+00:00"
    assert scope.end is None


def test_extract_time_scope_now():
    scope = extract_time_scope("what am I doing right now?", now=_FIXED_NOW)
    assert scope.kind is TimeScopeKind.NOW


def test_extract_time_scope_date_range():
    scope = extract_time_scope(
        "show me events from 2026-03-10 to 2026-03-15", now=_FIXED_NOW
    )
    assert scope.kind is TimeScopeKind.DATE_RANGE
    assert scope.start == "2026-03-10T00:00:00+00:00"
    assert scope.end == "2026-03-15T00:00:00+00:00"


def test_extract_time_scope_date_range_takes_priority_over_other_phrases():
    scope = extract_time_scope(
        "today I want events between 2026-03-10 and 2026-03-15", now=_FIXED_NOW
    )
    assert scope.kind is TimeScopeKind.DATE_RANGE


# -- context intent -----------------------------------------------------


def test_classify_context_intent_defaults_to_general():
    assert classify_context_intent("hello there") is ContextIntent.GENERAL


def test_classify_context_intent_scheduling():
    assert classify_context_intent("what's on my calendar today?") is ContextIntent.SCHEDULING


def test_classify_context_intent_status():
    assert classify_context_intent("what is blocking my robotics arm?") is ContextIntent.STATUS


def test_classify_context_intent_planning():
    assert classify_context_intent("what should I work on next?") is ContextIntent.PLANNING


def test_classify_context_intent_decision_support():
    assert (
        classify_context_intent("should I work on the robotics project or the assignment?")
        is ContextIntent.DECISION_SUPPORT
    )


def test_classify_context_intent_project_focus():
    assert (
        classify_context_intent("tell me about my robotics arm project")
        is ContextIntent.PROJECT_FOCUS
    )


def test_classify_context_intent_task_execution():
    assert classify_context_intent("finish the gripper print") is ContextIntent.TASK_EXECUTION


def test_classify_context_intent_reuses_task_status_capability_directly():
    assert (
        classify_context_intent("is that finished?", capability=Capability.TASK_STATUS)
        is ContextIntent.STATUS
    )


def test_classify_context_intent_task_capability_defaults_to_planning():
    assert (
        classify_context_intent("prepare everything for tomorrow", capability=Capability.TASK)
        is ContextIntent.PLANNING
    )


def test_classify_context_intent_task_capability_still_prefers_a_specific_keyword_match():
    assert (
        classify_context_intent(
            "organize my robotics project", capability=Capability.TASK
        )
        is ContextIntent.PROJECT_FOCUS
    )


def test_classify_context_intent_never_guesses_for_unrelated_capabilities():
    assert classify_context_intent("what's the weather", capability=Capability.SEARCH) is (
        ContextIntent.GENERAL
    )


# -- entity mention extraction -----------------------------------------


def test_build_request_context_extracts_generic_my_phrase_as_project_hint():
    context = build_request_context("What is blocking my robotics arm?", now=_FIXED_NOW)
    assert "robotics arm" in context.mentioned_projects


def test_build_request_context_extracts_explicit_project_keyword_phrase():
    context = build_request_context(
        "tell me about my robotics arm project", now=_FIXED_NOW
    )
    assert "robotics arm" in context.mentioned_projects


def test_build_request_context_extracts_explicit_goal_keyword_phrase():
    context = build_request_context(
        "what's the status of the gripper goal", now=_FIXED_NOW
    )
    assert "gripper" in context.mentioned_goals


def test_build_request_context_extracts_quoted_phrase_into_every_category():
    context = build_request_context('tell me about "Mars mission"', now=_FIXED_NOW)
    assert "Mars mission" in context.mentioned_projects
    assert "Mars mission" in context.mentioned_goals
    assert "Mars mission" in context.mentioned_tasks
    assert "Mars mission" in context.mentioned_commitments
    assert "Mars mission" in context.mentioned_resources


def test_build_request_context_generic_my_phrase_does_not_double_count_a_keyword_match():
    context = build_request_context(
        "what's the status of my gripper goal", now=_FIXED_NOW
    )
    assert context.mentioned_goals.count("gripper") == 1
    assert "gripper goal" not in context.mentioned_projects
    assert context.mentioned_projects == []


def test_build_request_context_with_no_mentions_has_empty_lists():
    context = build_request_context("what should I do today?", now=_FIXED_NOW)
    assert context.mentioned_projects == []
    assert context.mentioned_goals == []
    assert context.mentioned_tasks == []
    assert context.mentioned_commitments == []
    assert context.mentioned_resources == []


# -- full build_request_context wiring -----------------------------------


def test_build_request_context_sets_required_context_and_budget_from_intent():
    context = build_request_context("what's on my calendar today?", now=_FIXED_NOW)
    assert context.context_intent is ContextIntent.SCHEDULING
    from app.context_engine.base import default_required_context

    assert context.required_context == default_required_context(ContextIntent.SCHEDULING)
    assert context.context_budget.projects == 5


def test_build_request_context_preserves_raw_request_verbatim():
    message = "What is blocking my robotics arm?"
    context = build_request_context(message, now=_FIXED_NOW)
    assert context.raw_request == message
