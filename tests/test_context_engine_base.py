"""
Tests for the pure types in app/context_engine/base.py: RequestContext
defaults, ContextIntent/TimeScopeKind/RequiredContextCategory enums,
TimeScope, ContextBudget (including its centralized defaults), and
default_required_context.
"""

import pytest

from app.context_engine.base import (
    CategoryRetrieval,
    ContextBudget,
    ContextIntent,
    ContextIntentError,
    ContextWindow,
    DEFAULT_CONTEXT_BUDGET,
    ItemProvenance,
    RequestContext,
    RequestContextError,
    RequiredContextCategory,
    TimeScope,
    TimeScopeKind,
    default_required_context,
    parse_context_intent,
)


def test_context_intent_has_exactly_the_seven_specified_members():
    assert {member.value for member in ContextIntent} == {
        "general",
        "project_focus",
        "task_execution",
        "planning",
        "scheduling",
        "status",
        "decision_support",
    }


def test_parse_context_intent_accepts_valid_and_rejects_unknown():
    assert parse_context_intent("status") is ContextIntent.STATUS
    with pytest.raises(ContextIntentError):
        parse_context_intent("mood")


def test_time_scope_kind_has_exactly_the_eight_specified_members():
    assert {member.value for member in TimeScopeKind} == {
        "none",
        "now",
        "today",
        "tomorrow",
        "this_week",
        "date_range",
        "upcoming",
        "overdue",
    }


def test_required_context_category_has_exactly_the_nine_specified_members():
    assert {member.value for member in RequiredContextCategory} == {
        "projects",
        "goals",
        "tasks",
        "commitments",
        "events",
        "blockers",
        "resources",
        "preferences",
        "memory",
    }


def test_time_scope_defaults_to_none_with_no_bounds():
    scope = TimeScope()
    assert scope.kind is TimeScopeKind.NONE
    assert scope.start is None
    assert scope.end is None


def test_context_budget_matches_the_specified_defaults():
    budget = ContextBudget()
    assert budget.projects == 5
    assert budget.goals == 8
    assert budget.tasks == 10
    assert budget.commitments == 5
    assert budget.events == 8
    assert budget.blockers == 8
    assert budget.resources == 8
    assert budget.preferences == 10
    assert budget.memory == 10


def test_context_budget_for_category_reads_the_right_field():
    budget = ContextBudget(projects=3)
    assert budget.for_category(RequiredContextCategory.PROJECTS) == 3
    assert budget.for_category(RequiredContextCategory.GOALS) == 8


def test_default_context_budget_is_a_shared_instance():
    assert DEFAULT_CONTEXT_BUDGET.projects == 5


def test_default_required_context_excludes_memory_for_every_intent():
    for intent in ContextIntent:
        assert RequiredContextCategory.MEMORY not in default_required_context(intent)


def test_default_required_context_scheduling_is_narrow():
    categories = default_required_context(ContextIntent.SCHEDULING)
    assert categories == {
        RequiredContextCategory.EVENTS,
        RequiredContextCategory.COMMITMENTS,
        RequiredContextCategory.TASKS,
    }


def test_default_required_context_decision_support_includes_preferences():
    assert RequiredContextCategory.PREFERENCES in default_required_context(
        ContextIntent.DECISION_SUPPORT
    )


def test_request_context_defaults():
    context = RequestContext(raw_request="hello")

    assert context.context_intent is ContextIntent.GENERAL
    assert context.mentioned_projects == []
    assert context.mentioned_goals == []
    assert context.mentioned_tasks == []
    assert context.mentioned_commitments == []
    assert context.mentioned_resources == []
    assert context.time_scope == TimeScope()
    assert context.required_context == default_required_context(ContextIntent.GENERAL)
    assert context.context_budget is DEFAULT_CONTEXT_BUDGET


def test_request_context_accepts_explicit_overrides():
    budget = ContextBudget(projects=1)
    context = RequestContext(
        raw_request="x",
        context_intent=ContextIntent.STATUS,
        mentioned_projects=["robotics arm"],
        required_context=frozenset({RequiredContextCategory.PROJECTS}),
        context_budget=budget,
    )

    assert context.context_intent is ContextIntent.STATUS
    assert context.mentioned_projects == ["robotics arm"]
    assert context.required_context == {RequiredContextCategory.PROJECTS}
    assert context.context_budget.projects == 1


# -- structural validation (RequestContext/TimeScope/ContextBudget) -------


def test_request_context_rejects_a_non_enum_context_intent():
    with pytest.raises(RequestContextError):
        RequestContext(raw_request="x", context_intent="status")  # type: ignore[arg-type]


def test_request_context_rejects_a_non_time_scope_time_scope():
    with pytest.raises(RequestContextError):
        RequestContext(raw_request="x", time_scope="today")  # type: ignore[arg-type]


def test_request_context_rejects_a_non_context_budget_budget():
    with pytest.raises(RequestContextError):
        RequestContext(raw_request="x", context_budget={"projects": 5})  # type: ignore[arg-type]


def test_request_context_rejects_invalid_required_context_members():
    with pytest.raises(RequestContextError):
        RequestContext(raw_request="x", required_context=frozenset({"projects"}))  # type: ignore[arg-type]


def test_time_scope_rejects_a_non_enum_kind():
    with pytest.raises(RequestContextError):
        TimeScope(kind="today")  # type: ignore[arg-type]


def test_time_scope_accepts_a_valid_kind():
    scope = TimeScope(kind=TimeScopeKind.TODAY, start="2026-01-01T00:00:00+00:00")
    assert scope.kind is TimeScopeKind.TODAY


@pytest.mark.parametrize(
    "kwargs",
    [
        {"projects": -1},
        {"goals": -1},
        {"tasks": -1},
        {"commitments": -1},
        {"events": -1},
        {"blockers": -1},
        {"resources": -1},
        {"preferences": -1},
        {"memory": -1},
    ],
)
def test_context_budget_rejects_negative_values(kwargs):
    with pytest.raises(RequestContextError):
        ContextBudget(**kwargs)


def test_context_budget_accepts_zero():
    budget = ContextBudget(projects=0)  # a valid, deliberate "retrieve none" budget
    assert budget.projects == 0


# -- moved M16.2 retrieval-metadata types (previously in app/world_model) --


def test_item_provenance_holds_score_and_reasons():
    provenance = ItemProvenance(relevance_score=42.0, reasons=("explicit_match",))
    assert provenance.relevance_score == 42.0
    assert provenance.reasons == ("explicit_match",)


def test_category_retrieval_defaults_to_empty():
    retrieval = CategoryRetrieval()
    assert retrieval.considered == 0
    assert retrieval.selected == 0
    assert retrieval.items == {}


def test_context_window_defaults_to_no_bounds():
    window = ContextWindow()
    assert window.start is None
    assert window.end is None
