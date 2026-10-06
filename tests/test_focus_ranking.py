"""
Focus ranking: the pure, deterministic part. Every score is a sum of the existing RelevanceWeights; there is no store, no
calendar and no model here, so a changed weight, reason id or tie-break shows up as exactly one failing assertion.
"""

import itertools
import random
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.context_engine import scoring as S
from app.focus.base import RELATION_GOAL, RELATION_NAMED, RELATION_NONE, Candidate
from app.focus.service import REASON_DUE_UNREADABLE, match_projects, rank, score_goal, score_task, urgency
from app.tasks.base import Task, TaskPriority, TaskStatus
from app.world_model.base import Priority

W = S.DEFAULT_WEIGHTS
TZ = timezone(timedelta(hours=4))
NOW = datetime(2026, 9, 29, 8, 0, tzinfo=TZ)  # a Tuesday morning, local time


def task(text="Do the thing", priority=TaskPriority.MEDIUM, due=None, created="2026-09-01T00:00:00+00:00", tid="t1"):
    return Task(tid, text, TaskStatus.TODO, priority, due, created, created)


def cand(ref, score, due_ts=None, created="2026-09-01T00:00:00+00:00"):
    return Candidate("task", ref, ref, score, (), due_ts, created)


# --- urgency: the local clock, date-only vs timed -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "due,weight,reason",
    [
        ("2026-09-28", W.TIME_OVERDUE, S.REASON_TIME_OVERDUE),  # yesterday (date only)
        ("2026-09-29", W.TIME_TODAY, S.REASON_TIME_TODAY),  # today, date only: the whole day, NOT overdue
        ("2026-09-29T07:00:00+04:00", W.TIME_OVERDUE, S.REASON_TIME_OVERDUE),  # earlier this morning
        ("2026-09-29T08:30:00+04:00", W.TIME_NOW, S.REASON_TIME_NOW),  # within the hour
        ("2026-09-29T09:00:00+04:00", W.TIME_NOW, S.REASON_TIME_NOW),  # exactly one hour
        ("2026-09-29T09:01:00+04:00", W.TIME_TODAY, S.REASON_TIME_TODAY),
        ("2026-09-29T23:59:00+04:00", W.TIME_TODAY, S.REASON_TIME_TODAY),
        ("2026-09-30", W.TIME_TOMORROW, S.REASON_TIME_TOMORROW),
        ("2026-09-30T00:10:00+04:00", W.TIME_TOMORROW, S.REASON_TIME_TOMORROW),
        ("2026-10-06", W.TIME_THIS_WEEK, S.REASON_TIME_THIS_WEEK),  # seven days out
        ("2026-10-07", W.TIME_FAR_FUTURE, S.REASON_TIME_FAR_FUTURE),
        ("2027-01-01", W.TIME_FAR_FUTURE, S.REASON_TIME_FAR_FUTURE),
        ("2026-09-29T04:30:00+00:00", W.TIME_NOW, S.REASON_TIME_NOW),  # a UTC stamp is read as the local moment it is (08:30)
        ("2026-09-29T17:00:00", W.TIME_TODAY, S.REASON_TIME_TODAY),  # a naive stamp is read as local time
    ],
)
def test_urgency_follows_the_local_clock(due, weight, reason):
    assert urgency(due, NOW)[:2] == (weight, reason)


@pytest.mark.parametrize("value", [None, "", "   ", 5, [], {}])
def test_no_due_date_adds_nothing_and_says_nothing(value):
    assert urgency(value, NOW) == (0.0, None, None)


@pytest.mark.parametrize("value", ["next Friday", "soon", "2026-13-45", "tomorrow-ish", "30/09/2026"])
def test_an_unreadable_due_date_adds_nothing_but_is_flagged_not_dropped(value):
    assert urgency(value, NOW) == (0.0, REASON_DUE_UNREADABLE, None)


def test_a_date_only_due_today_is_never_scored_overdue_unlike_the_context_engine_scorer():
    """The reason this module has its own clock reading: the Context Engine's string comparison treats a date-only "today" as midnight."""
    assert S.time_relevance("2026-09-29", "2026-09-29T08:00:00")[1] == S.REASON_TIME_OVERDUE  # documents the trap
    assert urgency("2026-09-29", NOW)[1] == S.REASON_TIME_TODAY


# --- the score is RELATION + URGENCY + PRIORITY, from the existing weights -----------------------------------------------------------


def test_a_named_task_scores_relation_plus_urgency_plus_priority_with_named_reasons():
    c = score_task(task(priority=TaskPriority.HIGH, due="2026-09-29"), NOW, RELATION_NAMED)

    assert c.score == W.EXPLICIT_MATCH + W.TIME_TODAY + W.PRIORITY_HIGH == 132
    assert c.reasons == (S.REASON_EXPLICIT_MATCH, S.REASON_TIME_TODAY, S.REASON_PRIORITY_HIGH)


def test_a_goal_related_task_gets_the_smaller_relationship_weight_and_carries_the_goal():
    c = score_task(task(priority=TaskPriority.MEDIUM), NOW, RELATION_GOAL, via="Finish the servo prototype")

    assert c.score == W.RELATIONSHIP + W.PRIORITY_MEDIUM == 24
    assert c.reasons == (S.REASON_RELATIONSHIP_GOAL, S.REASON_PRIORITY_MEDIUM) and c.via == "Finish the servo prototype"


def test_with_nothing_named_there_is_no_relation_term():
    c = score_task(task(priority=TaskPriority.LOW, due="2026-09-27"), NOW)

    assert c.score == W.TIME_OVERDUE + W.PRIORITY_LOW == 29 and c.reasons == (S.REASON_TIME_OVERDUE, S.REASON_PRIORITY_LOW)


@pytest.mark.parametrize("priority,weight,reason", [
    (TaskPriority.HIGH, W.PRIORITY_HIGH, S.REASON_PRIORITY_HIGH),
    (TaskPriority.MEDIUM, W.PRIORITY_MEDIUM, S.REASON_PRIORITY_MEDIUM),
    (TaskPriority.LOW, W.PRIORITY_LOW, S.REASON_PRIORITY_LOW),
])
def test_priority_is_the_tasks_own_field(priority, weight, reason):
    c = score_task(task(priority=priority), NOW)
    assert c.score == weight and c.reasons == (reason,)


def test_priority_is_never_inferred_from_wording():
    plain = score_task(task("Water the plants"), NOW)
    shouting = score_task(task("URGENT!!! CRITICAL ASAP highest priority emergency deadline today"), NOW)
    assert plain.score == shouting.score and plain.reasons == shouting.reasons


def test_an_unreadable_due_date_is_carried_for_display():
    c = score_task(task(due="next Friday"), NOW)
    assert c.reasons == (REASON_DUE_UNREADABLE, S.REASON_PRIORITY_MEDIUM) and c.due_text == "next Friday" and c.score == W.PRIORITY_MEDIUM


def test_the_ranking_reasons_are_the_existing_scoring_constants_not_new_ids():
    used = {S.REASON_EXPLICIT_MATCH, S.REASON_RELATIONSHIP_GOAL, S.REASON_TIME_OVERDUE, S.REASON_TIME_NOW, S.REASON_TIME_TODAY,
            S.REASON_TIME_TOMORROW, S.REASON_TIME_THIS_WEEK, S.REASON_TIME_FAR_FUTURE, S.REASON_PRIORITY_HIGH,
            S.REASON_PRIORITY_MEDIUM, S.REASON_PRIORITY_LOW}
    seen = set()
    for relation, via in ((RELATION_NAMED, ""), (RELATION_GOAL, "g"), (RELATION_NONE, "")):
        for due in (None, "2026-09-28", "2026-09-29T08:30:00+04:00", "2026-09-29", "2026-09-30", "2026-10-03", "2027-01-01", "later"):
            for priority in TaskPriority:
                seen.update(score_task(task(priority=priority, due=due), NOW, relation, via).reasons)
    assert seen - {REASON_DUE_UNREADABLE} == used


def test_a_goal_is_scored_by_its_own_target_date_and_priority():
    goal = SimpleNamespace(id="g1", title="Finish the servo prototype", target_date="2026-09-29", priority=Priority.HIGH, created_at="2026-09-01")

    c = score_goal(goal, NOW)

    assert c.kind == "goal" and c.score == W.TIME_TODAY + W.PRIORITY_HIGH and c.reasons == (S.REASON_TIME_TODAY, S.REASON_PRIORITY_HIGH)


def test_a_named_task_outranks_everything_that_is_merely_urgent_or_goal_related():
    """The repo's own rule - what the user explicitly named dominates - holds for these weights: the weakest named task beats the strongest
    unnamed one, whatever stacks up on it."""
    weakest_named = score_task(task(priority=TaskPriority.LOW), NOW, RELATION_NAMED).score
    strongest_goal_related = score_task(task(priority=TaskPriority.HIGH, due="2026-09-27"), NOW, RELATION_GOAL, "g").score
    strongest_unrelated = score_task(task(priority=TaskPriority.HIGH, due="2026-09-29T08:30:00+04:00"), NOW).score

    assert weakest_named > strongest_goal_related and weakest_named > strongest_unrelated


def test_overdue_is_scored_high_never_dropped():
    overdue = score_task(task(due="2025-01-01"), NOW)
    tomorrow = score_task(task(due="2026-09-30"), NOW)
    assert overdue.score > tomorrow.score > score_task(task(due="2026-10-03"), NOW).score > score_task(task(due="2027-01-01"), NOW).score


# --- ordering and tie-breaks -------------------------------------------------------------------------------------------------------


def test_rank_is_highest_score_first():
    assert [c.ref for c in rank([cand("a", 10), cand("b", 30), cand("c", 20)])] == ["b", "c", "a"]


def test_equal_scores_break_on_the_earlier_due_moment_with_no_due_date_last():
    early, late = 1000.0, 2000.0
    ranked = rank([cand("none", 30), cand("late", 30, late), cand("early", 30, early)])
    assert [c.ref for c in ranked] == ["early", "late", "none"]


def test_equal_score_and_due_break_on_the_older_task_then_the_id():
    ranked = rank([
        cand("t-c", 30, 1.0, "2026-09-03"),
        cand("t-b", 30, 1.0, "2026-09-02"),
        cand("t-a", 30, 1.0, "2026-09-02"),
        cand("t-z", 30, 1.0, "2026-09-01"),
    ])
    assert [c.ref for c in ranked] == ["t-z", "t-a", "t-b", "t-c"]


def test_the_order_never_depends_on_the_order_the_store_returned_things_in():
    items = [cand("a", 30, 1.0, "2026-09-02"), cand("b", 30, 1.0, "2026-09-02"), cand("c", 30, None, "2026-09-01"), cand("d", 31),
             cand("e", 30, 0.5, "2026-09-05"), cand("f", 5)]
    expected = [c.ref for c in rank(items)]
    for order in itertools.islice(itertools.permutations(items), 120):
        assert [c.ref for c in rank(list(order))] == expected
    shuffled = list(items)
    random.Random(7).shuffle(shuffled)
    assert [c.ref for c in rank(shuffled)] == expected


def test_a_real_world_ordering_overdue_high_then_due_today_high_then_due_today_medium():
    tasks = [
        task("C", TaskPriority.MEDIUM, "2026-09-29", tid="c"),
        task("A", TaskPriority.HIGH, "2026-09-27", tid="a"),
        task("B", TaskPriority.HIGH, "2026-09-29", tid="b"),
        task("D", TaskPriority.HIGH, None, tid="d"),
    ]
    ranked = rank([score_task(t, NOW) for t in tasks])

    assert [c.text for c in ranked] == ["A", "B", "C", "D"] and [c.score for c in ranked] == [36.0, 32.0, 28.0, 8.0]


# --- which project a phrase means ----------------------------------------------------------------------------------------------------


def projects(*names):
    return [SimpleNamespace(id=f"p{i}", name=name) for i, name in enumerate(names)]


def names(found):
    return [p.name for p in found]


def test_a_unique_best_match_is_returned_alone():
    ps = projects("Arduino Weather Station", "Arduino Robot Arm", "Thesis")
    assert names(match_projects(("arduino", "weather"), ps)) == ["Arduino Weather Station"]  # two shared words beat one


def test_equal_best_matches_are_all_returned_so_the_caller_must_ask():
    ps = projects("Arduino Weather Station", "Arduino Robot Arm", "Thesis")
    assert names(match_projects(("arduino",), ps)) == ["Arduino Weather Station", "Arduino Robot Arm"]


def test_matching_is_by_whole_words_never_by_substring():
    ps = projects("Smart home", "AI research", "Training log")
    assert match_projects(("art",), ps) == []  # "art" is not "smart"
    assert match_projects(("train",), ps) == []
    assert names(match_projects(("research",), ps)) == ["AI research"]  # a real word still matches


def test_a_project_whose_name_has_no_distinctive_word_is_never_matched():
    assert match_projects(("arduino",), projects("My Project", "Stuff", "AI")) == []


def test_no_overlap_means_no_match():
    assert match_projects(("arduino",), projects("Thesis", "Garden")) == [] and match_projects(("arduino",), []) == []
