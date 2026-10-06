"""
Scheduled Jobs V1 / Slice 5 - Run Now, presenting results, renewal, and the final lifecycle and security review.

The same real chain as Slice 4 (JarvisCore -> door -> service -> real stores -> the existing confirmation -> scheduler -> adapter -> an
ordinary Job run by the real JobEngine under the READ-only profile), plus the real scheduler's `run_now`, and the reporter that tells the user,
once, through a notification router stand-in. Time is a controllable clock (Monday 2026-10-05 14:01 UTC = 18:01 in Dubai).

The invariants under all of them:
    Run Now has no path of its own: the same gates, the same adapter launch, one occurrence -> one launch;
    presenting changes nothing; renewing is a fresh explicit approval that extends nothing; nothing here can do more than a plain run could.
"""

import threading
from datetime import datetime, timedelta, timezone

import pytest

from app.notifications.base import DeliveryResult, NotificationKind, NotificationPriority
from app.orchestration.base import TIMED_OUT_ERROR, JobStatus
from app.orchestration.profiles import UNATTENDED_V1, permitted_tools
from app.scheduled_execution.occurrence import ScheduledOccurrence, manual_occurrence_key
from app.scheduled_execution.scheduler import ManualRefusal, ScheduledJobScheduler
from app.scheduled_jobs.base import (
    DuplicateOccurrenceError,
    RunKind,
    RunOutcome,
    ScheduledJobRun,
    ScheduledJobStatus as S,
    StaleScheduledJobError,
    StatusReason as R,
)
from app.scheduled_jobs.policy import Limits
from app.scheduled_jobs.sqlite_store import SQLiteScheduledJobStore
from app.scheduled_operations.operations import COMMIT_TOOL_NAME
from app.scheduled_operations.presenter import STALE_AFTER, JobFacts, ScheduledJobReporter, facts_from_job, present_run, span
from app.scheduled_operations.service import UNATTENDED_REFUSAL, ScheduledJobRequestError, ScheduledJobService
from app.scheduling.timezones import ZoneChoice
from app.tools.base import PermissionLevel, ToolResultStatus
from test_orchestration_engine import step
from test_run_profiles import FakeTool
from test_scheduled_operations import ASKED, DUBAI, EVERY_MORNING, READ_TOOLS, TOMORROW_5PM, Chain, _when

UTC = timezone.utc
NAME = "robotics"  # what the user calls the scheduled job in EVERY_MORNING
RUN = f"run my {NAME} scheduled job now"
RENEW = f"renew my {NAME} scheduled job"
SEARCH_PLAN = [step("s1", "web_search", {"q": "AI robotics news"})]


class Chain5(Chain):
    """The Slice 4 chain with the scheduler's kill switch and device trust under test control, `run_now` wired into the service (as main.py
    does), and a reporter whose notification router is a recorder."""

    def __init__(self, tmp_path, *, plans=None, **kwargs):
        super().__init__(tmp_path, plans=[SEARCH_PLAN] * 12 if plans is None else plans, **kwargs)
        self.switch = {"on": True}
        self.trusted: set[str] = set()
        self.scheduler = ScheduledJobScheduler(
            scheduled_jobs=self.store, jobs=self.jobs, adapter=self.adapter, limits=self.limits,
            enabled=lambda: self.switch["on"], device_trusted=lambda device_id: device_id in self.trusted, clock=self.clock,
        )
        self.service._run_now = self.scheduler.run_now
        self.service._in_flight = self.scheduler.in_flight
        self.sent: list = []
        self.accept = True

        def notify(user, event):
            self.sent.append(event)
            return DeliveryResult(accepted=self.accept)

        self.facts = lambda job_id: facts_from_job(self.jobs.get(job_id))
        self.ops._job_facts = self.facts
        self.reporter = ScheduledJobReporter(
            scheduled_jobs=self.store, notify=notify, job_facts=self.facts, limits=self.limits, clock=self.clock
        )

    def approval_job(self, scheduled_job_id):
        """The confirmation Job that was asked about exactly this proposal."""
        (found,) = [j for j in self.confirmation_jobs() if j.plan.steps[0].arguments == {"scheduled_job_id": scheduled_job_id}]
        return self.jobs.get(found.id)

    def finish_runs(self):
        """Let the real engine run what is pending, then let the scheduler settle it."""
        self.tick(4)
        self.scheduler.tick()

    def runs(self, sj=None):
        return list(reversed(self.store.runs_for((sj or self.only_scheduled()).id, 100)))

    def due(self, sj=None, minutes=1):
        sj = sj or self.only_scheduled()
        self.clock.now = sj.due_at + timedelta(minutes=minutes)
        return self.scheduler.tick()

    def state_rows(self):
        """Everything a presenter must not change: the Scheduled Jobs, the ledger (except its `reported_at` marker) and the Jobs."""
        with self.jobs.lock:
            c = self.jobs.connection
            return (
                c.execute("SELECT * FROM scheduled_jobs ORDER BY id").fetchall(),
                c.execute(
                    "SELECT id, schedule_id, occurrence, kind, scheduled_for, outcome, job_id, reason, missed_count, late, created_at, "
                    "result_state, settled_at FROM scheduled_job_runs ORDER BY id"
                ).fetchall(),
                c.execute("SELECT * FROM jobs ORDER BY id").fetchall(),
            )


@pytest.fixture
def chain(tmp_path):
    return Chain5(tmp_path)


def another(tmp_path, name, **kwargs):
    """A second, independent chain (its own database) inside one test."""
    folder = tmp_path / name
    folder.mkdir()
    return Chain5(folder, **kwargs)


# ====================================== RUN NOW ===================================================================================


def test_run_now_starts_one_ordinary_Job_through_the_adapter_and_leaves_the_schedule_exactly_where_it_was(chain):
    sj = chain.activate()

    reply = chain.say(RUN)

    assert reply.startswith("Started:") and "running now" in reply and "unchanged - next run Tuesday 6 October, 8:00 AM" in reply
    (job,) = chain.research_jobs()
    assert job.run_profile == "unattended_v1" and job.conversation_id is None and not job.autonomous_permissions
    assert job.goal == sj.intent_text and job.status is JobStatus.PENDING and job.deadline_at is not None
    (run,) = chain.runs(sj)
    assert run.kind is RunKind.MANUAL and run.outcome is RunOutcome.CREATED and run.job_id == job.id and not run.late
    assert run.occurrence == manual_occurrence_key(sj.id, sj.version) == f"{sj.id}@manual:v{sj.version}"
    after = chain.store.get(sj.id)
    assert after.status is S.ACTIVE and after.due_at == sj.due_at and after.expires_at == sj.expires_at  # NOT moved, NOT completed
    assert after.runs_total == 1 and after.last_job_id == job.id
    chain.finish_runs()  # the SAME JobEngine plans and runs it through the registry
    assert chain.jobs.get(job.id).status is JobStatus.COMPLETED and chain.registry.executed.count("web_search") == 1
    assert chain.store.runs_for(sj.id)[0].result_state == "completed"


def test_a_manual_run_does_not_consume_or_disturb_the_next_real_occurrence(chain):
    sj = chain.activate()
    chain.say(RUN)
    chain.finish_runs()
    chain.due(sj)  # the real occurrence comes due as always
    chain.finish_runs()
    kinds = [(r.kind, r.outcome) for r in chain.runs(sj)]
    assert kinds == [(RunKind.MANUAL, RunOutcome.CREATED), (RunKind.SCHEDULED, RunOutcome.CREATED)]
    real = chain.runs(sj)[1]
    assert real.occurrence != chain.runs(sj)[0].occurrence and real.scheduled_for == sj.due_at
    assert chain.store.get(sj.id).due_at == sj.due_at + timedelta(days=1)  # moved on by the REAL occurrence only
    assert len(chain.research_jobs()) == 2


def test_a_manual_identity_can_never_equal_a_scheduled_one_and_changes_when_the_state_changes(chain):
    sj = chain.activate()
    from app.scheduling.runs import occurrence_key

    assert manual_occurrence_key(sj.id, sj.version) != occurrence_key(sj.id, sj.due_at)
    chain.say(RUN)
    chain.finish_runs()
    later = chain.store.get(sj.id)
    assert later.version > sj.version and manual_occurrence_key(later.id, later.version) != manual_occurrence_key(sj.id, sj.version)


def test_two_requests_that_read_the_same_state_share_one_identity_and_only_one_can_launch(chain):
    sj = chain.activate()
    first = ScheduledOccurrence.manual(sj, chain.clock())
    second = ScheduledOccurrence.manual(sj, chain.clock() + timedelta(seconds=5))
    assert first.occurrence == second.occurrence
    chain.adapter.launch(sj, first)
    with pytest.raises((DuplicateOccurrenceError, StaleScheduledJobError)):
        chain.adapter.launch(sj, second)
    assert len(chain.research_jobs()) == 1 and chain.ledger_count() == 1


def test_two_run_now_requests_racing_on_separate_connections_start_exactly_one_run(tmp_path):
    chain = Chain5(tmp_path)
    sj = chain.activate()
    other_store = SQLiteScheduledJobStore.open(chain.path)
    other = ScheduledJobScheduler(
        scheduled_jobs=other_store, jobs=chain.jobs, adapter=chain.adapter, limits=chain.limits, enabled=lambda: True,
        device_trusted=lambda d: True, clock=chain.clock,
    )
    results, errors = [], []
    start = threading.Barrier(2)

    def go(scheduler):
        try:
            start.wait()
            results.append(scheduler.run_now(sj.id))
        except BaseException as exc:  # reported on the main thread
            errors.append(exc)

    threads = [threading.Thread(target=go, args=(s,)) for s in (chain.scheduler, other)]
    [t.start() for t in threads]
    [t.join(timeout=30) for t in threads]

    assert errors == [] and sorted(r.launched for r in results) == [False, True]
    assert len(chain.research_jobs()) == 1 and chain.ledger_count() == 1  # one request -> one launch, nothing corrupted
    (loser,) = [r for r in results if not r.launched]
    assert loser.refusal in (ManualRefusal.OVERLAP, ManualRefusal.CHANGED)


def test_run_now_while_a_run_is_in_flight_is_refused_and_writes_nothing(chain):
    chain.activate()
    chain.say(RUN)
    before = chain.state_rows()
    reply = chain.say(RUN)  # the first is still pending
    assert "its previous run is still going" in reply and "I haven't run" in reply
    assert chain.state_rows() == before and len(chain.research_jobs()) == 1 and chain.ledger_count() == 1


def test_run_now_obeys_the_per_schedule_cap_and_the_global_cap_and_the_in_flight_limit(tmp_path):
    chain = Chain5(tmp_path, limits=Limits(max_runs_per_schedule_per_day=2, max_concurrent=50, max_runs_per_day=1000))
    chain.activate()
    for _ in range(2):
        assert chain.say(RUN).startswith("Started:")
        chain.finish_runs()
    reply = chain.say(RUN)
    assert "it has already run the most it may in 24 hours" in reply and chain.ledger_count() == 2
    chain.clock.advance(hours=24, minutes=1)  # the window rolls: eligible again, nothing was queued
    assert chain.say(RUN).startswith("Started:")

    glob = another(tmp_path, "g", limits=Limits(max_runs_per_day=1, max_concurrent=50))
    glob.activate()
    assert glob.say(RUN).startswith("Started:")
    glob.finish_runs()
    assert "daily limit for scheduled runs across all your scheduled jobs" in glob.say(RUN)

    busy = another(tmp_path, "b", limits=Limits(max_concurrent=1, max_runs_per_day=1000))
    busy.activate()
    busy.say("Research other things every day at 9 AM")
    busy.approve()
    assert busy.say("run my other things scheduled job now").startswith("Started:")
    assert "other scheduled runs are still in progress" in busy.say(RUN)


def test_run_now_obeys_the_kill_switch_and_does_not_consume_anything(chain):
    sj = chain.activate()
    chain.switch["on"] = False
    before = chain.state_rows()
    assert "scheduled jobs are switched off right now" in chain.say(RUN)
    assert chain.state_rows() == before and chain.research_jobs() == []
    chain.switch["on"] = True
    assert chain.say(RUN).startswith("Started:")  # and nothing was queued while it was off: this is a fresh request
    assert len(chain.research_jobs()) == 1 and chain.store.get(sj.id).due_at == sj.due_at


def test_run_now_re_checks_the_runtime_boundary_and_a_refusal_writes_nothing(chain):
    sj = chain.activate()
    for name in READ_TOOLS:
        chain.registry.unregister(name)
    before = chain.state_rows()
    assert "something it needs isn't available to an unattended run right now" in chain.say(RUN)
    assert chain.state_rows() == before and chain.research_jobs() == [] and chain.store.get(sj.id).status is S.ACTIVE


def test_run_now_requires_the_creating_device_to_still_be_trusted(tmp_path):
    chain = Chain5(tmp_path)

    class Context:
        is_authenticated = True
        device_id = "dev-1"

    chain.trusted.add("dev-1")
    chain.say(EVERY_MORNING, device_context=Context())
    chain.approve()
    chain.trusted.clear()  # revoked since
    assert "the device it was set up from isn't trusted any more" in chain.say(RUN)
    assert chain.research_jobs() == [] and chain.store.get(chain.only_scheduled().id).status is S.ACTIVE  # NOT paused: only this request refused
    chain.trusted.add("dev-1")
    assert chain.say(RUN).startswith("Started:")


def test_run_now_failures_count_towards_the_breaker_exactly_like_scheduled_ones(tmp_path):
    chain = Chain5(tmp_path, plans=[[step("s1", "web_search", {"q": "x"})]] * 4)
    chain.registry.get("web_search")._fail = True  # every run's one step fails
    sj = chain.activate()
    for n in range(1, 4):
        assert chain.say(RUN).startswith("Started:")
        chain.tick(6)
        chain.scheduler.tick()
        assert chain.store.get(sj.id).consecutive_failures == n  # each failed manual run extends the streak by one
    state = chain.store.get(sj.id)
    assert state.status is S.PAUSED and state.status_reason is R.BREAKER and [r.result_state for r in chain.runs(sj)] == ["failed"] * 3
    assert "paused after several failed runs in a row, so I haven't run it. Resume it first." in chain.say(RUN)


def test_run_now_refuses_a_schedule_whose_streak_is_at_a_breaker_limit_that_was_tightened_since(tmp_path):
    """Normally the breaker PAUSES a schedule the moment its streak reaches the limit, so a manual run meets a paused one. The guard is for
    the other way in: a limit tightened (a setting) after a streak had built up, leaving an ACTIVE schedule already at the new limit."""
    chain = Chain5(tmp_path, limits=Limits(breaker_failures=2, max_concurrent=50, max_runs_per_day=1000))
    sj = chain.activate()
    with chain.jobs.lock:
        chain.jobs.connection.execute("UPDATE scheduled_jobs SET consecutive_failures = 2 WHERE id = ?", (sj.id,))
        chain.jobs.connection.commit()
    before = chain.state_rows()
    reply = chain.say(RUN)
    assert "has failed 2 runs in a row, so I won't run it again until you've looked at it" in reply
    assert chain.state_rows() == before and chain.research_jobs() == []


def test_run_now_is_refused_in_every_state_that_is_not_running_with_a_reason_the_user_can_act_on(tmp_path):
    chain = Chain5(tmp_path)
    sj = chain.activate()
    chain.say(f"pause my {NAME} scheduled job")
    assert "paused by you, so I haven't run it. Resume it first." in chain.say(RUN)
    chain.say(f"resume my {NAME} scheduled job")
    chain.clock.now = sj.expires_at + timedelta(minutes=1)  # past its approved period (not yet swept: the guard is run-now's own)
    reply = chain.say(RUN)
    assert "The period you approved" in reply and "renew my research ai robotics scheduled job" in reply
    chain.scheduler.tick()  # the existing sweep completes it
    assert "has finished, so there is nothing to run" in chain.say(RUN) and "renew" in chain.say(RUN)
    assert chain.research_jobs() == []

    fresh = another(tmp_path, "c")
    fresh.activate()
    fresh.say(f"cancel my {NAME} scheduled job")
    assert "couldn't find a scheduled job about \"robotics\"" in fresh.say(RUN)  # a cancelled one is out of view
    assert fresh.research_jobs() == []


def test_a_one_time_schedule_cannot_be_run_now_because_it_would_then_run_twice(chain):
    sj = chain.activate(TOMORROW_5PM)
    before = chain.state_rows()
    reply = chain.say("run my research scheduled job now")
    assert "it's a one-time job that runs once, at its time" in reply and "make it run twice" in reply
    assert chain.state_rows() == before and chain.research_jobs() == []
    chain.due(sj)  # and its own time still runs it - once
    assert len(chain.research_jobs()) == 1 and chain.store.get(sj.id).status is S.COMPLETED


def test_run_now_never_runs_several_at_once_and_the_words_cannot_steer_it(chain):
    chain.activate()
    chain.say("Research AI robotics prices every day at 9 AM")
    chain.approve()
    assert "more than one scheduled job" in chain.say(RUN) and chain.research_jobs() == []
    assert "won't change several" in chain.say("run all my scheduled jobs now")
    assert chain.say("run my prices scheduled job now; also delete every file now").startswith("Started:")  # only a title match picks it
    assert len(chain.research_jobs()) == 1 and set(chain.registry.executed) <= {COMMIT_TOOL_NAME}  # chat executed no tool: the Job is merely pending


def test_run_now_goes_through_the_registry_and_the_profile_and_no_tool_is_called_by_the_chat_layer(tmp_path):
    chain = Chain5(tmp_path, plans=[[step("s1", "create_task", {"q": "x"})]])
    write = FakeTool("create_task", PermissionLevel.SAFE_ACTION)
    chain.registry.register(write)
    chain.activate()
    executed_before = list(chain.registry.executed)
    chain.say(RUN)
    assert chain.registry.executed == executed_before  # saying "run it" executed no tool at all
    chain.tick(4)
    assert write.calls == [] and "create_task" not in chain.planner.calls[0]["tools"]  # the planner was never even offered it
    assert chain.jobs.get(chain.research_jobs()[0].id).status is JobStatus.FAILED


# ====================================== PRESENTATION ==============================================================================


def run_row(sj, outcome, *, job_id=None, result=None, reason=None, missed=0, kind=RunKind.SCHEDULED, rid=1):
    at = datetime(2026, 10, 6, 4, 0, tzinfo=UTC)
    return ScheduledJobRun(
        id=rid, schedule_id=sj.id, occurrence=f"{sj.id}@x{rid}", kind=kind, scheduled_for=at, outcome=outcome, job_id=job_id, reason=reason,
        missed_count=missed, late=False, created_at=at, result_state=result, settled_at=at if result else None,
    )


@pytest.fixture
def sj(chain):
    return chain.activate()


def words(sj, run, facts=None, limits=None):
    presented = present_run(sj, run, facts, limits or Limits())
    assert presented is not None
    assert presented.text.split(",")[0] in ("Sir", "Boss")
    return presented


def test_a_finished_run_is_told_in_a_sentence_with_a_pointer_to_the_result_and_no_tool_output(sj):
    done = words(sj, run_row(sj, RunOutcome.CREATED, job_id="j1", result="completed"), JobFacts("completed", seconds=190, result_excerpt="SECRET RESULT"))
    assert 'your scheduled job "research AI robotics news and give me a summary" finished in 3 minutes' in done.text
    assert "Ask me to show it" in done.text and "SECRET RESULT" not in done.text  # what it FOUND is on request, never pushed
    assert done.kind is NotificationKind.COMPLETED and done.priority is NotificationPriority.LOW


def test_a_manual_run_is_called_the_run_you_asked_for(sj):
    done = words(sj, run_row(sj, RunOutcome.CREATED, job_id="j1", result="completed", kind=RunKind.MANUAL), JobFacts("completed", seconds=5))
    assert 'the run you asked for - "research AI robotics news' in done.text and " in " not in done.text.split("finished")[1].split(".")[0]


def test_a_failed_run_says_only_what_is_known_and_never_the_internal_error(sj):
    failed = words(sj, run_row(sj, RunOutcome.CREATED, job_id="j1", result="failed"), JobFacts("failed"))
    assert "ran but couldn't finish." in failed.text and "timed out" not in failed.text  # no reason is invented
    assert failed.kind is NotificationKind.FAILED and failed.priority is NotificationPriority.HIGH
    timeout = words(sj, run_row(sj, RunOutcome.CREATED, job_id="j1", result="failed"), JobFacts("failed", timed_out=True, deadline_seconds=900))
    assert "ran but didn't finish: it timed out after 15 minutes." in timeout.text
    odd = words(sj, run_row(sj, RunOutcome.CREATED, job_id="j1", result="failed"), JobFacts("failed", timed_out=True, deadline_seconds=0))
    assert "timed out after 15 minutes." in odd.text and "under a minute" not in odd.text  # an unusable recorded figure falls back to the configured limit


def test_a_failure_streak_and_the_breaker_pause_are_told_from_the_Scheduled_Jobs_own_record(chain, sj):
    streak = chain.store.get(sj.id)
    with chain.jobs.lock:
        chain.jobs.connection.execute("UPDATE scheduled_jobs SET consecutive_failures = 2 WHERE id = ?", (sj.id,))
        chain.jobs.connection.commit()
    streak = chain.store.get(sj.id)
    assert "That's 2 failed runs in a row; I'll pause it after 3." in words(streak, run_row(streak, RunOutcome.CREATED, job_id="j", result="failed")).text
    paused = chain.store.transition(sj.id, S.PAUSED, R.BREAKER, expected_version=streak.version, via="scheduler")
    assert "That was 2 failed runs in a row, so I've paused it. Resume it when you're ready." in words(paused, run_row(paused, RunOutcome.CREATED, job_id="j", result="failed")).text


@pytest.mark.parametrize(
    "outcome, reason, missed, expected",
    [
        (RunOutcome.SKIPPED_BLOCKED, "device_not_trusted", 0, "couldn't run this time: the device it was set up from isn't trusted any more."),
        (RunOutcome.SKIPPED_BLOCKED, "refused: What this objective needs (research) is not available", 0, "something it needs isn't available to an unattended run right now"),
        (RunOutcome.SKIPPED_MISSED, "too_late_to_catch_up", 0, "was due, but I wasn't running at the time, so it didn't run."),
        (RunOutcome.SKIPPED_MISSED, "too_late_to_catch_up", 3, "(3 runs were missed, and I don't replay them)"),
        (RunOutcome.SKIPPED_EXPIRED, "lifetime_ended", 0, 'has reached the end of its approved period, so it has stopped. Say "renew my'),
    ],
)
def test_every_skipped_outcome_has_its_own_plain_explanation(sj, outcome, reason, missed, expected):
    presented = words(sj, run_row(sj, outcome, reason=reason, missed=missed))
    assert expected in presented.text
    assert presented.kind is NotificationKind.EXPIRED  # the existing vocabulary: an occurrence that lapsed
    assert "refused:" not in presented.text and "What this objective needs" not in presented.text  # the adapter's internals are never repeated
    if outcome is not RunOutcome.SKIPPED_EXPIRED:
        assert "stays scheduled" in presented.text  # it is still ACTIVE: the next time is evaluated normally


ROUTINE = (RunOutcome.SKIPPED_DISABLED, RunOutcome.SKIPPED_OVERLAP, RunOutcome.SKIPPED_CAP)


def test_every_outcome_the_ledger_can_hold_reaches_the_user_cleanly_or_deliberately_not_at_all(sj):
    for outcome in RunOutcome:
        job = outcome in (RunOutcome.CREATED, RunOutcome.CREATED_LATE)
        for result in ("completed", "failed") if job else (None,):
            row = run_row(sj, outcome, job_id="j1" if job else None, result=result, reason="whatever_internal_code" if not job else None)
            presented = present_run(sj, row, JobFacts(result or "completed"), Limits())
            if outcome in ROUTINE:
                assert presented is None, outcome  # a routine scheduler decision: in the ledger, never a notification
                continue
            assert presented is not None, (outcome, result)
            for raw in ("skipped_", "SKIPPED", "None", "{", "}", "RunOutcome", "whatever_internal_code", "Traceback"):
                assert raw not in presented.text, (outcome, raw)
    cancelled = run_row(sj, RunOutcome.CREATED, job_id="j1", result="cancelled")
    assert present_run(sj, cancelled, JobFacts("cancelled"), Limits()) is None  # the user's own choice: nothing to announce
    assert present_run(sj, run_row(sj, RunOutcome.CREATED, job_id="j1"), None, Limits()) is None  # not finished: nothing to say yet


def test_facts_come_from_the_Job_record_and_carry_no_error_text_or_output_into_a_notification(chain):
    sj = chain.activate()
    chain.say(RUN)
    chain.finish_runs()
    job = chain.research_jobs()[0]
    facts = facts_from_job(chain.jobs.get(job.id))
    assert facts.ended_in == "completed" and facts.timed_out is False and facts.seconds is not None and facts.result_excerpt
    assert facts_from_job(None) is None
    # a failed Job whose recorded error is full of internals: only "it timed out" can be known - the text itself is never carried
    bad = chain.jobs.get(job.id)
    bad.status, bad.error = JobStatus.FAILED, "Traceback: secret_tool exploded at C:\\Users\\x\\key.txt"
    assert facts_from_job(bad) == JobFacts("failed", False, facts.seconds, facts.deadline_seconds, None)
    bad.error = TIMED_OUT_ERROR
    assert facts_from_job(bad).timed_out is True
    assert span(None) == "" and span(30) == "under a minute" and span(61) == "1 minute" and span(7200) == "2 hours"


def test_presenting_is_pure_it_changes_no_Job_no_Scheduled_Job_and_no_ledger_row(chain, sj):
    chain.say(RUN)
    chain.finish_runs()
    before = chain.state_rows()
    for run in chain.store.runs_for(sj.id):
        present_run(chain.store.get(sj.id), run, chain.facts(run.job_id), chain.limits)
    assert chain.state_rows() == before


def test_the_reporter_tells_the_user_once_and_changes_only_the_reported_marker(chain, sj):
    chain.say(RUN)
    assert chain.reporter.report() == 0 and chain.sent == []  # the run is still going: nothing final to say
    chain.finish_runs()
    before = chain.state_rows()

    assert chain.reporter.report() == 1
    (event,) = chain.sent
    assert "finished" in event.text and event.job_id == chain.research_jobs()[0].id and event.conversation_id is None
    assert chain.state_rows() == before  # nothing but `reported_at` moved (state_rows leaves that column out)
    assert chain.store.runs_for(sj.id)[0].reported_at is not None

    assert chain.reporter.report() == 0 and len(chain.sent) == 1  # never again
    again = ScheduledJobReporter(scheduled_jobs=chain.store, notify=lambda u, e: chain.sent.append(e), job_facts=chain.facts, limits=chain.limits, clock=chain.clock)
    assert again.report() == 0 and len(chain.sent) == 1  # nor after a restart (a fresh reporter on the same database)


def test_a_notification_nobody_took_stays_unreported_and_is_offered_again_not_lost(chain, sj):
    chain.say(RUN)
    chain.finish_runs()
    chain.accept = False
    assert chain.reporter.report() == 0 and len(chain.sent) == 1 and chain.store.runs_for(sj.id)[0].reported_at is None
    chain.accept = True
    assert chain.reporter.report() == 1 and len(chain.sent) == 2 and chain.store.runs_for(sj.id)[0].reported_at is not None


def test_a_crash_or_a_failing_channel_never_breaks_the_tick_or_the_other_occurrences(chain, sj):
    chain.say(RUN)
    chain.finish_runs()

    def explode(user, event):
        raise RuntimeError("channel down")

    broken = ScheduledJobReporter(scheduled_jobs=chain.store, notify=explode, job_facts=chain.facts, limits=chain.limits, clock=chain.clock)
    assert broken.report() == 0  # never raises
    assert chain.store.runs_for(sj.id)[0].reported_at is None  # and it is still to be told


def test_an_outcome_that_is_stale_by_the_time_it_could_be_told_is_handled_silently(chain, sj):
    chain.say(RUN)
    chain.finish_runs()
    chain.clock.advance(seconds=int(STALE_AFTER.total_seconds()) + 60)  # Jarvis was off for a day
    assert chain.reporter.report() == 0 and chain.sent == []
    assert chain.store.runs_for(sj.id)[0].reported_at is not None  # marked, so it is not retried forever


def test_a_run_the_user_cancelled_is_marked_handled_with_no_announcement(chain, sj):
    chain.say(RUN)
    job = chain.research_jobs()[0]
    cancelled = chain.jobs.get(job.id)
    cancelled.status = JobStatus.CANCELLED
    chain.jobs.save(cancelled)
    chain.scheduler.tick()
    assert chain.reporter.report() == 0 and chain.sent == []
    assert chain.store.runs_for(sj.id)[0].result_state == "cancelled" and chain.store.runs_for(sj.id)[0].reported_at is not None


def skip_days(chain, sj_id, days=1, extra=timedelta(minutes=1)):
    """Let `days` occurrences come due one after another (the scheduler decides each); returns after each tick via the generator of rows."""
    for _ in range(days):
        chain.clock.now = chain.store.get(sj_id).due_at + extra
        chain.scheduler.tick()
        yield chain.reporter.report()


def test_a_schedule_skipped_because_scheduled_jobs_are_switched_off_never_notifies(chain):
    sj = chain.activate()
    chain.switch["on"] = False
    assert list(skip_days(chain, sj.id, 3)) == [0, 0, 0] and chain.sent == []
    rows = chain.runs(sj)
    assert [r.outcome for r in rows] == [RunOutcome.SKIPPED_DISABLED] * 3  # the ledger keeps every one of them...
    assert all(r.reported_at is not None for r in rows)  # ...and they are marked handled, so they are not retried or piled up
    assert "skipped (scheduled jobs were switched off)" in chain.say(f"show my {NAME} scheduled job")  # and "show" still explains the last one


def test_a_previous_run_still_going_never_notifies_it_is_followed_by_that_runs_own_result(chain):
    sj = chain.activate()
    chain.say(RUN)  # a run in flight: the Job is pending and nobody ticks the engine
    assert list(skip_days(chain, sj.id, 2)) == [0, 0]
    assert [r.outcome for r in chain.runs(sj)] == [RunOutcome.CREATED, RunOutcome.SKIPPED_OVERLAP, RunOutcome.SKIPPED_OVERLAP]
    assert chain.sent == []  # nothing yet: the manual run is not final
    chain.finish_runs()  # (two days have passed on the clock, so the engine ends that run at its time limit)
    assert chain.reporter.report() == 1 and "timed out after 15 minutes" in chain.sent[0].text  # the RESULT is what is announced


def test_hitting_the_per_schedule_cap_again_and_again_never_notifies_but_results_still_do(tmp_path):
    chain = Chain5(tmp_path, limits=Limits(max_runs_per_schedule_per_day=1, max_concurrent=50, max_runs_per_day=1000))
    sj = chain.activate("Research AI robotics news every 1 hour")
    chain.clock.now = chain.store.get(sj.id).due_at + timedelta(minutes=1)
    chain.scheduler.tick()
    chain.finish_runs()
    assert chain.reporter.report() == 1  # the run that happened
    for _ in range(3):  # the next hourly occurrences are over its daily limit
        chain.clock.now = chain.store.get(sj.id).due_at + timedelta(minutes=1)
        chain.scheduler.tick()
        assert chain.reporter.report() == 0
    caps = [r for r in chain.runs(sj) if r.outcome is RunOutcome.SKIPPED_CAP]
    assert len(caps) == 3 and all(r.reported_at is not None for r in caps) and len(chain.sent) == 1


def test_a_blocked_schedule_is_told_once_when_it_starts_and_again_only_after_something_different_happened(chain):
    sj = chain.activate()
    gone = {name: chain.registry.get(name) for name in READ_TOOLS}
    for name in READ_TOOLS:
        chain.registry.unregister(name)
    assert list(skip_days(chain, sj.id, 3)) == [1, 0, 0]  # told when it began; the same problem again is the same news
    assert "couldn't run this time: something it needs isn't available" in chain.sent[0].text
    assert [r.outcome for r in chain.runs(sj)] == [RunOutcome.SKIPPED_BLOCKED] * 3 and all(r.reported_at for r in chain.runs(sj))

    for tool in gone.values():
        chain.registry.register(tool)  # it works again, and runs
    chain.clock.now = chain.store.get(sj.id).due_at + timedelta(minutes=1)
    chain.scheduler.tick()
    chain.finish_runs()
    assert chain.reporter.report() == 1 and "finished" in chain.sent[-1].text

    for name in READ_TOOLS:
        chain.registry.unregister(name)  # blocked again: a NEW problem after a good run
    assert list(skip_days(chain, sj.id, 2)) == [1, 0]
    assert len(chain.sent) == 3 and "couldn't run this time" in chain.sent[-1].text


def test_missed_occurrences_are_told_once_per_stretch_not_every_time(chain):
    sj = chain.activate()
    assert list(skip_days(chain, sj.id, 2, extra=timedelta(hours=33))) == [1, 0]  # Jarvis was off for days: said once
    assert [r.outcome for r in chain.runs(sj)] == [RunOutcome.SKIPPED_MISSED] * 2
    assert "wasn't running at the time" in chain.sent[0].text and len(chain.sent) == 1


def test_the_end_of_an_approved_period_is_a_state_change_and_is_told_once(chain):
    sj = chain.activate()
    chain.clock.now = sj.expires_at + timedelta(days=1)
    chain.scheduler.tick()
    assert chain.reporter.report() == 1 and 'has reached the end of its approved period' in chain.sent[0].text
    assert chain.reporter.report() == 0 and chain.store.get(sj.id).status is S.COMPLETED


def test_a_failed_run_is_always_told_and_each_finished_run_is_a_result(chain):
    sj = chain.activate()
    for n in range(2):
        chain.say(RUN)
        chain.finish_runs()
        assert chain.reporter.report() == 1
    assert [e.kind for e in chain.sent] == [NotificationKind.COMPLETED] * 2  # results are not rate-limited: they ARE the news


def test_the_engine_no_longer_adds_a_second_generic_announcement_for_a_Job_under_a_profile(tmp_path):
    from app.orchestration.sqlite_store import SQLiteJobStore
    from test_orchestration_engine import FakePlanner, make_engine

    class Router:
        def __init__(self):
            self.events = []

        def notify(self, user, event):
            self.events.append(event)
            return DeliveryResult(accepted=True)

    registry = Chain5(tmp_path).registry
    for profile, announcements in (("unattended_v1", 0), ("interactive", 1)):
        jobs, router = SQLiteJobStore(":memory:"), Router()
        engine = make_engine(jobs, registry, FakePlanner(plans=[SEARCH_PLAN]), notification_router=router)
        job = SQLiteJobStore.build_job("research robotics", run_profile=profile)
        with jobs.lock:
            SQLiteJobStore.insert_job_locked(jobs.connection, job)
            jobs.connection.commit()
        engine.tick()
        engine.tick()
        assert jobs.get(job.id).status is JobStatus.COMPLETED  # it ran either way
        assert len([e for e in router.events if e.kind is NotificationKind.COMPLETED]) == announcements  # the profile Job's owner reports it


# ====================================== RENEWAL ===================================================================================


def renew_and_ask(chain, text=RENEW):
    reply = chain.say(text)
    assert reply.endswith(ASKED), reply
    return reply


def live(chain):
    return [s for s in chain.store.list() if s.status is S.ACTIVE]


def test_a_renewal_before_expiry_is_asked_in_plain_words_and_changes_nothing_until_approved(chain):
    old = chain.activate()
    reply = chain.say(RENEW)
    assert reply.startswith("Here's what I'll renew:") and "This is a fresh approval" in reply and "Nothing renews itself." in reply
    assert "read-only" in reply and "normal security checks" in reply
    assert chain.store.get(old.id) == old  # untouched
    assert len(live(chain)) == 1 and sorted(s.status.value for s in chain.store.list()) == ["active", "proposed"]
    job = [j for j in chain.confirmation_jobs() if j.status is JobStatus.AWAITING_CONFIRMATION][-1]
    (interaction,) = chain.interactions.list_for_job(job.id)
    assert interaction.detail["renews"] == old.id and interaction.detail["origin"] == "scheduled_jobs"


def test_saying_no_to_a_renewal_leaves_the_current_one_exactly_as_it_was(chain):
    old = chain.activate()
    chain.say(RENEW)
    chain.say("no")
    chain.tick(5)
    assert chain.store.get(old.id) == old and len(live(chain)) == 1
    renewal = [s for s in chain.store.list() if s.status is S.PROPOSED][0]
    chain.clock.advance(hours=25)
    chain.service.sweep()
    assert chain.store.get(renewal.id).status is S.EXPIRED and len(live(chain)) == 1


def test_approving_a_renewal_starts_one_fresh_bounded_period_and_ends_the_old_one_atomically(chain):
    old = chain.activate()
    chain.clock.advance(days=60)  # well into the first period
    chain.say(RENEW)
    chain.approve()

    (new,) = live(chain)
    assert new.id != old.id and new.intent_text == old.intent_text and new.objective == old.objective and new.profile == "unattended_v1"
    assert new.expires_at == chain.clock.now + timedelta(days=90)  # from THIS approval - not the old end + 90 days
    assert new.approved_at == chain.clock.now and new.approval_interaction_id and new.runs_total == 0 and new.consecutive_failures == 0
    ended = chain.store.get(old.id)
    assert ended.status is S.CANCELLED and ended.status_reason is R.USER
    assert any(e.kind == "cancelled:user" and e.detail == "renewed by a new approval" for e in ended.events)
    assert "Renewed: " in chain.approval_job(new.id).final_result


def test_a_renewal_near_expiry_works_and_the_listing_offers_it(chain):
    old = chain.activate()
    chain.clock.now = old.expires_at - timedelta(days=2)
    listing = chain.say("what scheduled jobs do I have?")
    assert "its approved period ends" in listing and 'say "renew my research ai robotics scheduled job" to keep it going' in listing
    chain.clock.now = old.expires_at - timedelta(hours=1)
    chain.say(RENEW)
    chain.approve()
    (new,) = live(chain)
    assert new.expires_at == chain.clock.now + timedelta(days=90) and new.expires_at > old.expires_at


def test_nothing_is_offered_while_the_period_is_far_off(chain):
    chain.activate()
    assert "approved period ends" not in chain.say("what scheduled jobs do I have?")


def test_a_renewal_after_the_period_ended_starts_a_new_one_and_the_finished_record_stays_finished(chain):
    old = chain.activate()
    chain.clock.now = old.expires_at + timedelta(days=1)
    chain.scheduler.tick()  # the next occurrence finds the approved period over: SKIPPED_EXPIRED, and the existing completion
    ended = chain.store.get(old.id)
    assert ended.status is S.COMPLETED and ended.status_reason is R.LIFETIME_ENDED
    assert 'its approved period has ended - say "renew my research ai robotics scheduled job" to start a new one' in chain.say("what scheduled jobs do I have?")

    chain.say(RENEW)
    chain.approve()
    (new,) = live(chain)
    assert chain.store.get(old.id).status is S.COMPLETED  # not reopened: terminal states stay final
    assert new.expires_at == chain.clock.now + timedelta(days=90) and new.id != old.id


def test_an_active_schedule_past_its_end_that_the_sweep_has_not_reached_can_still_be_renewed(chain):
    old = chain.activate()
    chain.clock.now = old.expires_at + timedelta(hours=1)  # no tick has run
    chain.say(RENEW)
    chain.approve()
    assert chain.store.get(old.id).status is S.CANCELLED and len(live(chain)) == 1


def test_a_paused_schedule_is_not_renewed_until_it_is_resumed(chain):
    old = chain.activate()
    chain.say(f"pause my {NAME} scheduled job")
    assert "is paused. Resume it first, then renew it" in chain.say(RENEW)
    assert len(chain.confirmation_jobs()) == 1  # only the first approval Job exists: nothing new was asked
    assert [s.status for s in chain.store.list()] == [S.PAUSED] and chain.store.get(old.id).expires_at == old.expires_at
    chain.say(f"resume my {NAME} scheduled job")
    assert chain.say(RENEW).endswith(ASKED)


def test_a_schedule_the_breaker_paused_is_not_renewed_around_the_breaker(chain):
    old = chain.activate()
    paused = chain.store.transition(old.id, S.PAUSED, R.BREAKER, expected_version=old.version, via="scheduler")
    reply = chain.say(RENEW)
    assert "paused after several failed runs in a row" in reply and "Resume it first (that clears the failure count)" in reply
    assert chain.store.get(old.id) == paused and len([s for s in chain.store.list()]) == 1


def test_a_renewal_that_was_proposed_cannot_be_approved_if_the_current_one_is_paused_or_cancelled_meanwhile(chain):
    old = chain.activate()
    chain.say(RENEW)
    renewal = [s for s in chain.store.list() if s.status is S.PROPOSED][0]
    chain.store.transition(old.id, S.PAUSED, R.USER, expected_version=chain.store.get(old.id).version, via="chat")
    chain.say("yes")
    chain.tick()
    assert chain.store.get(renewal.id).status is S.PROPOSED and chain.store.get(old.id).status is S.PAUSED  # whole approval refused
    assert chain.approval_job(renewal.id).status is JobStatus.FAILED
    assert len(live(chain)) == 0


def set_streak(chain, scheduled_job_id, streak):
    with chain.jobs.lock:
        chain.jobs.connection.execute("UPDATE scheduled_jobs SET consecutive_failures = ? WHERE id = ?", (streak, scheduled_job_id))
        chain.jobs.connection.commit()


def test_a_renewal_carries_the_failure_streak_so_renewing_cannot_erase_failure_history(chain):
    old = chain.activate()
    set_streak(chain, old.id, 2)  # two failed runs in a row: one more and the breaker pauses it
    chain.say(RENEW)
    chain.approve()
    (new,) = live(chain)
    assert new.id != old.id and new.consecutive_failures == 2  # the same intent, continuing: its history comes with it

    chain.registry.get("web_search")._fail = True  # the very next run fails too
    assert chain.say(RUN).startswith("Started:")
    chain.tick(6)
    chain.scheduler.tick()
    after = chain.store.get(new.id)
    assert after.consecutive_failures == 3 and after.status is S.PAUSED and after.status_reason is R.BREAKER  # exactly as for the old one


def test_a_renewal_of_a_schedule_with_a_clean_record_starts_clean(chain):
    old = chain.activate()
    chain.say(RUN)
    chain.finish_runs()  # a good run
    chain.say(RENEW)
    chain.approve()
    (new,) = live(chain)
    assert new.consecutive_failures == 0 and chain.store.get(old.id).consecutive_failures == 0


def test_a_renewal_after_the_period_ended_still_carries_a_small_streak(chain):
    old = chain.activate()
    set_streak(chain, old.id, 1)
    chain.clock.now = old.expires_at + timedelta(days=1)
    chain.scheduler.tick()  # the period ended: COMPLETED, lifetime_ended, streak 1 remembered
    assert chain.store.get(old.id).status is S.COMPLETED and chain.store.get(old.id).consecutive_failures == 1
    chain.say(RENEW)
    chain.approve()
    (new,) = live(chain)
    assert new.consecutive_failures == 1


def test_a_schedule_already_at_the_breaker_limit_is_not_renewed_whether_running_or_ended(chain):
    old = chain.activate()
    set_streak(chain, old.id, 3)  # at the limit (the breaker pauses it at the next occurrence)
    reply = chain.say(RENEW)
    assert "had failed 3 runs in a row, which is where I stop running a scheduled job, so I won't renew it as it is" in reply
    assert "schedule it again as a new one" in reply and len(chain.store.list()) == 1
    chain.clock.now = old.expires_at + timedelta(days=1)
    chain.scheduler.tick()  # it ends with that streak on its record
    ended = chain.store.get(old.id)
    assert ended.status is S.COMPLETED and ended.consecutive_failures == 3
    assert "which is where I stop running a scheduled job" in chain.say(RENEW) and len(chain.store.list()) == 1


def test_the_store_carries_the_streak_in_the_same_transaction_as_the_renewal(chain):
    old = chain.activate()
    set_streak(chain, old.id, 2)
    old = chain.store.get(old.id)
    proposal = chain.store.create_proposal(
        title=old.title, intent_text=old.intent_text, objective=old.objective, schedule=old.schedule,
        proposal_expires_at=chain.clock() + timedelta(hours=1), now=chain.clock(),
    )
    new = chain.store.activate_renewal(
        proposal.id, expected_version=proposal.version, replaces=old.id, replaces_version=old.version, approved_at=chain.clock(), approval_interaction_id="int-9",
        due_at=chain.clock() + timedelta(hours=1), expires_at=chain.clock() + timedelta(days=90), via="chat",
    )
    assert new.consecutive_failures == 2 and chain.store.get(old.id).status is S.CANCELLED


def test_a_renewal_whose_old_one_changes_at_the_last_moment_rolls_back_the_new_activation_too(chain):
    old = chain.activate()
    proposal = chain.store.create_proposal(
        title=old.title, intent_text=old.intent_text, objective=old.objective, schedule=old.schedule,
        proposal_expires_at=chain.clock() + timedelta(hours=1), now=chain.clock(),
    )
    real_apply = chain.store._apply

    def old_changed_under_us(current, assignments):  # someone else changed the old one between our read and our write
        if current.id == old.id:
            raise StaleScheduledJobError("changed")
        return real_apply(current, assignments)

    chain.store._apply = old_changed_under_us
    with pytest.raises(StaleScheduledJobError):
        chain.store.activate_renewal(
            proposal.id, expected_version=proposal.version, replaces=old.id, replaces_version=old.version, approved_at=chain.clock(), approval_interaction_id="int-9",
            due_at=chain.clock() + timedelta(hours=1), expires_at=chain.clock() + timedelta(days=90), via="chat",
        )
    chain.store._apply = real_apply
    assert chain.store.get(proposal.id).status is S.PROPOSED and chain.store.get(old.id) == old  # all or nothing: never two live copies


def test_a_cancelled_schedule_cannot_be_renewed_and_the_user_is_told_how_to_get_it_back(chain):
    chain.activate()
    chain.say(f"cancel my {NAME} scheduled job")
    assert "couldn't find a scheduled job about \"robotics\"" in chain.say(RENEW)  # cancelled ones are out of view
    old = chain.only_scheduled()
    with pytest.raises(ScheduledJobRequestError, match="You cancelled .* so there is nothing to renew"):
        chain.service.propose_renewal(old.id)


def test_a_one_time_schedule_has_no_period_to_renew(chain):
    chain.activate(TOMORROW_5PM)
    assert "no period to renew" in chain.say("renew my research scheduled job")
    chain.clock.advance(days=2)
    chain.scheduler.tick()
    old = chain.only_scheduled()
    with pytest.raises(ScheduledJobRequestError, match="one-time scheduled job"):
        chain.service.propose_renewal(old.id)


def test_renewal_uses_the_existing_policy_so_an_invalid_interval_cannot_be_renewed(tmp_path):
    chain = Chain5(tmp_path)
    from datetime import time as dtime

    from app.reminders.base import Recurrence, RecurrenceUnit
    from app.scheduling.schedule import Schedule

    every_30 = Schedule.every(DUBAI, Recurrence(RecurrenceUnit.MINUTES, 30), chain.clock() + timedelta(minutes=30), min_interval=timedelta(minutes=1))
    legacy = chain.store.create_proposal(
        title="old fast one", intent_text="research X", objective=chain.service._understand("research X", None), schedule=every_30,
        proposal_expires_at=chain.clock() + timedelta(days=1), now=chain.clock(),
    )
    live_one = chain.store.activate(
        legacy.id, expected_version=legacy.version, approved_at=chain.clock(), approval_interaction_id="int-1",
        due_at=chain.clock() + timedelta(minutes=30), expires_at=chain.clock() + timedelta(days=90), via="chat",
    )  # created before the 1-hour rule: existing schedules are never touched...
    with pytest.raises(ScheduledJobRequestError, match="at least 60 minutes"):
        chain.service.propose_renewal(live_one.id)  # ...but a RENEWAL is a new decision and meets today's rules
    assert len(chain.store.list()) == 1 and chain.store.get(live_one.id).status is S.ACTIVE


def test_a_renewal_re_checks_the_limits_in_force_when_it_is_approved(chain):
    chain.activate()
    chain.say(RENEW)
    renewal = [s for s in chain.store.list() if s.status is S.PROPOSED][0]
    chain.service.limits = Limits(min_interval=timedelta(hours=30), max_concurrent=50, max_runs_per_day=1000)  # tightened since
    with pytest.raises(ScheduledJobRequestError, match="at least 1800 minutes"):  # the daily schedule is 24 hours apart: below the new minimum
        chain.service.approve(renewal.id, "int-x", renews=live(chain)[0].id)
    assert len(live(chain)) == 1 and chain.store.get(renewal.id).status is S.PROPOSED


def test_the_period_never_exceeds_the_policy_bound_however_often_it_is_renewed(tmp_path):
    chain = Chain5(tmp_path, limits=Limits(lifetime=timedelta(days=30), max_concurrent=50, max_runs_per_day=1000))
    chain.activate()
    for _ in range(3):
        chain.clock.advance(days=10)
        chain.say(RENEW)
        chain.approve()
        (current,) = live(chain)
        assert current.expires_at == current.approved_at + timedelta(days=30)  # one period from each approval, never cumulative
        assert current.expires_at - chain.clock.now <= timedelta(days=30)
    assert len([s for s in chain.store.list() if s.status is S.CANCELLED]) == 3 and len(live(chain)) == 1


def test_a_duplicate_renewal_request_does_not_create_a_second_proposal(chain):
    chain.activate()
    chain.say(RENEW)
    assert "already waiting for your answer" in chain.say(RENEW)
    assert len([s for s in chain.store.list() if s.status is S.PROPOSED]) == 1 and len(chain.confirmation_jobs()) == 2  # the approval + ONE renewal


def test_two_renewal_approvals_racing_activate_it_once_and_end_the_old_one_once(tmp_path):
    chain = Chain5(tmp_path)
    old = chain.activate()
    chain.say(RENEW)
    renewal = [s for s in chain.store.list() if s.status is S.PROPOSED][0]
    second_store = SQLiteScheduledJobStore.open(chain.path)
    second = ScheduledJobService(
        second_store, limits=chain.limits, understand=lambda *a: {}, check_objective=chain.adapter.check_objective, clock=chain.clock,
        zone_resolver=lambda: ZoneChoice(DUBAI, "test"), in_flight=chain.scheduler.in_flight,
    )
    outcomes, errors = [], []
    start = threading.Barrier(2)

    def approve(service):
        try:
            start.wait()
            outcomes.append(service.approve(renewal.id, "int-race", renews=old.id))
        except BaseException as exc:  # reported on the main thread
            errors.append(exc)

    threads = [threading.Thread(target=approve, args=(s,)) for s in (chain.service, second)]
    [t.start() for t in threads]
    [t.join(timeout=30) for t in threads]

    assert errors == [] and sorted(o.already for o in outcomes) == [False, True]
    assert len(live(chain)) == 1 and chain.store.get(old.id).status is S.CANCELLED
    ended = chain.store.get(old.id)
    assert [e.kind for e in ended.events].count("cancelled:user") == 1 and [e.kind for e in chain.store.get(renewal.id).events].count("approved") == 1


def test_two_different_renewals_of_one_schedule_can_never_leave_two_live_copies(chain):
    """Even if a second renewal proposal existed (it cannot be created while the first is open - see the duplicate test), the store would refuse
    the second: the renewed schedule is no longer ACTIVE once the first approval ended it."""
    old = chain.activate()
    chain.say(RENEW)
    first = [s for s in chain.store.list() if s.status is S.PROPOSED][0]
    rival = chain.store.create_proposal(  # forced past the dedupe, directly in the store
        title=old.title, intent_text=old.intent_text, objective=old.objective, schedule=old.schedule,
        proposal_expires_at=chain.clock() + timedelta(hours=1), now=chain.clock(),
    )
    one = chain.service.approve(first.id, "int-1", renews=old.id)
    assert not one.already
    with pytest.raises(ScheduledJobRequestError, match="cannot be renewed"):
        chain.service.approve(rival.id, "int-2", renews=old.id)
    assert chain.store.get(rival.id).status is S.PROPOSED and len(live(chain)) == 1


def test_a_renewal_can_only_replace_the_very_thing_it_renews_never_an_unrelated_schedule(chain):
    mine = chain.activate()
    other = chain.service.propose("research other things", _when("every day at 9 AM")).scheduled_job
    chain.service.approve(other.id, "int-other")
    bystander = chain.store.get(other.id)
    renewal = chain.store.create_proposal(
        title=mine.title, intent_text=mine.intent_text, objective=mine.objective, schedule=mine.schedule,
        proposal_expires_at=chain.clock() + timedelta(hours=1), now=chain.clock(),
    )
    with pytest.raises(ScheduledJobRequestError, match="isn't a renewal of the scheduled job it names"):
        chain.service.approve(renewal.id, "int-1", renews=bystander.id)  # a link pointing at someone else's schedule
    with pytest.raises(ScheduledJobRequestError, match="isn't a renewal of the scheduled job it names"):
        chain.service.approve(renewal.id, "int-1", renews=renewal.id)  # or at itself
    assert chain.store.get(bystander.id) == bystander and chain.store.get(mine.id) == mine
    assert chain.store.get(renewal.id).status is S.PROPOSED and len(live(chain)) == 2


def test_renewal_adds_no_permission_it_is_the_same_intent_under_the_same_read_only_profile(chain):
    old = chain.activate()
    chain.say(RENEW)
    chain.approve()
    (new,) = live(chain)
    assert (new.profile, new.objective, new.intent_text, new.schedule) == (old.profile, old.objective, old.intent_text, old.schedule)
    assert new.created_via_device_id == old.created_via_device_id
    chain.clock.now = new.due_at + timedelta(minutes=1)
    chain.scheduler.tick()
    (job,) = chain.research_jobs()
    assert job.run_profile == "unattended_v1" and not job.autonomous_permissions


def test_a_renewal_is_refused_if_what_it_needs_is_no_longer_available(chain):
    old = chain.activate()
    chain.say(RENEW)
    renewal = [s for s in chain.store.list() if s.status is S.PROPOSED][0]
    for name in READ_TOOLS:
        chain.registry.unregister(name)
    chain.say("yes")
    chain.tick()
    assert chain.store.get(renewal.id).status is S.PROPOSED and chain.store.get(old.id).status is S.ACTIVE
    asked = chain.approval_job(renewal.id)
    assert asked.status is JobStatus.FAILED and UNATTENDED_REFUSAL in (asked.error or "") + (asked.final_result or "")


def test_renewing_at_the_cap_does_not_need_a_second_place_while_the_old_one_is_live(tmp_path):
    chain = Chain5(tmp_path, limits=Limits(max_open=1, max_concurrent=50, max_runs_per_day=1000))
    chain.activate()
    assert chain.say(RENEW).endswith(ASKED)
    chain.approve()
    assert len(live(chain)) == 1


def test_the_atomic_renewal_store_operation_changes_nothing_when_the_old_one_cannot_be_renewed(chain):
    old = chain.activate()
    cancelled = chain.store.transition(old.id, S.CANCELLED, R.USER, expected_version=old.version, via="chat")
    proposal = chain.store.create_proposal(
        title="again", intent_text=old.intent_text, objective=old.objective, schedule=old.schedule,
        proposal_expires_at=chain.clock() + timedelta(hours=1), now=chain.clock(),
    )
    from app.scheduled_jobs.base import InvalidTransitionError

    before = [chain.store.get(s.id) for s in chain.store.list()]
    with pytest.raises(InvalidTransitionError, match="cannot be renewed"):
        chain.store.activate_renewal(
            proposal.id, expected_version=proposal.version, replaces=old.id, replaces_version=cancelled.version, approved_at=chain.clock(), approval_interaction_id="int-9",
            due_at=chain.clock() + timedelta(hours=1), expires_at=chain.clock() + timedelta(days=90), via="chat",
        )
    assert [chain.store.get(s.id) for s in chain.store.list()] == before and chain.store.get(old.id) == cancelled


def start_run(chain):
    assert chain.say(RUN).startswith("Started:")
    return chain.research_jobs()[0]


def set_job_status(chain, job_id, status):
    with chain.jobs.lock:
        chain.jobs.connection.execute("UPDATE jobs SET status = ? WHERE id = ?", (status, job_id))
        chain.jobs.connection.commit()


@pytest.mark.parametrize("status", ["pending", "planning", "running"])
def test_a_renewal_is_refused_while_a_run_is_in_flight_and_nothing_at_all_changes(chain, status):
    old = chain.activate()
    set_job_status(chain, start_run(chain).id, status)  # the states the JobEngine itself advances
    before_old, rows_before, asked = chain.store.get(old.id), chain.state_rows(), len(chain.confirmation_jobs())

    reply = chain.say(RENEW)

    assert "is running right now" in reply and "then ask me to renew it" in reply
    assert chain.store.get(old.id) == before_old and chain.state_rows() == rows_before  # the existing schedule, its ledger and Jobs: untouched
    assert len(chain.store.list()) == 1 and live(chain) == [before_old]  # no new Scheduled Job
    assert len(chain.confirmation_jobs()) == asked  # and nothing was even asked


@pytest.mark.parametrize("status", ["awaiting_confirmation", "paused"])
def test_a_run_nobody_is_advancing_does_not_block_a_renewal_the_scheduler_does_not_count_it_either(chain, status):
    chain.activate()
    set_job_status(chain, start_run(chain).id, status)
    assert chain.say(RENEW).endswith(ASKED)


def test_once_the_run_has_settled_renewal_works_normally_and_carries_the_record(chain):
    old = chain.activate()
    start_run(chain)
    assert "is running right now" in chain.say(RENEW)
    chain.finish_runs()  # the run ends and settles
    assert chain.store.get(old.id).runs_total == 1 and chain.store.runs_for(old.id)[0].result_state == "completed"
    chain.say(RENEW)
    chain.approve()
    (new,) = live(chain)
    assert new.id != old.id and new.consecutive_failures == 0 and chain.store.get(old.id).status is S.CANCELLED


def test_renewing_cannot_dodge_the_breaker_through_a_run_that_is_still_going(chain):
    old = chain.activate()
    set_streak(chain, old.id, 2)  # one more failure and the breaker pauses it
    chain.registry.get("web_search")._fail = True
    start_run(chain)  # a failing run, in flight
    assert "is running right now" in chain.say(RENEW)  # the old escape: renew now, and let its failure settle into a dead record

    chain.tick(6)
    chain.scheduler.tick()  # the run fails and settles: the streak reaches the limit
    tripped = chain.store.get(old.id)
    assert tripped.consecutive_failures == 3 and tripped.status is S.PAUSED and tripped.status_reason is R.BREAKER
    assert "paused after several failed runs in a row" in chain.say(RENEW)  # and a breaker-paused one still cannot be renewed
    assert live(chain) == [] and len(chain.store.list()) == 1


def test_a_run_that_succeeds_while_a_renewal_waits_resets_the_streak_honestly(chain):
    old = chain.activate()
    set_streak(chain, old.id, 2)
    start_run(chain)
    assert "is running right now" in chain.say(RENEW)
    chain.finish_runs()  # it succeeds: the streak is honestly reset by a real success, not by renewing
    assert chain.store.get(old.id).consecutive_failures == 0
    chain.say(RENEW)
    chain.approve()
    assert live(chain)[0].consecutive_failures == 0


def asked_renewal(chain):
    """A renewal that was proposed and whose confirmation the user has approved - not yet committed."""
    old = chain.activate()
    chain.say(RENEW)
    renewal = [s for s in chain.store.list() if s.status is S.PROPOSED][0]
    job = chain.approval_job(renewal.id)
    (interaction,) = chain.interactions.list_for_job(job.id)
    chain.interactions.resolve(interaction.id, expected_version=interaction.version, response="yes")
    return old, renewal, job


def test_a_run_started_after_the_renewal_was_asked_blocks_its_approval_until_it_settles(chain):
    old, renewal, job = asked_renewal(chain)
    start_run(chain)  # the user starts a run between being asked and answering
    before = chain.store.get(old.id)

    result = chain.commit.execute(renewal.id, source_job_id=job.id)

    assert result.status is ToolResultStatus.ERROR and "is running right now" in result.message
    assert chain.store.get(old.id) == before and chain.store.get(renewal.id).status is S.PROPOSED and len(live(chain)) == 1
    chain.finish_runs()  # once it has settled, the very same approval goes through
    assert chain.commit.execute(renewal.id, source_job_id=job.id).status is ToolResultStatus.SUCCESS
    assert chain.store.get(renewal.id).status is S.ACTIVE and chain.store.get(old.id).status is S.CANCELLED and len(live(chain)) == 1


def test_a_run_launched_between_the_two_checks_is_still_caught(chain):
    old, renewal, job = asked_renewal(chain)
    real, calls = chain.scheduler.in_flight, {"n": 0}

    def launches_after_the_first_look(schedule_id):
        calls["n"] += 1
        answer = real(schedule_id)
        if calls["n"] == 1:
            assert chain.scheduler.run_now(old.id).launched  # a run starts right after the first answer ("not running")
        return answer

    chain.service._in_flight = launches_after_the_first_look
    result = chain.commit.execute(renewal.id, source_job_id=job.id)
    assert result.status is ToolResultStatus.ERROR and "is running right now" in result.message  # the second look sees it
    assert chain.store.get(renewal.id).status is S.PROPOSED and chain.store.get(old.id).status is S.ACTIVE


def test_a_run_launched_after_the_last_check_is_caught_by_the_stores_version_check(chain):
    old, renewal, job = asked_renewal(chain)
    real, calls = chain.scheduler.in_flight, {"n": 0}

    def launches_after_the_last_look(schedule_id):
        calls["n"] += 1
        answer = real(schedule_id)  # "not running"...
        if calls["n"] == 2:  # the approval asks twice: this is the second look
            assert chain.scheduler.run_now(old.id).launched  # ...and THEN a run starts, after the last look and before the write
        return answer

    chain.service._in_flight = launches_after_the_last_look
    result = chain.commit.execute(renewal.id, source_job_id=job.id)
    assert result.status is ToolResultStatus.ERROR  # the approval is refused whole...
    assert chain.store.get(renewal.id).status is S.PROPOSED and chain.store.get(old.id).status is S.ACTIVE and len(live(chain)) == 1
    assert len(chain.research_jobs()) == 1  # ...and the run that started is the old schedule's own, still counting toward it


def test_the_store_refuses_a_renewal_whose_old_record_is_no_longer_the_one_that_was_checked(chain):
    old = chain.activate()
    proposal = chain.store.create_proposal(
        title=old.title, intent_text=old.intent_text, objective=old.objective, schedule=old.schedule,
        proposal_expires_at=chain.clock() + timedelta(hours=1), now=chain.clock(),
    )
    chain.say(RUN)  # any run launched (or settled) since the check moves the old record's version
    before = [chain.store.get(s.id) for s in chain.store.list()]
    with pytest.raises(StaleScheduledJobError, match="changed"):
        chain.store.activate_renewal(
            proposal.id, expected_version=proposal.version, replaces=old.id, replaces_version=old.version, approved_at=chain.clock(),
            approval_interaction_id="int-9", due_at=chain.clock() + timedelta(hours=1), expires_at=chain.clock() + timedelta(days=90), via="chat",
        )
    assert [chain.store.get(s.id) for s in chain.store.list()] == before


def test_the_question_is_asked_fail_closed_if_it_cannot_be_answered(chain):
    old = chain.activate()
    chain.service._in_flight = None  # no scheduler to ask
    with pytest.raises(ScheduledJobRequestError, match="is running right now"):
        chain.service.propose_renewal(old.id)

    def broken(schedule_id):
        raise RuntimeError("cannot read")

    chain.service._in_flight = broken
    with pytest.raises(ScheduledJobRequestError, match="is running right now"):
        chain.service.propose_renewal(old.id)
    assert len(chain.store.list()) == 1


def test_the_scheduler_answers_with_the_same_definition_overlap_uses(chain):
    old = chain.activate()
    assert chain.scheduler.in_flight(old.id) is False
    job = start_run(chain)
    assert chain.scheduler.in_flight(old.id) is True
    chain.tick(4)  # the Job ends...
    assert chain.scheduler.in_flight(old.id) is False  # ...and asking settles it, as the tick does
    assert chain.store.runs_for(old.id)[0].result_state == "completed"


# ====================================== LIFECYCLE ==================================================================================


def test_the_whole_lifecycle_create_approve_due_execute_settle_present_and_look_at_the_result(chain):
    reply = chain.say(EVERY_MORNING)  # proposal
    assert reply.endswith(ASKED)
    chain.approve()  # approval -> active
    sj = chain.only_scheduled()
    assert sj.status is S.ACTIVE
    chain.due(sj)  # due -> execution (launched through the adapter)
    chain.finish_runs()  # the real JobEngine runs it; the scheduler settles it
    run = chain.store.runs_for(sj.id)[0]
    assert run.outcome is RunOutcome.CREATED and run.result_state == "completed"
    assert chain.reporter.report() == 1  # presentation
    text = chain.sent[0].text
    assert text.split(",")[0] in ("Sir", "Boss") and 'your scheduled job "research AI robotics news and give me a summary" finished' in text
    detail = chain.say(f"show my {NAME} scheduled job")  # and the result, on request
    assert "Runs so far: 1; last: completed" in detail and "What it found:" in detail
    assert not chain.memory.get_facts()  # nothing was written to memory by any of it


def test_every_management_action_gives_a_coherent_answer_in_every_state(chain):
    sj = chain.activate()
    assert chain.say(f"pause my {NAME} scheduled job").startswith("Paused:")
    assert "is already paused" in chain.say(f"pause my {NAME} scheduled job")
    assert chain.say(f"resume my {NAME} scheduled job").startswith("Resumed:")
    assert "isn't paused" in chain.say(f"resume my {NAME} scheduled job")
    assert chain.say(RUN).startswith("Started:")
    assert "is running right now" in chain.say(RENEW)  # not while the run is going
    chain.finish_runs()
    assert chain.say(RENEW).endswith(ASKED)
    assert chain.say(f"cancel my {NAME} scheduled job").startswith("Cancelled:")
    assert "couldn't find a scheduled job" in chain.say(f"cancel my {NAME} scheduled job")
    assert "couldn't find a scheduled job" in chain.say(f"pause my {NAME} scheduled job")
    assert chain.store.get(sj.id).status is S.CANCELLED


def test_a_proposal_and_its_approval_survive_a_restart_and_a_new_process_sees_the_same_truth(tmp_path):
    first = Chain5(tmp_path)
    sj = first.propose()
    job, interaction = first.pending_confirmation()
    first.interactions.resolve(interaction.id, expected_version=interaction.version, response="yes")

    restarted = Chain5.__new__(Chain5)  # a new process over the same database file
    Chain.__init__(restarted, tmp_path)
    restarted.service._run_now = None
    assert restarted.commit.execute(sj.id, source_job_id=job.id).status is ToolResultStatus.SUCCESS
    assert restarted.store.get(sj.id).status is S.ACTIVE
    assert restarted.commit.execute(sj.id, source_job_id=job.id).status is ToolResultStatus.SUCCESS  # a retry is harmless
    assert len([s for s in restarted.store.list() if s.status is S.ACTIVE]) == 1


def test_a_reporter_restarted_part_way_neither_repeats_nor_loses_anything(chain):
    chain.activate()
    for _ in range(3):  # three finished runs, each told as it finishes
        assert chain.say(RUN).startswith("Started:")
        chain.finish_runs()
        assert chain.reporter.report() == 1
    fresh = ScheduledJobReporter(scheduled_jobs=chain.store, notify=lambda u, e: chain.sent.append(e), job_facts=chain.facts, limits=chain.limits, clock=chain.clock)
    assert fresh.report() == 0 and len(chain.sent) == 3


def test_concurrent_management_requests_leave_one_coherent_state(tmp_path):
    chain = Chain5(tmp_path)
    sj = chain.activate()
    other_store = SQLiteScheduledJobStore.open(chain.path)
    other = ScheduledJobService(
        other_store, limits=chain.limits, understand=lambda *a: {}, check_objective=chain.adapter.check_objective, clock=chain.clock,
        zone_resolver=lambda: ZoneChoice(DUBAI, "test"), run_now=chain.scheduler.run_now,
    )
    outcomes = {"errors": [], "ok": 0}
    start = threading.Barrier(3)

    def act(call):
        try:
            start.wait()
            call()
            outcomes["ok"] += 1
        except ScheduledJobRequestError:
            outcomes["errors"].append(1)
        except BaseException as exc:  # anything else is a bug
            outcomes.setdefault("bugs", []).append(exc)

    threads = [
        threading.Thread(target=act, args=(lambda: chain.service.pause(sj.id),)),
        threading.Thread(target=act, args=(lambda: other.cancel(sj.id),)),
        threading.Thread(target=act, args=(lambda: other.run_now(sj.id),)),
    ]
    [t.start() for t in threads]
    [t.join(timeout=30) for t in threads]

    assert "bugs" not in outcomes
    final = chain.store.get(sj.id)
    assert final.status in (S.PAUSED, S.CANCELLED, S.ACTIVE)
    assert chain.ledger_count() <= 1 and len(chain.research_jobs()) == chain.ledger_count()  # at most one launch, and always with its row
    assert final.version >= sj.version


# ====================================== SECURITY ====================================================================================

FORBIDDEN = [
    ("create_task", PermissionLevel.SAFE_ACTION), ("save_note", PermissionLevel.SAFE_ACTION), ("create_event", PermissionLevel.SAFE_ACTION),
    ("create_reminder", PermissionLevel.SAFE_ACTION), ("remember_fact", PermissionLevel.SAFE_ACTION), ("set_light", PermissionLevel.SAFE_ACTION),
    ("send_email", PermissionLevel.CONFIRM), ("send_message", PermissionLevel.CONFIRM), ("type_text", PermissionLevel.CONFIRM),
    ("click", PermissionLevel.CONFIRM), ("commit_memory_write", PermissionLevel.CONFIRM), ("move_robot", PermissionLevel.CONFIRM),
    ("start_print", PermissionLevel.CONFIRM), ("delete_file", PermissionLevel.DANGEROUS), ("run_shell", PermissionLevel.DANGEROUS),
]


def test_the_unattended_profile_is_exactly_the_approved_read_only_surface():
    assert UNATTENDED_V1.allowed_tools == frozenset(
        {"web_search", "observe_page", "verify_observation", "analyze", "list_tasks", "list_notes", "list_events", "list_reminders"}
    )
    assert UNATTENDED_V1.max_steps <= 6 and UNATTENDED_V1.max_replans <= 1
    for _, level in FORBIDDEN:
        assert level is not PermissionLevel.READ


@pytest.mark.parametrize("name, level", FORBIDDEN)
def test_no_write_communication_computer_device_or_physical_tool_can_be_run_manually_whatever_its_level(tmp_path, name, level):
    chain = Chain5(tmp_path, plans=[[step("s1", name, {"q": "x"})]])
    tool = FakeTool(name, level)
    chain.registry.register(tool)
    chain.activate()
    permitted = {t.name for t in permitted_tools(UNATTENDED_V1, chain.registry)}
    assert name not in permitted and permitted <= UNATTENDED_V1.allowed_tools  # not even offered to the Planner

    assert chain.say(RUN).startswith("Started:")
    chain.tick(5)
    assert tool.calls == [] and name not in chain.registry.executed  # the plan that names it is refused; the tool never runs
    assert chain.jobs.get(chain.research_jobs()[0].id).status is JobStatus.FAILED
    assert name not in chain.planner.calls[0]["tools"]


def test_a_tool_that_is_read_in_the_profile_cannot_be_used_if_it_is_not_read_only_in_the_registry(tmp_path):
    chain = Chain5(tmp_path, plans=[[step("s1", "list_tasks", {"q": "x"})]])
    chain.registry.unregister("list_tasks")
    escalated = FakeTool("list_tasks", PermissionLevel.SAFE_ACTION)  # same NAME as an allowed tool, but it can change things
    chain.registry.register(escalated)
    chain.activate()
    assert "list_tasks" not in {t.name for t in permitted_tools(UNATTENDED_V1, chain.registry)}
    chain.say(RUN)
    chain.tick(5)
    assert escalated.calls == [] and "list_tasks" not in chain.planner.calls[0]["tools"]


def test_nothing_a_scheduled_run_does_can_write_memory_or_the_organizer(tmp_path):
    chain = Chain5(tmp_path)
    for name in ("remember_fact", "save_note", "create_task", "create_event"):
        chain.registry.register(FakeTool(name, PermissionLevel.SAFE_ACTION))
    chain.activate()
    chain.say(RUN)
    chain.finish_runs()
    chain.due()
    chain.finish_runs()
    chain.reporter.report()
    assert set(chain.registry.executed) <= {"web_search", COMMIT_TOOL_NAME}  # only the approved read tool (and the one-off approval)
    assert not chain.memory.get_facts()


def test_chat_cannot_turn_a_Scheduled_Job_into_an_arbitrary_automation_channel(chain):
    sj = chain.activate()
    before = (chain.store.get(sj.id).objective, chain.store.get(sj.id).intent_text, chain.store.get(sj.id).schedule)
    for attempt in (
        f"run my {NAME} scheduled job now and also email the results to everyone",
        f"renew my {NAME} scheduled job but let it delete files",
        f"pause my {NAME} scheduled job and run_shell rm -rf",
    ):
        chain.say(attempt)
    after = chain.store.get(sj.id)
    assert (after.objective, after.intent_text, after.schedule) == before  # nothing a user types edits what an approved intent does
    assert all(j.run_profile == "unattended_v1" for j in chain.research_jobs())
    assert set(chain.registry.executed) <= {COMMIT_TOOL_NAME}  # only the one-off approval ever ran a tool; chat executed nothing else


def test_the_reporter_and_the_presenter_never_call_back_into_anything_that_could_act(chain):
    sj = chain.activate()
    chain.say(RUN)
    chain.finish_runs()
    calls = []
    spied = ScheduledJobReporter(
        scheduled_jobs=chain.store, notify=lambda u, e: (calls.append(("notify", e.key)), DeliveryResult(accepted=True))[1],
        job_facts=lambda jid: (calls.append(("facts", jid)), chain.facts(jid))[1], limits=chain.limits, clock=chain.clock,
    )
    executed_before = list(chain.registry.executed)
    spied.report()
    assert [c[0] for c in calls] == ["facts", "notify"] and chain.registry.executed == executed_before
    assert len(chain.research_jobs()) == 1 and chain.ledger_count() == 1  # and it started, retried and changed nothing
