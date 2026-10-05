"""
Scheduled Jobs V1 / Slice 3 - the scheduler: due discovery, the runtime gates, launch through the Slice 2 adapter, and settlement.

The scheduler only DECIDES whether an occurrence becomes a Job and records what became of it. The Job it makes is an ordinary one that the
real JobEngine plans and runs through the real ToolRegistry, so these tests drive the real engine, the real stores (including independent
SQLite connections) and the real adapter, with injected clocks, switches and device answers. The invariant under all of them:

    one occurrence -> at most one launch -> at most one terminal settlement.
"""

import threading
from datetime import datetime, timedelta, timezone

import pytest

from app.devices.base import TrustState
from app.devices.registry import DeviceRegistry
from app.devices.store import SQLiteDeviceStore
from app.orchestration.base import TERMINAL_JOB_STATUSES, TIMED_OUT_ERROR, JobStatus
from app.orchestration.engine import _ACTIVE_STATUSES
from app.reminders.base import Recurrence, RecurrenceUnit
from app.scheduled_execution.scheduler import IN_FLIGHT_JOB_STATUSES, ScheduledJobScheduler
from app.scheduled_jobs.base import (
    RESULT_CANCELLED,
    RESULT_COMPLETED,
    RESULT_FAILED,
    RunOutcome,
    ScheduledJobError,
    ScheduledJobStatus as S,
    StatusReason as R,
)
from app.scheduled_jobs.base import OPEN_STATUSES
from app.scheduled_jobs.policy import Limits, next_future_due
from app.scheduling.runs import occurrence_key
from app.scheduling.schedule import Schedule, next_due
from app.tools.base import PermissionLevel
from test_orchestration_engine import step
from test_run_profiles import READ, SAFE, FakeTool
from test_scheduled_execution_adapter import (
    APPROVED_AT, DUBAI, FIRST, INTENT, LIFETIME_END, OBJECTIVE, RESEARCH_OBJECTIVE, WEEKLY, Rig,
)

UTC = timezone.utc
HOURLY = Schedule.every(DUBAI, Recurrence(RecurrenceUnit.HOURS, 1), FIRST)
ANALYSIS_TOOLS = ("web_search", "observe_page", "verify_observation", "analyze", "list_tasks")


def lim(**overrides):
    """Limits with the two GLOBAL limits (daily cap, in-flight) out of the way, so a test about one control is not about another."""
    return Limits(**{"max_concurrent": 50, "max_runs_per_day": 1000, **overrides})


def hour(n, minutes=1):
    """The n-th hourly occurrence (n = 0 is FIRST = 14:00 UTC on 2026-10-05), `minutes` after it is due."""
    return FIRST + timedelta(hours=n, minutes=minutes)


class SRig(Rig):
    """The Slice 2 rig (real stores, registry, adapter, JobEngine) plus a scheduler with an injectable kill switch and device answers."""

    def __init__(self, path=":memory:", *, limits=None, enabled=True, trusted=(), now=None, **kwargs):
        super().__init__(path, **kwargs)
        if now is not None:
            self.clock.now = now
        self.limits = limits or lim()  # the global limits have their own tests; the rest are not about them
        self.switch = {"on": enabled}
        self.trusted = set(trusted)
        self.scheduler = self.new_scheduler()

    def new_scheduler(self, **override):
        arguments = dict(
            scheduled_jobs=self.store, jobs=self.jobs, adapter=self.adapter, limits=self.limits,
            enabled=lambda: self.switch["on"], device_trusted=lambda device_id: device_id in self.trusted, clock=self.clock,
        )
        arguments.update(override)
        return ScheduledJobScheduler(**arguments)

    def tick(self):
        return self.scheduler.tick()

    def schedule(self, schedule=HOURLY, objective=None, title="Hourly check", device=None):
        proposal = self.store.create_proposal(
            title=title, intent_text=INTENT, objective=dict(objective if objective is not None else OBJECTIVE), schedule=schedule,
            proposal_expires_at=APPROVED_AT + timedelta(days=1), created_via_device_id=device, now=APPROVED_AT,
        )
        once = schedule.kind.value == "once"
        return self.store.activate(
            proposal.id, expected_version=proposal.version, approved_at=APPROVED_AT + timedelta(minutes=5),
            approval_interaction_id="int-1", due_at=next_due(schedule, APPROVED_AT + timedelta(minutes=5)),
            expires_at=None if once else LIFETIME_END, via="chat",
        )

    def at(self, moment):
        self.clock.now = moment

    def finish(self, job_id, status=JobStatus.COMPLETED, error=None):
        job = self.jobs.get(job_id)
        job.status, job.error = status, error
        self.jobs.save(job)

    def runs(self, sj):
        return sorted(self.store.runs_for(sj.id, limit=200), key=lambda r: r.scheduled_for)

    def outcomes(self, sj):
        return [r.outcome for r in self.runs(sj)]

    def current(self, sj):
        return self.store.get(sj.id)

    def launched(self, sj):
        return [r for r in self.runs(sj) if r.job_id is not None]


def jobs_in(rig):
    return rig.job_count()


# ====================================== scheduler basics ===========================================================================


def test_a_due_scheduled_job_is_launched_through_the_adapter_as_one_ordinary_pending_job():
    rig = SRig()
    sj = rig.schedule()
    calls = []
    real = rig.adapter.launch
    rig.adapter.launch = lambda scheduled_job, occurrence: calls.append(occurrence) or real(scheduled_job, occurrence)

    result = rig.tick()

    assert result.launched == 1 and len(calls) == 1  # the one thing that makes a Job is the adapter
    [run] = rig.runs(sj)
    assert run.outcome is RunOutcome.CREATED and run.job_id and run.occurrence == occurrence_key(sj.id, FIRST)
    job = rig.jobs.get(run.job_id)
    assert job.status is JobStatus.PENDING and job.run_profile == "unattended_v1" and job.deadline_at is not None
    assert rig.current(sj).due_at == hour(1, 0) and rig.current(sj).runs_total == 1


def test_a_schedule_that_is_not_due_yet_is_ignored():
    rig = SRig(now=FIRST - timedelta(minutes=5))
    sj = rig.schedule()
    result = rig.tick()
    assert result.launched == 0 and rig.runs(sj) == [] and jobs_in(rig) == 0 and rig.current(sj).due_at == FIRST


def test_a_paused_or_cancelled_scheduled_job_is_never_looked_at():
    rig = SRig()
    paused, cancelled = rig.schedule(title="paused"), rig.schedule(title="cancelled")
    rig.store.transition(paused.id, S.PAUSED, R.USER, expected_version=paused.version, via="chat")
    rig.store.transition(cancelled.id, S.CANCELLED, R.USER, expected_version=cancelled.version, via="chat")
    assert rig.tick().launched == 0 and jobs_in(rig) == 0 and rig.ledger_count() == 0


def test_a_one_time_scheduled_job_launches_once_and_completes():
    rig = SRig()
    once = Schedule.once(DUBAI, datetime(2026, 10, 5, 18, 0))  # 14:00 UTC
    sj = rig.schedule(schedule=once)
    rig.tick()
    rig.tick()
    assert jobs_in(rig) == 1 and rig.current(sj).status is S.COMPLETED and rig.current(sj).status_reason is R.ONE_TIME_DONE


def test_a_recurring_scheduled_job_launches_again_at_each_occurrence():
    rig = SRig()
    sj = rig.schedule()
    for n in range(3):
        rig.at(hour(n))
        assert rig.tick().launched == 1
        rig.finish(rig.launched(sj)[-1].job_id)
    assert jobs_in(rig) == 3 and rig.current(sj).due_at == hour(3, 0) and rig.current(sj).status is S.ACTIVE


def test_several_scheduled_jobs_are_each_decided_on_their_own():
    rig = SRig()
    sjs = [rig.schedule(title=f"job {i}") for i in range(3)]
    result = rig.tick()
    assert result.launched == 3 and jobs_in(rig) == 3
    assert all(len(rig.launched(sj)) == 1 for sj in sjs)


def test_one_scheduled_job_failing_does_not_stop_the_others_in_the_tick():
    rig = SRig()
    bad, good = rig.schedule(title="bad"), rig.schedule(title="good")
    real = rig.adapter.launch

    def launch(scheduled_job, occurrence):
        if scheduled_job.id == bad.id:
            raise RuntimeError("something unexpected")
        return real(scheduled_job, occurrence)

    rig.adapter.launch = launch
    result = rig.tick()
    assert result.errors == 1 and result.launched == 1
    assert rig.launched(good) and rig.runs(bad) == [] and rig.current(bad).due_at == FIRST  # untouched: it is simply still due


def test_an_unexpected_error_is_not_retried_inside_the_tick_and_is_bounded_by_the_catch_up_window():
    rig = SRig()
    sj = rig.schedule(schedule=WEEKLY)  # a sparse schedule: the occurrence at FIRST is the only one in play
    real = rig.adapter.launch
    rig.adapter.launch = lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
    assert rig.tick().errors == 1  # one attempt, once
    rig.adapter.launch = real
    rig.at(FIRST + timedelta(hours=7))  # past the 6-hour catch-up window: it is no longer retried, it is skipped
    rig.tick()
    assert rig.outcomes(sj) == [RunOutcome.SKIPPED_MISSED] and jobs_in(rig) == 0


def test_too_many_runs_in_flight_defers_a_new_one_which_stays_due_and_is_not_dropped():
    rig = SRig(limits=Limits())  # the default concurrency limit of 2
    sjs = [rig.schedule(title=f"job {i}") for i in range(3)]
    result = rig.tick()
    assert result.launched == 2 and result.deferred == 1 and rig.ledger_count() == 2  # the third is not recorded as anything
    waiting = next(sj for sj in sjs if not rig.launched(sj))
    assert rig.current(waiting).due_at == FIRST  # still due, not consumed
    assert rig.tick().launched == 0  # still full
    for sj in sjs:
        if rig.launched(sj):
            rig.finish(rig.launched(sj)[0].job_id)
    assert rig.tick().launched == 1 and len(rig.launched(waiting)) == 1  # decided again once there is room


def test_a_tick_is_bounded_by_its_batch():
    rig = SRig()
    sjs = [rig.schedule(title=f"job {i}") for i in range(5)]
    small = rig.new_scheduler(batch=2)
    assert small.tick().launched == 2
    assert small.tick().launched == 2
    assert small.tick().launched == 1
    assert jobs_in(rig) == 5 and all(len(rig.launched(sj)) == 1 for sj in sjs)


def test_a_store_that_cannot_be_read_never_breaks_the_tick():
    rig = SRig()
    rig.schedule()

    class Broken:
        def list_due(self, *a, **k):
            raise RuntimeError("database is locked")

        def unsettled_runs(self, *a, **k):
            raise RuntimeError("database is locked")

        def list(self, *a, **k):
            raise RuntimeError("database is locked")

    result = rig.new_scheduler(scheduled_jobs=Broken()).tick()
    assert result.errors == 3 and result.launched == 0  # the settle pass, the paused sweep and the due listing each fail alone


# ====================================== launch: the scheduler never executes anything ============================================


def test_the_scheduler_never_executes_a_tool_or_plans_and_the_normal_job_engine_runs_the_job():
    rig = SRig(plans=[[step("s1", "analyze")]])
    sj = rig.schedule()
    rig.tick()
    rig.tick()
    assert rig.registry.executed == [] and rig.planner.calls == []  # nothing ran, nothing was planned: only a PENDING Job exists

    rig.engine.tick()  # the existing JobEngine plans it with the existing planner and runs it through the existing registry
    [run] = rig.launched(sj)
    assert rig.jobs.get(run.job_id).status is JobStatus.COMPLETED
    assert rig.registry.executed == ["analyze"] and len(rig.planner.calls) == 1
    assert rig.planner.calls[0]["tools"] == sorted(ANALYSIS_TOOLS)  # the profile-filtered offer, not the whole registry

    result = rig.tick()  # and the scheduler only then settles it, from the Job's own state
    assert result.settled == 1
    assert rig.store.get_run(sj.id, run.occurrence).result_state == RESULT_COMPLETED


def test_a_scheduled_run_goes_through_the_normal_engine_security_and_cannot_use_a_forbidden_tool():
    rig = SRig(
        tools=[FakeTool("web_search"), FakeTool("analyze"), FakeTool("save_note", SAFE), FakeTool("delete_event", PermissionLevel.DANGEROUS)],
        plans=[[step("s1", "save_note"), step("s2", "delete_event")]],
    )
    sj = rig.schedule()
    rig.tick()
    rig.engine.tick()
    [run] = rig.launched(sj)
    job = rig.jobs.get(run.job_id)
    assert job.status is JobStatus.FAILED and "not permitted" in job.error
    assert rig.registry.executed == []  # the registry was never asked to run either
    rig.tick()
    assert rig.store.get_run(sj.id, run.occurrence).result_state == RESULT_FAILED and rig.current(sj).consecutive_failures == 1


def test_scheduled_runs_never_write_memory_or_organizer_state():
    writes = []
    rig = SRig(
        tools=[FakeTool("web_search"), FakeTool("analyze"), FakeTool("remember", SAFE, on_execute=lambda: writes.append("memory")),
               FakeTool("create_task", SAFE, on_execute=lambda: writes.append("task"))],
        plans=[[step("s1", "remember"), step("s2", "create_task")]],
    )
    rig.schedule()
    rig.tick()
    rig.engine.tick()
    rig.tick()
    assert writes == [] and rig.registry.executed == []


# ====================================== uniqueness: one occurrence -> at most one launch ==========================================


def test_a_second_tick_cannot_launch_the_same_occurrence_again():
    rig = SRig()
    sj = rig.schedule()
    assert rig.tick().launched == 1
    assert rig.tick().launched == 0 and rig.tick().launched == 0
    assert jobs_in(rig) == 1 and rig.ledger_count() == 1 and len(rig.launched(sj)) == 1


def test_two_ticks_racing_on_the_same_occurrence_through_independent_connections_make_exactly_one_job(tmp_path):
    path = str(tmp_path / "orchestration.db")
    first = SRig(path)
    sj = first.schedule()
    rigs = [first] + [SRig(path) for _ in range(5)]  # six schedulers: each its own connection, lock, store and adapter
    barrier = threading.Barrier(len(rigs))
    results = [None] * len(rigs)

    def attempt(i):
        barrier.wait()
        results[i] = rigs[i].tick()

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(len(rigs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(r is not None for r in results)
    assert sum(r.launched for r in results) == 1  # at most one launch
    assert first.job_count() == 1 and first.ledger_count() == 1
    assert len(first.launched(sj)) == 1 and first.current(sj).runs_total == 1


def test_the_loser_of_a_race_inside_a_tick_records_nothing_and_creates_nothing():
    rig = SRig()
    sj = rig.schedule()
    second = rig.new_scheduler()  # a competing scheduler over the same database
    real = rig.adapter.launch
    state = {"first": True}

    def launch(scheduled_job, occurrence):
        if state["first"]:  # a competing scheduler gets there between this tick's decision and its launch
            state["first"] = False
            second.tick()
        return real(scheduled_job, occurrence)

    rig.adapter.launch = launch
    result = rig.tick()
    assert result.launched == 0 and result.lost_races == 1
    assert jobs_in(rig) == 1 and rig.ledger_count() == 1 and len(rig.launched(sj)) == 1


def test_a_restart_cannot_launch_an_occurrence_that_was_already_launched(tmp_path):
    path = str(tmp_path / "orchestration.db")
    before = SRig(path)
    sj = before.schedule()
    assert before.tick().launched == 1
    before.jobs.connection.close()  # the process dies

    after = SRig(path)  # ...restarts at the same moment
    assert after.tick().launched == 0
    assert after.job_count() == 1 and len(after.launched(sj)) == 1
    assert after.store.get(sj.id).due_at == hour(1, 0)  # launch identity survived the restart


def test_a_job_that_exists_always_has_its_ledger_row_and_a_moved_schedule():
    rig = SRig()
    sj = rig.schedule()
    real = rig.adapter.launch

    def crash_after_the_job_is_built(scheduled_job, occurrence):
        raise RuntimeError("process died")  # before the one transaction commits

    rig.adapter.launch = crash_after_the_job_is_built
    rig.tick()
    assert jobs_in(rig) == 0 and rig.ledger_count() == 0 and rig.current(sj).due_at == FIRST  # nothing half-written
    rig.adapter.launch = real
    rig.tick()
    assert jobs_in(rig) == 1 and rig.ledger_count() == 1 and rig.current(sj).due_at == hour(1, 0)  # and then all of it


# ====================================== settlement ==================================================================================


def test_a_finished_run_is_settled_once_from_the_state_its_job_ended_in():
    rig = SRig()
    sjs = {state: rig.schedule(title=state) for state in ("completed", "failed", "cancelled")}
    rig.tick()
    ended = {"completed": JobStatus.COMPLETED, "failed": JobStatus.FAILED, "cancelled": JobStatus.CANCELLED}
    for name, sj in sjs.items():
        rig.finish(rig.launched(sj)[0].job_id, ended[name])
    result = rig.tick()
    assert result.settled == 3
    states = {name: rig.store.get_run(sj.id, rig.launched(sj)[0].occurrence).result_state for name, sj in sjs.items()}
    assert states == {"completed": RESULT_COMPLETED, "failed": RESULT_FAILED, "cancelled": RESULT_CANCELLED}
    assert [rig.current(sjs[n]).consecutive_failures for n in ("completed", "failed", "cancelled")] == [0, 1, 0]


def test_a_run_whose_job_is_still_going_is_left_unsettled():
    rig = SRig()
    sj = rig.schedule()
    rig.tick()
    for status in (JobStatus.PENDING, JobStatus.RUNNING, JobStatus.PAUSED):
        rig.finish(rig.launched(sj)[0].job_id, status)
        assert rig.tick().settled == 0
    assert len(rig.store.unsettled_runs()) == 1


def test_a_job_that_no_longer_exists_is_settled_as_failed_not_left_active_forever():
    rig = SRig()
    sj = rig.schedule()
    rig.tick()
    rig.scheduler = rig.new_scheduler(jobs=type("Gone", (), {"get": staticmethod(lambda job_id: None)})())
    assert rig.tick().settled == 1
    assert rig.store.get_run(sj.id, rig.runs(sj)[0].occurrence).result_state == RESULT_FAILED


def test_settlement_happens_at_most_once_however_many_ticks_or_schedulers_see_it():
    rig = SRig()
    sj = rig.schedule()
    rig.tick()
    rig.finish(rig.launched(sj)[0].job_id, JobStatus.FAILED)
    other = rig.new_scheduler()
    assert rig.tick().settled == 1
    assert other.tick().settled == 0 and rig.tick().settled == 0
    assert rig.current(sj).consecutive_failures == 1  # the failure was counted once, not once per tick


def test_a_scheduler_working_from_a_stale_list_cannot_settle_a_run_a_second_time():
    rig = SRig()
    sj = rig.schedule()
    rig.tick()
    rig.finish(rig.launched(sj)[0].job_id, JobStatus.FAILED)
    stale = rig.store.unsettled_runs()  # scheduler B listed the unsettled runs BEFORE scheduler A settled this one

    class StaleList:
        def __init__(self, inner):
            self._inner = inner

        def unsettled_runs(self, schedule_id=None):
            return stale if schedule_id is None else self._inner.unsettled_runs(schedule_id)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    assert rig.tick().settled == 1  # A settles it
    late = rig.new_scheduler(scheduled_jobs=StaleList(rig.store)).tick()  # B still holds the old list and tries it again
    assert late.settled == 0
    assert rig.current(sj).consecutive_failures == 1  # the failure was counted once
    assert [e.kind for e in rig.current(sj).events].count("run_failed") == 1


def test_independent_connections_settling_one_finished_run_settle_it_exactly_once(tmp_path):
    path = str(tmp_path / "orchestration.db")
    first = SRig(path)
    sj = first.schedule()
    first.tick()
    first.finish(first.launched(sj)[0].job_id, JobStatus.FAILED)
    rigs = [first] + [SRig(path) for _ in range(5)]
    barrier = threading.Barrier(len(rigs))
    results = [None] * len(rigs)

    def attempt(i):
        barrier.wait()
        results[i] = rigs[i].tick()

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(len(rigs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(r.settled for r in results if r is not None) == 1
    assert first.current(sj).consecutive_failures == 1 and not first.store.unsettled_runs()


def test_settlement_and_the_launch_identity_survive_a_restart(tmp_path):
    path = str(tmp_path / "orchestration.db")
    before = SRig(path)
    sj = before.schedule()
    before.tick()
    job_id = before.launched(sj)[0].job_id
    before.jobs.connection.close()  # crash: the Job is not finished, nothing is settled

    middle = SRig(path)
    assert middle.store.get_run(sj.id, occurrence_key(sj.id, FIRST)).job_id == job_id  # the launch identity
    assert len(middle.store.unsettled_runs()) == 1
    middle.finish(job_id, JobStatus.COMPLETED)  # the Job finished (the engine ran it) while the scheduler was down
    middle.jobs.connection.close()

    after = SRig(path)
    assert after.tick().settled == 1  # the first tick after restart settles it from the Job's own state
    assert after.store.get_run(sj.id, occurrence_key(sj.id, FIRST)).result_state == RESULT_COMPLETED
    assert after.tick().settled == 0 and after.job_count() == 1


def test_every_skip_is_its_own_terminal_settlement_and_is_never_settled_again():
    rig = SRig(enabled=False)
    sj = rig.schedule()
    rig.tick()
    [run] = rig.runs(sj)
    assert run.outcome is RunOutcome.SKIPPED_DISABLED and run.job_id is None
    assert rig.store.unsettled_runs() == []  # a skipped row has no result to settle
    with pytest.raises(ScheduledJobError):
        rig.store.settle_run(run.id, RESULT_FAILED, breaker_limit=3)  # and cannot be given one
    assert rig.current(sj).consecutive_failures == 0


# ====================================== the kill switch =============================================================================


def test_with_the_kill_switch_off_nothing_launches_and_the_occurrence_is_consumed_not_queued():
    rig = SRig(enabled=False)
    sj = rig.schedule()
    calls = []
    real = rig.adapter.launch
    rig.adapter.launch = lambda *a: calls.append(a) or real(*a)
    result = rig.tick()
    assert result.launched == 0 and calls == [] and jobs_in(rig) == 0  # the adapter was not even asked
    assert rig.outcomes(sj) == [RunOutcome.SKIPPED_DISABLED]
    now = rig.current(sj)
    assert now.status is S.ACTIVE and now.status_reason is None and now.due_at == hour(1, 0)  # not paused, not cancelled, moved on


def test_the_kill_switch_does_not_touch_a_job_that_is_already_running_and_it_still_settles():
    rig = SRig(plans=[[step("s1", "analyze")]])
    sj = rig.schedule()
    rig.tick()
    rig.switch["on"] = False
    rig.at(hour(0, 5))
    rig.tick()
    job = rig.jobs.get(rig.launched(sj)[0].job_id)
    assert job.status is JobStatus.PENDING  # not cancelled, not failed, not paused
    rig.engine.tick()  # the engine still runs it to the end
    assert rig.jobs.get(job.id).status is JobStatus.COMPLETED
    assert rig.tick().settled == 1  # and the finished run is settled even while scheduling is switched off


def test_re_enabling_does_not_replay_what_came_due_while_it_was_off():
    rig = SRig(enabled=False)
    sj = rig.schedule()
    for n in range(5):  # five occurrences come due while it is off, each seen by a tick
        rig.at(hour(n))
        rig.tick()
    assert rig.outcomes(sj) == [RunOutcome.SKIPPED_DISABLED] * 5 and jobs_in(rig) == 0

    rig.switch["on"] = True
    rig.at(hour(4, 30))
    assert rig.tick().launched == 0 and jobs_in(rig) == 0  # nothing from the backlog
    assert rig.current(sj).due_at == hour(5, 0)


def test_a_long_outage_with_the_switch_off_is_one_skip_not_a_backlog():
    rig = SRig(enabled=False)
    sj = rig.schedule()
    rig.at(FIRST + timedelta(days=3))  # three days of occurrences, no tick in between
    rig.tick()
    assert rig.outcomes(sj) == [RunOutcome.SKIPPED_DISABLED] and jobs_in(rig) == 0
    rig.switch["on"] = True
    assert rig.tick().launched == 0


def test_a_future_occurrence_runs_normally_after_the_switch_is_turned_back_on():
    rig = SRig(enabled=False)
    sj = rig.schedule()
    rig.tick()
    rig.switch["on"] = True
    rig.at(hour(1))
    assert rig.tick().launched == 1
    assert rig.outcomes(sj) == [RunOutcome.SKIPPED_DISABLED, RunOutcome.CREATED] and jobs_in(rig) == 1


def test_a_switch_thrown_while_the_tick_is_deciding_still_stops_the_launch():
    rig = SRig()
    sj = rig.schedule()
    answers = iter([True, False])  # on at the first gate, off just before the launch
    result = rig.new_scheduler(enabled=lambda: next(answers)).tick()
    assert result.launched == 0 and jobs_in(rig) == 0 and rig.outcomes(sj) == [RunOutcome.SKIPPED_DISABLED]


def test_a_kill_switch_that_cannot_be_read_counts_as_off():
    rig = SRig()
    sj = rig.schedule()

    def broken():
        raise RuntimeError("settings unavailable")

    assert rig.new_scheduler(enabled=broken).tick().launched == 0
    assert jobs_in(rig) == 0 and rig.outcomes(sj) == [RunOutcome.SKIPPED_DISABLED]


def test_the_kill_switch_is_only_a_reduction_it_cannot_make_anything_run():
    rig = SRig(enabled=True, tools=[FakeTool("list_tasks")])  # on, but the capability is gone: still blocked
    sj = rig.schedule(objective=RESEARCH_OBJECTIVE)
    assert rig.tick().launched == 0 and rig.outcomes(sj) == [RunOutcome.SKIPPED_BLOCKED]


# ====================================== missed occurrences ==========================================================================


def test_after_a_gap_there_is_exactly_one_catch_up_run_never_a_replay():
    rig = SRig(now=hour(2, 30))  # 14:00, 15:00 and 16:00 all came due while Jarvis was off; it starts at 16:30
    sj = rig.schedule()
    assert rig.tick().launched == 1
    [run] = rig.runs(sj)
    assert run.outcome is RunOutcome.CREATED_LATE and run.late and run.missed_count == 3
    assert run.scheduled_for == FIRST + timedelta(hours=2)  # the most recent missed occurrence is the one identity
    assert rig.current(sj).due_at == hour(3, 0)
    for _ in range(4):
        assert rig.tick().launched == 0  # no storm of the other two
    assert jobs_in(rig) == 1


def test_a_gap_longer_than_the_catch_up_window_is_skipped_once_and_the_schedule_moves_to_the_future():
    rig = SRig(now=FIRST + timedelta(days=3))  # a weekly schedule: its Monday occurrence is three days old, past the 6-hour window
    sj = rig.schedule(schedule=WEEKLY)
    rig.tick()
    rig.tick()
    assert rig.outcomes(sj) == [RunOutcome.SKIPPED_MISSED] and jobs_in(rig) == 0
    assert rig.current(sj).due_at == FIRST + timedelta(days=7) and rig.current(sj).status is S.ACTIVE


def test_after_a_restart_a_missed_occurrence_is_caught_up_once(tmp_path):
    path = str(tmp_path / "orchestration.db")
    before = SRig(path)
    sj = before.schedule()
    before.jobs.connection.close()  # Jarvis is off through 14:00 and 15:00 ...

    after = SRig(path, now=hour(1, 30))  # ... and starts at 15:30
    assert after.tick().launched == 1 and after.tick().launched == 0
    assert after.outcomes(sj) == [RunOutcome.CREATED_LATE] and after.job_count() == 1


def test_a_one_time_occurrence_missed_beyond_the_window_completes_as_missed():
    rig = SRig(now=FIRST + timedelta(days=2))
    sj = rig.schedule(schedule=Schedule.once(DUBAI, datetime(2026, 10, 5, 18, 0)))
    rig.tick()
    assert rig.outcomes(sj) == [RunOutcome.SKIPPED_MISSED] and jobs_in(rig) == 0
    assert rig.current(sj).status is S.COMPLETED and rig.current(sj).status_reason is R.ONE_TIME_MISSED


def test_when_the_approved_period_has_ended_nothing_runs_and_the_scheduled_job_completes():
    rig = SRig(now=LIFETIME_END + timedelta(hours=1))
    sj = rig.schedule()
    rig.tick()
    assert rig.outcomes(sj) == [RunOutcome.SKIPPED_EXPIRED] and jobs_in(rig) == 0
    assert rig.current(sj).status is S.COMPLETED and rig.current(sj).status_reason is R.LIFETIME_ENDED
    assert rig.tick().launched == 0 and rig.ledger_count() == 1  # and it is not decided a second time


def test_the_last_occurrence_inside_the_lifetime_ends_the_scheduled_job_without_a_run_past_it():
    rig = SRig(now=LIFETIME_END - timedelta(minutes=59))  # the occurrence at LIFETIME_END - 1h runs; the next would be at the end
    sj = rig.schedule()
    rig.at(LIFETIME_END - timedelta(hours=1) + timedelta(minutes=1))
    rig.tick()
    assert rig.current(sj).status is S.COMPLETED and rig.current(sj).status_reason is R.LIFETIME_ENDED


# ====================================== overlap =====================================================================================


def test_an_active_job_blocks_the_next_occurrence_which_is_skipped_not_queued_and_the_job_is_left_alone():
    rig = SRig()
    sj = rig.schedule()
    rig.tick()
    job_id = rig.launched(sj)[0].job_id
    before = rig.jobs.get(job_id)

    rig.at(hour(1))  # the next occurrence while the first Job has not finished
    result = rig.tick()
    assert result.launched == 0 and result.skipped == {"skipped_overlap": 1}
    assert rig.outcomes(sj) == [RunOutcome.CREATED, RunOutcome.SKIPPED_OVERLAP] and jobs_in(rig) == 1
    after = rig.jobs.get(job_id)
    assert after.status is before.status and after.version == before.version  # not cancelled, not touched
    assert rig.current(sj).due_at == hour(2, 0)  # the next FUTURE occurrence is evaluated normally

    rig.finish(job_id)
    assert rig.tick().launched == 0  # the skipped occurrence is not queued: finishing the Job does not resurrect it
    rig.at(hour(2))
    assert rig.tick().launched == 1


@pytest.mark.parametrize("status", [JobStatus.PENDING, JobStatus.PLANNING, JobStatus.RUNNING])
def test_a_job_the_engine_is_advancing_blocks_the_next_occurrence(status):
    rig = SRig()
    sj = rig.schedule()
    rig.tick()
    rig.finish(rig.launched(sj)[0].job_id, status)
    rig.at(hour(1))
    assert rig.tick().launched == 0
    assert rig.outcomes(sj) == [RunOutcome.CREATED, RunOutcome.SKIPPED_OVERLAP]


@pytest.mark.parametrize("status", [JobStatus.PAUSED, JobStatus.AWAITING_CONFIRMATION, JobStatus.AWAITING_INPUT])
def test_a_job_nobody_is_advancing_cannot_block_future_occurrences_forever(status):
    rig = SRig()
    sj = rig.schedule()
    rig.tick()
    stuck = rig.launched(sj)[0].job_id
    rig.finish(stuck, status)  # paused by the user, or waiting for an answer / confirmation: the engine will not touch it
    before = rig.jobs.get(stuck)

    for n in (1, 2, 3):  # occurrence after occurrence: each runs (and ends) normally
        rig.at(hour(n))
        assert rig.tick().launched == 1, (status, n)
        rig.finish(rig.launched(sj)[-1].job_id)
    assert rig.outcomes(sj) == [RunOutcome.CREATED] * 4  # never SKIPPED_OVERLAP
    after = rig.jobs.get(stuck)
    assert after.status is status and after.version == before.version  # the waiting Job was left exactly as it was
    rig.tick()  # settles the three that ended
    assert [r.job_id for r in rig.store.unsettled_runs()] == [stuck]  # and it is still unsettled: its real end is still to come


def test_a_job_nobody_is_advancing_holds_no_global_slot_either():
    rig = SRig(limits=Limits(max_runs_per_day=1000))  # the default in-flight limit of 2
    sjs = [rig.schedule(title=name) for name in ("first", "second", "third")]
    rig.tick()  # two launch; the third is deferred by the in-flight limit
    assert rig.tick().launched == 0
    running = [sj for sj in sjs if rig.launched(sj)]
    waiting = next(sj for sj in sjs if not rig.launched(sj))
    rig.finish(rig.launched(running[0])[0].job_id, JobStatus.PAUSED)
    rig.finish(rig.launched(running[1])[0].job_id, JobStatus.AWAITING_INPUT)
    assert rig.tick().launched == 1 and len(rig.launched(waiting)) == 1  # two stuck Jobs do not fill the slots


def test_a_waiting_job_that_later_ends_is_still_settled_once_from_its_real_state():
    rig = SRig()
    sj = rig.schedule()
    rig.tick()
    job_id = rig.launched(sj)[0].job_id
    rig.finish(job_id, JobStatus.PAUSED)
    assert rig.tick().settled == 0
    rig.finish(job_id, JobStatus.CANCELLED)  # the user finally drops it
    assert rig.tick().settled == 1 and rig.tick().settled == 0
    assert rig.store.get_run(sj.id, rig.launched(sj)[0].occurrence).result_state == RESULT_CANCELLED
    assert rig.current(sj).consecutive_failures == 0


def test_the_in_flight_states_are_exactly_the_ones_the_job_engine_advances():
    assert set(IN_FLIGHT_JOB_STATUSES) == set(_ACTIVE_STATUSES)  # the scheduler may not import the engine: this pins them together
    waiting = {JobStatus.PAUSED, JobStatus.AWAITING_CONFIRMATION, JobStatus.AWAITING_INPUT}
    assert set(JobStatus) == set(IN_FLIGHT_JOB_STATUSES) | TERMINAL_JOB_STATUSES | waiting  # every state is classified, none by accident


def test_a_completed_failed_or_cancelled_job_does_not_block_the_next_occurrence():
    for ended in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
        rig = SRig()
        sj = rig.schedule()
        rig.tick()
        rig.finish(rig.launched(sj)[0].job_id, ended)
        rig.at(hour(1))
        assert rig.tick().launched == 1, ended  # the tick settles the finished run first, so it is not mistaken for active
        assert rig.outcomes(sj) == [RunOutcome.CREATED, RunOutcome.CREATED]


def test_a_job_that_finishes_during_the_tick_is_not_treated_as_active():
    rig = SRig()
    sj = rig.schedule()
    rig.tick()
    job_id = rig.launched(sj)[0].job_id
    rig.at(hour(1))

    class FinishingMidTick:
        """The Job is still running when the tick starts settling, and has completed by the time overlap is decided."""

        def __init__(self, inner):
            self.inner, self.calls = inner, 0

        def get(self, requested):
            self.calls += 1
            if self.calls == 2:
                rig.finish(job_id)
            return self.inner.get(requested)

    result = rig.new_scheduler(jobs=FinishingMidTick(rig.jobs)).tick()
    assert result.launched == 1 and rig.outcomes(sj) == [RunOutcome.CREATED, RunOutcome.CREATED]
    assert rig.store.get_run(sj.id, rig.launched(sj)[0].occurrence).result_state == RESULT_COMPLETED  # settled exactly once


def test_a_job_that_is_still_running_at_the_decision_means_a_skip_and_that_skip_is_final():
    rig = SRig()
    sj = rig.schedule()
    rig.tick()
    job_id = rig.launched(sj)[0].job_id
    rig.at(hour(1))
    rig.tick()  # skipped: the Job is still going
    rig.finish(job_id)  # ... and then completes a moment later
    rig.tick()
    assert rig.outcomes(sj) == [RunOutcome.CREATED, RunOutcome.SKIPPED_OVERLAP] and jobs_in(rig) == 1


def test_a_skipped_overlap_is_not_an_execution_and_not_a_failure():
    rig = SRig()
    sj = rig.schedule()
    rig.tick()
    rig.at(hour(1))
    rig.tick()
    now = rig.current(sj)
    assert now.runs_total == 1 and now.consecutive_failures == 0
    assert rig.store.count_runs_since(FIRST - timedelta(days=1), sj.id) == 1


def test_overlap_is_per_scheduled_job_one_does_not_block_another():
    rig = SRig()
    a, b = rig.schedule(title="a"), rig.schedule(title="b")
    rig.tick()
    rig.at(hour(1))
    rig.finish(rig.launched(b)[0].job_id)  # b's run ended, a's is still going
    rig.tick()
    assert rig.outcomes(a) == [RunOutcome.CREATED, RunOutcome.SKIPPED_OVERLAP]
    assert rig.outcomes(b) == [RunOutcome.CREATED, RunOutcome.CREATED]


# ====================================== the failure breaker =========================================================================


def run_and_end(rig, sj, n, status):
    """Occurrence n launches, then its Job ends in `status`, then the scheduler settles it."""
    rig.at(hour(n))
    rig.tick()
    rig.finish(rig.launched(sj)[-1].job_id, status)
    rig.tick()


def test_one_and_two_failures_do_not_pause_the_schedule():
    rig = SRig()
    sj = rig.schedule()
    run_and_end(rig, sj, 0, JobStatus.FAILED)
    assert rig.current(sj).consecutive_failures == 1 and rig.current(sj).status is S.ACTIVE
    run_and_end(rig, sj, 1, JobStatus.FAILED)
    assert rig.current(sj).consecutive_failures == 2 and rig.current(sj).status is S.ACTIVE


def test_three_consecutive_failures_pause_the_schedule_and_it_stops_running():
    rig = SRig()
    sj = rig.schedule()
    for n in range(3):
        run_and_end(rig, sj, n, JobStatus.FAILED)
    now = rig.current(sj)
    assert now.status is S.PAUSED and now.status_reason is R.BREAKER and now.consecutive_failures == 3
    jobs_before = jobs_in(rig)
    for n in range(3, 8):  # no more automatic executions, no silent reset as time passes
        rig.at(hour(n))
        assert rig.tick().launched == 0
    assert jobs_in(rig) == jobs_before == 3 and rig.current(sj).status is S.PAUSED and rig.current(sj).consecutive_failures == 3


def test_a_timeout_counts_as_a_failure_toward_the_breaker_but_only_a_failure_does():
    rig = SRig()
    sj = rig.schedule()
    rig.tick()
    rig.at(hour(0, 20))  # past the 15-minute run deadline: the existing engine fails the Job, the scheduler only settles it
    rig.engine.tick()
    job = rig.jobs.get(rig.launched(sj)[0].job_id)
    assert job.status is JobStatus.FAILED and job.error == TIMED_OUT_ERROR
    rig.tick()
    assert rig.current(sj).consecutive_failures == 1


def test_a_success_resets_the_streak():
    rig = SRig()
    sj = rig.schedule()
    run_and_end(rig, sj, 0, JobStatus.FAILED)
    run_and_end(rig, sj, 1, JobStatus.FAILED)
    run_and_end(rig, sj, 2, JobStatus.COMPLETED)
    assert rig.current(sj).consecutive_failures == 0
    run_and_end(rig, sj, 3, JobStatus.FAILED)
    assert rig.current(sj).consecutive_failures == 1 and rig.current(sj).status is S.ACTIVE  # a fresh streak, not 4


def test_a_cancelled_run_is_neutral_for_the_streak():
    rig = SRig()
    sj = rig.schedule()
    run_and_end(rig, sj, 0, JobStatus.FAILED)
    run_and_end(rig, sj, 1, JobStatus.CANCELLED)
    assert rig.current(sj).consecutive_failures == 1


def test_blocked_skipped_overlap_cap_and_disabled_occurrences_never_extend_the_streak():
    rig = SRig(tools=[FakeTool("list_tasks")])  # RESEARCH objective below: blocked
    sj = rig.schedule(objective=RESEARCH_OBJECTIVE)
    for n in range(5):
        rig.at(hour(n))
        rig.tick()
    assert set(rig.outcomes(sj)) == {RunOutcome.SKIPPED_BLOCKED} and rig.current(sj).consecutive_failures == 0
    rig.switch["on"] = False
    rig.at(hour(6))
    rig.tick()
    assert rig.current(sj).consecutive_failures == 0 and rig.current(sj).status is S.ACTIVE

    overlap = SRig()
    other = overlap.schedule()
    overlap.tick()
    overlap.at(hour(1))
    overlap.tick()
    assert overlap.outcomes(other)[-1] is RunOutcome.SKIPPED_OVERLAP and overlap.current(other).consecutive_failures == 0

    capped = SRig(limits=lim(max_runs_per_schedule_per_day=1))
    third = capped.schedule()
    capped.tick()
    capped.finish(capped.launched(third)[0].job_id)
    capped.at(hour(1))
    capped.tick()
    assert capped.outcomes(third)[-1] is RunOutcome.SKIPPED_CAP and capped.current(third).consecutive_failures == 0


def test_the_breaker_is_checked_at_launch_too_so_a_streak_at_the_limit_never_launches():
    rig = SRig()
    sj = rig.schedule()
    for n in range(3):  # a streak of 3 reached WITHOUT the breaker having tripped (e.g. the limit was raised, then lowered)
        rig.at(hour(n))
        rig.tick()
        rig.finish(rig.launched(sj)[-1].job_id, JobStatus.FAILED)
        for run in rig.store.unsettled_runs():
            rig.store.settle_run(run.id, RESULT_FAILED, breaker_limit=99, via="scheduler")
    assert rig.current(sj).consecutive_failures == 3 and rig.current(sj).status is S.ACTIVE
    rig.at(hour(3))
    assert rig.tick().launched == 0
    assert rig.current(sj).status is S.PAUSED and rig.current(sj).status_reason is R.BREAKER and jobs_in(rig) == 3


def test_an_explicit_resume_starts_a_fresh_streak_and_runs_again_at_the_next_future_occurrence():
    rig = SRig()
    sj = rig.schedule()
    for n in range(3):
        run_and_end(rig, sj, n, JobStatus.FAILED)
    paused = rig.current(sj)
    rig.at(hour(5))
    upcoming = next_future_due(paused.schedule, rig.clock(), paused.expires_at)
    resumed = rig.store.resume(sj.id, next_due_at=upcoming, expected_version=paused.version, via="chat")
    assert resumed.status is S.ACTIVE and resumed.consecutive_failures == 0
    rig.at(upcoming)
    assert rig.tick().launched == 1


def test_the_breaker_belongs_to_the_schedule_not_to_one_job():
    rig = SRig()
    flaky, steady = rig.schedule(title="flaky"), rig.schedule(title="steady")
    for n in range(3):
        rig.at(hour(n))
        rig.tick()
        rig.finish(rig.launched(flaky)[-1].job_id, JobStatus.FAILED)
        rig.finish(rig.launched(steady)[-1].job_id, JobStatus.COMPLETED)
        rig.tick()
    assert rig.current(flaky).status is S.PAUSED and rig.current(steady).status is S.ACTIVE


# ====================================== the execution cap ===========================================================================


def complete_all(rig, sj):
    for run in rig.launched(sj):
        job = rig.jobs.get(run.job_id)
        if job.status is not JobStatus.COMPLETED:
            rig.finish(run.job_id)


def test_ten_executions_in_a_rolling_day_are_allowed_and_the_eleventh_is_skipped_not_queued():
    rig = SRig()
    sj = rig.schedule()
    for n in range(10):
        rig.at(hour(n))
        assert rig.tick().launched == 1, n
        complete_all(rig, sj)
    rig.at(hour(10))
    result = rig.tick()
    assert result.launched == 0 and result.skipped == {"skipped_cap": 1}
    assert len(rig.launched(sj)) == 10 and rig.outcomes(sj)[-1] is RunOutcome.SKIPPED_CAP
    assert rig.current(sj).status is S.ACTIVE and rig.current(sj).due_at == hour(11, 0)  # not queued, not disabled


def test_the_hard_ceiling_holds_whatever_a_caller_or_setting_asks_for():
    rig = SRig(limits=lim(max_runs_per_schedule_per_day=500))
    sj = rig.schedule()
    for n in range(12):
        rig.at(hour(n))
        rig.tick()
        complete_all(rig, sj)
    assert len(rig.launched(sj)) == 10


def test_skipped_occurrences_do_not_count_toward_the_cap():
    rig = SRig(tools=[FakeTool("list_tasks")])  # RESEARCH objective: blocked until the capability returns
    sj = rig.schedule(objective=RESEARCH_OBJECTIVE)
    for n in range(6):  # six blocked occurrences
        rig.at(hour(n))
        rig.tick()
    rig.switch["on"] = False
    rig.at(hour(6))
    rig.tick()  # one disabled
    rig.switch["on"] = True
    rig.registry.register(FakeTool("web_search"))  # the capability returns
    launched = 0
    for n in range(7, 17):  # ten real executions are still all allowed
        rig.at(hour(n))
        launched += rig.tick().launched
        complete_all(rig, sj)
    assert launched == 10
    rig.at(hour(17))
    assert rig.tick().launched == 0 and rig.outcomes(sj)[-1] is RunOutcome.SKIPPED_CAP


def test_failed_and_completed_executions_both_count():
    rig = SRig(limits=lim(max_runs_per_schedule_per_day=4))
    sj = rig.schedule()
    for n, status in enumerate((JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.COMPLETED, JobStatus.FAILED)):
        rig.at(hour(n))
        rig.tick()
        rig.finish(rig.launched(sj)[-1].job_id, status)
    rig.at(hour(4))
    rig.tick()
    assert len(rig.launched(sj)) == 4 and rig.outcomes(sj)[-1] is RunOutcome.SKIPPED_CAP


def test_the_rolling_window_boundary_a_run_exactly_a_day_old_is_outside_it():
    limits = lim(max_runs_per_schedule_per_day=1, catch_up_window=timedelta(days=3))
    rig = SRig(limits=limits, now=FIRST)  # the run happens at exactly 14:00:00
    sj = rig.schedule()
    assert rig.tick().launched == 1
    rig.finish(rig.launched(sj)[0].job_id)
    created = rig.launched(sj)[0].created_at
    assert created == FIRST

    rig.at(FIRST + timedelta(days=1) - timedelta(hours=1))  # 13:00 next day: 23 hours later - the run is inside the window
    assert rig.tick().launched == 0 and rig.outcomes(sj)[-1] is RunOutcome.SKIPPED_CAP

    rig.at(FIRST + timedelta(days=1) - timedelta(seconds=1))  # 13:59:59 - still inside, but nothing is due yet
    assert rig.tick().launched == 0

    rig.at(FIRST + timedelta(days=1))  # 14:00:00 next day: exactly 24 hours - outside
    assert rig.tick().launched == 1
    assert len(rig.launched(sj)) == 2


def test_the_global_daily_cap_from_slice_1_still_defers_across_all_scheduled_jobs():
    rig = SRig(limits=lim(max_runs_per_day=3))  # 3 actual runs per rolling day ACROSS Scheduled Jobs; each is far below its own cap
    sjs = [rig.schedule(title=f"job {i}") for i in range(5)]
    result = rig.tick()
    assert result.launched == 3 and result.deferred == 2 and rig.ledger_count() == 3  # the others are not recorded as anything
    waiting = [sj for sj in sjs if not rig.launched(sj)]
    assert len(waiting) == 2 and all(rig.current(sj).due_at == FIRST for sj in waiting)  # still due: deferred, not consumed
    assert RunOutcome.SKIPPED_CAP.value not in {r.outcome.value for sj in sjs for r in rig.runs(sj)}  # a global deferral is not a per-Schedule skip
    for sj in sjs:
        for run in rig.launched(sj):
            rig.finish(run.job_id)
    assert rig.tick().launched == 0  # still inside the window

    rig.at(FIRST + timedelta(days=1, minutes=1))  # a day later the old runs have left the window: capacity returns by itself
    assert rig.tick().launched == 3


def test_a_global_deferral_is_not_a_queue_once_it_is_too_late_it_is_skipped_as_missed():
    rig = SRig(limits=lim(max_runs_per_day=1))
    first, second = rig.schedule(schedule=WEEKLY, title="first"), rig.schedule(schedule=WEEKLY, title="second")
    assert rig.tick().launched == 1
    waiting = second if not rig.launched(second) else first
    rig.at(FIRST + timedelta(hours=7))  # past the 6-hour catch-up window while the global cap is still used
    rig.tick()
    assert rig.outcomes(waiting) == [RunOutcome.SKIPPED_MISSED] and rig.current(waiting).due_at == FIRST + timedelta(days=7)


def test_a_schedules_own_cap_is_checked_before_the_global_cap():
    rig = SRig(limits=lim(max_runs_per_schedule_per_day=1, max_runs_per_day=1))  # both used at once
    sj = rig.schedule()
    rig.tick()
    rig.finish(rig.launched(sj)[0].job_id)
    rig.at(hour(1))
    result = rig.tick()
    assert result.skipped == {"skipped_cap": 1} and result.deferred == 0  # the specific control wins: skipped, not deferred


def test_capacity_returns_by_itself_and_the_cap_never_disables_the_schedule():
    rig = SRig(limits=lim(max_runs_per_schedule_per_day=2))
    sj = rig.schedule()
    for n in range(2):
        rig.at(hour(n, 0))
        rig.tick()
        complete_all(rig, sj)
    rig.at(hour(2, 0))
    rig.tick()
    assert rig.outcomes(sj)[-1] is RunOutcome.SKIPPED_CAP and rig.current(sj).status is S.ACTIVE
    for n in range(3, 30):  # a day passes
        rig.at(hour(n, 0))
        rig.tick()
        complete_all(rig, sj)
    assert len(rig.launched(sj)) > 2 and rig.current(sj).status is S.ACTIVE  # it ran again once old runs left the window


def test_the_cap_is_per_scheduled_job():
    rig = SRig(limits=lim(max_runs_per_schedule_per_day=1))
    busy = rig.schedule(title="busy")
    rig.tick()
    complete_all(rig, busy)
    rig.at(hour(1))
    other = rig.schedule(title="other")  # approved later: its own count starts at zero, whatever busy has used
    rig.at(hour(1, 2))
    rig.tick()
    assert rig.outcomes(busy)[-1] is RunOutcome.SKIPPED_CAP  # busy used its one in the window
    assert len(rig.launched(other)) == 1  # a different Scheduled Job is not limited by busy's runs


def test_racing_schedulers_at_the_cap_cannot_exceed_it(tmp_path):
    path = str(tmp_path / "orchestration.db")
    first = SRig(path)
    sj = first.schedule()
    for n in range(9):
        first.at(hour(n))
        first.tick()
        complete_all(first, sj)
    first.at(hour(9))
    rigs = [first] + [SRig(path, now=hour(9)) for _ in range(5)]
    barrier = threading.Barrier(len(rigs))
    results = [None] * len(rigs)

    def attempt(i):
        barrier.wait()
        results[i] = rigs[i].tick()

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(len(rigs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(r.launched for r in results) == 1 and len(first.launched(sj)) == 10  # the tenth, once - never an eleventh
    first.finish(first.launched(sj)[-1].job_id)
    first.at(hour(10))
    assert first.tick().launched == 0 and first.outcomes(sj)[-1] is RunOutcome.SKIPPED_CAP


# ====================================== the runtime capability boundary ============================================================


def test_an_unavailable_required_family_is_blocked_not_launched_and_the_schedule_stays_active():
    rig = SRig(tools=[FakeTool("analyze"), FakeTool("list_tasks")])  # web_search (what RESEARCH needs) is gone
    sj = rig.schedule(objective=RESEARCH_OBJECTIVE)
    result = rig.tick()
    assert result.launched == 0 and result.skipped == {"skipped_blocked": 1} and jobs_in(rig) == 0
    [run] = rig.runs(sj)
    assert run.outcome is RunOutcome.SKIPPED_BLOCKED and "research" in run.reason and run.job_id is None
    assert rig.current(sj).status is S.ACTIVE and rig.registry.executed == [] and rig.planner.calls == []
    assert rig.store.unsettled_runs() == []


def test_the_capability_returning_lets_a_later_occurrence_run_normally():
    rig = SRig(tools=[FakeTool("analyze"), FakeTool("list_tasks")])
    sj = rig.schedule(objective=RESEARCH_OBJECTIVE)
    rig.tick()
    rig.registry.register(FakeTool("web_search"))
    assert rig.tick().launched == 0  # the blocked occurrence was consumed: it is not retried or replayed
    rig.at(hour(1))
    assert rig.tick().launched == 1
    assert rig.outcomes(sj) == [RunOutcome.SKIPPED_BLOCKED, RunOutcome.CREATED]


def test_a_tool_unregistered_between_ticks_blocks_the_next_occurrence():
    rig = SRig()
    sj = rig.schedule(objective=RESEARCH_OBJECTIVE)
    rig.tick()
    rig.finish(rig.launched(sj)[0].job_id)
    rig.registry.unregister("web_search")
    rig.at(hour(1))
    rig.tick()
    assert rig.outcomes(sj) == [RunOutcome.CREATED, RunOutcome.SKIPPED_BLOCKED] and jobs_in(rig) == 1


def test_a_required_tool_that_is_no_longer_read_only_is_blocked():
    rig = SRig(tools=[FakeTool("web_search", SAFE), FakeTool("analyze"), FakeTool("list_tasks")])
    sj = rig.schedule(objective=RESEARCH_OBJECTIVE)
    rig.tick()
    assert rig.outcomes(sj) == [RunOutcome.SKIPPED_BLOCKED] and jobs_in(rig) == 0


def test_an_invalid_stored_family_is_blocked():
    rig = SRig()
    sj = rig.schedule(objective=dict(OBJECTIVE, required_families=["time_travel"], goal_family="analysis"))
    rig.tick()
    assert rig.outcomes(sj) == [RunOutcome.SKIPPED_BLOCKED] and jobs_in(rig) == 0
    assert rig.current(sj).status is S.ACTIVE and rig.current(sj).consecutive_failures == 0


def test_the_profile_filtered_tool_set_is_what_the_boundary_is_checked_against_not_the_whole_registry():
    request = "Click the Refresh button on the dashboard page"
    objective = dict(OBJECTIVE, original_request=request, goal_family="computer_workflow", required_families=["computer_workflow"])
    rig = SRig(tools=[FakeTool("observe_page"), FakeTool("verify_observation"), FakeTool("click_mouse", SAFE)])
    sj = rig.schedule(objective=objective)  # click_mouse IS registered and plannable, but the unattended profile does not permit it
    rig.tick()
    assert rig.outcomes(sj) == [RunOutcome.SKIPPED_BLOCKED] and jobs_in(rig) == 0 and rig.registry.executed == []


def test_a_blocked_occurrence_creates_no_job_no_execution_and_touches_no_state_but_its_own_ledger_row():
    rig = SRig(tools=[FakeTool("analyze")])
    sj = rig.schedule(objective=RESEARCH_OBJECTIVE)
    rig.tick()
    now = rig.current(sj)
    assert jobs_in(rig) == 0 and now.runs_total == 0 and now.last_job_id is None and rig.registry.executed == []
    assert rig.ledger_count() == 1 and now.due_at == hour(1, 0)


# ====================================== device trust ================================================================================


def test_a_device_that_is_not_trusted_blocks_only_that_occurrence_and_the_schedule_stays_active():
    rig = SRig(trusted=())
    sj = rig.schedule(device="dev_laptop")
    rig.tick()
    assert jobs_in(rig) == 0 and rig.outcomes(sj) == [RunOutcome.SKIPPED_BLOCKED] and "device" in rig.runs(sj)[0].reason
    now = rig.current(sj)
    assert now.status is S.ACTIVE and now.status_reason is None and now.due_at == hour(1, 0)  # NOT paused: no device policy is invented
    assert now.consecutive_failures == 0 and now.runs_total == 0

    rig.at(hour(1))  # the next occurrence is asked the same question again, normally
    rig.tick()
    rig.at(hour(2))
    rig.tick()
    assert rig.outcomes(sj) == [RunOutcome.SKIPPED_BLOCKED] * 3 and rig.current(sj).status is S.ACTIVE and jobs_in(rig) == 0

    rig.trusted.add("dev_laptop")  # and when the device is trusted again, the next occurrence simply runs
    rig.at(hour(3))
    assert rig.tick().launched == 1
    assert rig.outcomes(sj)[-1] is RunOutcome.CREATED and rig.current(sj).status is S.ACTIVE


def test_a_trusted_device_runs_normally_and_a_scheduled_job_without_a_device_needs_none():
    rig = SRig(trusted={"dev_laptop"})
    with_device, without = rig.schedule(device="dev_laptop", title="with"), rig.schedule(title="without")
    assert rig.tick().launched == 2
    assert rig.launched(with_device) and rig.launched(without)


def test_a_device_check_that_is_unavailable_or_fails_blocks_rather_than_assumes_trust():
    rig = SRig()
    sj = rig.schedule(device="dev_laptop")
    assert rig.new_scheduler(device_trusted=None).tick().launched == 0
    assert rig.outcomes(sj) == [RunOutcome.SKIPPED_BLOCKED]

    other = SRig()
    sj2 = other.schedule(device="dev_laptop")

    def broken(device_id):
        raise RuntimeError("registry unavailable")

    assert other.new_scheduler(device_trusted=broken).tick().launched == 0
    assert other.outcomes(sj2) == [RunOutcome.SKIPPED_BLOCKED] and jobs_in(other) == 0


def test_with_the_real_device_registry_revoking_the_creating_device_blocks_the_next_run():
    rig = SRig()
    registry = DeviceRegistry(SQLiteDeviceStore.for_job_store(rig.jobs))
    device, _credential = registry.register("laptop", "computer")
    registry.trust(device.device_id)

    def trusted(device_id):  # exactly the check main.py wires in
        found = registry.get(device_id)
        return found is not None and found.trust_state is TrustState.TRUSTED

    scheduler = rig.new_scheduler(device_trusted=trusted)
    sj = rig.schedule(device=device.device_id)
    assert scheduler.tick().launched == 1
    rig.finish(rig.launched(sj)[0].job_id)

    registry.revoke(device.device_id)  # revocation is final in the registry
    for n in (1, 2, 3):
        rig.at(hour(n))
        assert scheduler.tick().launched == 0
    assert rig.outcomes(sj) == [RunOutcome.CREATED] + [RunOutcome.SKIPPED_BLOCKED] * 3  # each occurrence blocked on its own...
    assert rig.current(sj).status is S.ACTIVE and rig.current(sj).status_reason is None  # ...and the Schedule is never paused for it


# ====================================== cancellation and the deadline are the engine's, not the scheduler's ========================


def test_a_cancelled_job_is_settled_as_cancelled_and_nothing_retries_or_duplicates_it():
    rig = SRig()
    sj = rig.schedule()
    rig.tick()
    job = rig.jobs.get(rig.launched(sj)[0].job_id)
    rig.engine.cancel_job(job.id, expected_job_version=job.version)  # the existing cancellation mechanism
    result = rig.tick()
    assert result.settled == 1 and result.launched == 0
    run = rig.launched(sj)[0]
    assert rig.store.get_run(sj.id, run.occurrence).result_state == RESULT_CANCELLED
    assert rig.current(sj).consecutive_failures == 0 and jobs_in(rig) == 1  # neutral, and not a retry trigger


def test_the_deadline_is_enforced_by_the_engine_and_the_scheduler_only_settles_the_outcome():
    rig = SRig()
    sj = rig.schedule()
    rig.tick()
    rig.at(hour(0, 20))
    rig.engine.tick()
    [run] = rig.launched(sj)
    assert rig.jobs.get(run.job_id).error == TIMED_OUT_ERROR
    assert rig.tick().settled == 1
    assert rig.store.get_run(sj.id, run.occurrence).result_state == RESULT_FAILED


# ====================================== the approved period is authoritative while paused =========================================


def pause(rig, sj, reason=R.USER):
    return rig.store.transition(sj.id, S.PAUSED, reason, expected_version=rig.current(sj).version, via="chat", now=rig.clock())


def test_a_schedule_paused_before_expiry_is_completed_when_its_period_ends_not_kept_forever():
    rig = SRig(now=hour(0, 30))
    sj = rig.schedule()
    pause(rig, sj)

    rig.at(LIFETIME_END - timedelta(seconds=1))  # still inside the approved period: left alone
    assert rig.tick().expired == 0 and rig.current(sj).status is S.PAUSED

    rig.at(LIFETIME_END)  # the boundary is the same one an active Scheduled Job uses: at the end, not after it
    result = rig.tick()
    assert result.expired == 1 and result.launched == 0 and jobs_in(rig) == 0
    done = rig.current(sj)
    assert done.status is S.COMPLETED and done.status_reason is R.LIFETIME_ENDED
    assert rig.ledger_count() == 0  # a paused one has no occurrence in play: the state and its event are the record
    assert any(e.kind == "completed:lifetime_ended" for e in done.events)
    assert rig.store.count(OPEN_STATUSES) == 0  # it no longer occupies one of the user's Scheduled Job slots
    assert rig.tick().expired == 0  # and is not completed a second time


def test_resuming_after_the_period_has_ended_cannot_produce_an_execution():
    rig = SRig(now=hour(0, 30))
    sj = rig.schedule()
    paused = pause(rig, sj)
    rig.at(LIFETIME_END + timedelta(days=30))  # it was paused before expiry and is touched again a month after it

    for due in (LIFETIME_END - timedelta(hours=1), rig.clock() + timedelta(hours=1)):  # a past time, and a future one
        with pytest.raises(ScheduledJobError):
            rig.store.resume(sj.id, next_due_at=due, expected_version=paused.version, via="chat", now=rig.clock())
    assert rig.current(sj).status is S.PAUSED  # nothing was resumed
    assert rig.tick().launched == 0 and jobs_in(rig) == 0
    assert rig.current(sj).status is S.COMPLETED and rig.current(sj).status_reason is R.LIFETIME_ENDED
    with pytest.raises(ScheduledJobError):  # and a completed Scheduled Job cannot be resumed at all
        rig.store.resume(sj.id, next_due_at=rig.clock() + timedelta(hours=1), expected_version=rig.current(sj).version, via="chat", now=rig.clock())
    assert rig.tick().launched == 0 and jobs_in(rig) == 0


def test_pausing_and_resuming_inside_the_period_does_not_extend_it():
    rig = SRig(now=hour(0, 30))
    sj = rig.schedule()
    paused = pause(rig, sj)
    rig.at(LIFETIME_END - timedelta(days=20))
    upcoming = next_future_due(paused.schedule, rig.clock(), paused.expires_at)
    resumed = rig.store.resume(sj.id, next_due_at=upcoming, expected_version=paused.version, via="chat", now=rig.clock())
    assert resumed.expires_at == LIFETIME_END  # the same end as before it was ever paused
    rig.at(LIFETIME_END + timedelta(minutes=1))  # an occurrence past the end never runs
    rig.tick()
    assert rig.outcomes(sj) and rig.outcomes(sj)[-1] is RunOutcome.SKIPPED_EXPIRED and jobs_in(rig) <= 1
    assert rig.current(sj).status is S.COMPLETED and rig.current(sj).status_reason is R.LIFETIME_ENDED


def test_a_breaker_paused_schedule_also_reaches_the_end_of_its_period():
    rig = SRig()
    sj = rig.schedule()
    for n in range(3):
        run_and_end(rig, sj, n, JobStatus.FAILED)
    assert rig.current(sj).status_reason is R.BREAKER
    rig.at(LIFETIME_END + timedelta(hours=1))
    rig.tick()
    assert rig.current(sj).status is S.COMPLETED and rig.current(sj).status_reason is R.LIFETIME_ENDED


def test_ending_a_paused_period_is_bookkeeping_not_a_launch_so_the_kill_switch_does_not_matter():
    rig = SRig(enabled=False, now=hour(0, 30))
    sj = rig.schedule()
    pause(rig, sj)
    rig.at(LIFETIME_END + timedelta(minutes=1))
    assert rig.tick().expired == 1 and rig.current(sj).status is S.COMPLETED


def test_a_paused_one_time_schedule_has_no_period_to_end_and_is_left_alone():
    rig = SRig(now=hour(0, 30))
    sj = rig.schedule(schedule=Schedule.once(DUBAI, datetime(2026, 12, 1, 18, 0)))
    assert sj.expires_at is None
    pause(rig, sj)
    rig.at(LIFETIME_END + timedelta(days=100))
    assert rig.tick().expired == 0 and rig.current(sj).status is S.PAUSED


# ====================================== regression guards ===========================================================================


def test_the_scheduler_is_constructed_only_with_the_policy_limits():
    rig = SRig()
    with pytest.raises(TypeError):
        ScheduledJobScheduler(scheduled_jobs=rig.store, jobs=rig.jobs, adapter=rig.adapter, limits={"max_runs_per_day": 500})


def test_a_failing_run_is_settled_through_the_normal_job_state_with_no_silent_retry():
    rig = SRig(tools=[FakeTool("web_search", fail=True)], plans=[[step("s1", "web_search")], [step("r1", "web_search")]])
    sj = rig.schedule(objective=RESEARCH_OBJECTIVE)
    rig.tick()
    for _ in range(4):
        rig.engine.tick()
    [run] = rig.launched(sj)
    assert rig.jobs.get(run.job_id).status is JobStatus.FAILED
    rig.tick()
    assert rig.store.get_run(sj.id, run.occurrence).result_state == RESULT_FAILED
    assert jobs_in(rig) == 1 and rig.current(sj).consecutive_failures == 1 and rig.current(sj).status is S.ACTIVE
