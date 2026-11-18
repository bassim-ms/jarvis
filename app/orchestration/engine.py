"""
Job Engine (M15): a background poller that turns PENDING Jobs into
validated Plans and RUNNING Jobs into completed/failed ones - the direct
sibling of AutomationEngine (app/automation/engine.py) and
ReminderScheduler (app/reminders/scheduler.py), reusing the exact same
"poll on a daemon thread, isolate every per-item failure, isolate the
poll itself too" shape.

This is the ONE place plan execution actually happens, and it is built
entirely on the existing M5 tool system: `_run_step` only ever calls
`ToolRegistry.execute(tool_name, params=..., confirmed=...)` for an
already-validated PlanStep - never Python/shell directly, never a tool
outside the registry, and never `confirmed=True` for a step whose
permission requires confirmation until an explicit ConfirmJobTool
approval set `step.confirmed = True` first (see `_needs_confirmation`/
`_confirmed_value`). Permission enforcement is entirely ToolRegistry's,
unchanged - this milestone adds a gate in FRONT of it (deciding whether
to even attempt a step this tick), never a bypass behind it.

Core safety discipline - "never claim success when an action actually
failed" - is structural, not a convention: a Job's status can only ever
become COMPLETED in `_advance`'s completion check, which requires every
step to be SUCCEEDED or SKIPPED; a single unresolved FAILED step makes
the whole Job FAILED, with `error` naming the step and reason, no matter
how much of the rest of the plan succeeded.

Restart recovery is mostly a property of persistence alone - exactly
like ReminderScheduler/AutomationEngine, a Job's current status/plan is
just read back from JobStore on the next tick, so nothing "disappears"
across a restart. The one thing that needs explicit handling is a step
caught mid-execution when the process died: `_run_step` persists
`status=RUNNING` (with `attempts` already incremented) BEFORE calling
ToolRegistry.execute, and persists the step's OUTCOME again immediately
after determining it - not only via the next step's own pre-call save or
`_advance`'s trailing save at the end of a whole batch. Both saves matter:
the first means a crash during the call itself leaves a durable RUNNING
marker; the second means a crash any time AFTER the call returns can
never leave an already-decided outcome (SUCCEEDED, in particular) sitting
on disk as RUNNING. Together they guarantee that the ONLY step
`_recover_interrupted_steps` can ever find RUNNING on the next tick is
the single step that was genuinely still in-flight at the moment of the
crash - never one that had already finished. That step's real outcome is
still unknown (it may have already happened), so it's never blindly
re-run: it's marked FAILED and routed through the normal failure/replan
path, the same "never automatically repeat an irreversible action merely
because Jarvis restarted" requirement this project calls out explicitly.
"""

from __future__ import annotations

import copy
import json
import threading
import time
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from app.notifications.base import DEFAULT_USER, NotificationRouter
from app.orchestration.base import (
    CLARIFICATIONS_KEY,
    CLARIFICATION_STATE_KEY,
    OBJECTIVE_KEY,
    DECLINED_STEP_ERROR,
    EXPIRED_STEP_ERROR,
    INTERACTIVE_PROFILE,
    TIMED_OUT_ERROR,
    EVALUATION_INSUFFICIENT,
    EVALUATION_SUFFICIENT,
    EVALUATION_UNAVAILABLE,
    AutonomyLevel,
    Job,
    JobEvent,
    JobStatus,
    JobStore,
    MAX_CLARIFICATIONS,
    Plan,
    PlanStep,
    REQUEST_CONTEXT_KEY,
    RetryPolicy,
    StaleJobError,
    StepStatus,
    TERMINAL_JOB_STATUSES,
    TERMINAL_STEP_STATUSES,
)
from app.orchestration.evaluator import EvaluationError, ResultEvaluator
from app.orchestration.interactions import (
    MAX_INPUT_QUESTIONS_PER_JOB,
    MAX_INTERACTIONS_PER_JOB,
    InteractionError,
    InteractionJobStateError,
    InteractionLimitError,
    InteractionRequest,
    InteractionStatus,
    InteractionStore,
    InteractionType,
    default_ttl_seconds,
)
from app.orchestration.notification_events import completed_event, failed_event
from app.orchestration.report import build_final_report
from app.orchestration.planner import Planner, PlanningError, ReplanContext
from app.orchestration.references import (
    ReferenceError,
    looks_like_reference,
    resolve_reference,
    validate_reference,
)
from app.orchestration.profiles import RunProfile, UnknownProfileError, permitted_tools, profile_for
from app.orchestration.validator import (
    PlanValidationError,
    validate_plan,
    validate_resolved_arguments,
)
from app.tools.base import (
    AUTOMATIC_PERMISSIONS,
    PermissionLevel,
    ToolArgumentError,
    ToolResult,
    ToolResultStatus,
)
from app.reminders.notifier import Notifier
from app.tools.registry import ToolRegistry

#: How often JobEngine polls by default when none is configured - matches
#: ReminderScheduler/AutomationEngine's own default of 30s, though
#: settings.orchestration_poll_interval_seconds defaults lower (15s,
#: see app/config.py) since a Job in progress benefits more from being
#: advanced promptly than a due reminder does.
DEFAULT_POLL_INTERVAL_SECONDS = 15.0

#: Statuses JobEngine's own tick ever needs to touch - PAUSED and every
#: terminal status are deliberately excluded, so a paused/cancelled/
#: finished Job is never re-planned or re-advanced (see `tick`).
_ACTIVE_STATUSES = (JobStatus.PENDING, JobStatus.PLANNING, JobStatus.RUNNING)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class AppliedInteraction:
    """What a coupled interaction transition committed: the interaction in
    its new terminal state and the Job as updated in the same transaction."""

    interaction: InteractionRequest
    job: Job


@dataclass(frozen=True)
class FollowUp:
    """What answering a parked question led to (see `JobEngine.follow_up`):
    `question` - the next question, now open; `stopped` - the Job could not go
    on and has been failed; neither - it simply carries on."""

    question: InteractionRequest | None = None
    stopped: bool = False


def _run_in_daemon_thread(deliver: Callable[[], None]) -> None:
    """Default way a completion notification is delivered: on its own
    daemon thread, so a slow or hanging notifier can never stall (or keep
    alive) the JobEngine."""
    threading.Thread(target=deliver, name="jarvis-job-notification", daemon=True).start()


def _serialize_tool_result(result: ToolResult) -> dict:
    """A small, JSON-safe summary of a ToolResult for persistence -
    never the live object. `data` is kept as-is only if it's already
    JSON-safe; otherwise stringified, so a step's stored result never
    breaks JobStore's own JSON encoding regardless of what a tool
    returns."""
    data = result.data
    if data is not None and not isinstance(data, (dict, list, str, int, float, bool)):
        data = str(data)
    return {
        "status": result.status.value,
        "message": result.message,
        "data": data,
    }


class JobEngine:
    """Polls `store` for active Jobs and advances each one: planning a
    PENDING Job, then executing a RUNNING Job's ready steps, one batch at
    a time, respecting dependencies/permissions/retries/replans.

    `tick()` is the actual check-and-run step, fully synchronous and with
    all its side effects visible before it returns - tests call it
    directly instead of racing a background thread; `start()`/`stop()`
    just wrap it in a daemon polling loop for the running application,
    same convention as AutomationEngine/ReminderScheduler.
    """

    def __init__(
        self,
        store: JobStore,
        tools: ToolRegistry,
        planner: Planner,
        *,
        max_steps: int,
        max_total_steps: int,
        max_retries_per_step: int,
        max_replans: int,
        retry_delay_seconds: float,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        context_provider=None,
        sleep=time.sleep,
        evaluator: ResultEvaluator | None = None,
        notifier: Notifier | None = None,
        notification_dispatcher: Callable[[Callable[[], None]], None] | None = None,
        interaction_store: InteractionStore | None = None,
        notification_router: NotificationRouter | None = None,
        notify_on_completion: bool = True,
        objective_continuer=None,
        clock: Callable[[], datetime] | None = None,
    ):
        self.store = store
        self.tools = tools
        self.planner = planner
        self.max_steps = max_steps
        self.max_total_steps = max_total_steps
        self.max_retries_per_step = max_retries_per_step
        self.max_replans = max_replans
        self.retry_delay_seconds = retry_delay_seconds
        self.poll_interval_seconds = poll_interval_seconds
        # Optional zero-arg callable returning a fresh JSON-safe context
        # dict (see app/orchestration/tools.py:CreateJobTool - the same
        # kind of provider, typically `memory.get_facts`). `None` (the
        # default, and every existing test's default) means "never
        # refresh" - a Job's context stays exactly the one-time snapshot
        # taken at creation, byte-for-byte the pre-refresh behavior.
        self.context_provider = context_provider
        # Consulted ONLY for a step whose plan entry carries an `expectation`
        # (see `_evaluate_pending`). `None` means no evaluator is available:
        # such a step then gets an "unavailable" verdict and fails safely -
        # it is never silently treated as sufficient.
        self.evaluator = evaluator
        # Optional, best-effort desktop notification when a Job finishes
        # (COMPLETED or FAILED) - see `_notify_if_finished`. It is never
        # part of the critical path: it runs after the Job's final state is
        # persisted, on its own daemon thread, and any failure is swallowed.
        # `notification_dispatcher` (tests) replaces how the delivery
        # callable is run.
        self.notifier = notifier
        # Where this engine's Jobs' interactions live (see
        # app/orchestration/interactions.py). `None` (every existing caller)
        # means interactions aren't in use: resolve/decline_interaction then
        # refuse, and planning is given no clarifications - behavior is
        # exactly as before interactions existed.
        self.interactions = interaction_store
        self._dispatch_notification = notification_dispatcher or _run_in_daemon_thread
        # When set, a Job's final outcome is announced through this router
        # (channel-neutral, prioritised, de-duplicated - see
        # app/notifications/) INSTEAD of the single-toast `notifier` above.
        # A failure is announced only once the Job is terminally FAILED, i.e.
        # after retries and replanning could not save it; a completion only
        # for a background Job, at LOW priority. Never critical.
        self.notification_router = notification_router
        self.notify_on_completion = notify_on_completion
        # Duck-typed (app/handle_this/continuation.py): given a parked Job's
        # stored objective and the user's new answers, says whether the
        # ORIGINAL objective is now ready to plan, needs one more question, or
        # cannot go on. `None` (every existing caller) means a Job with an
        # objective is planned exactly as before. It is pure and never called
        # for a Job that has no resolved Handle This question to fold in.
        self.objective_continuer = objective_continuer
        # An aware-datetime callable, used ONLY to compare a Job's own `deadline_at` against "now". None (every existing caller) is
        # the real clock; tests inject one so a deadline can be crossed without waiting.
        self._clock = clock
        # Callables run at the end of every tick (see `add_tick_hook`).
        self._tick_hooks: list[Callable[[], None]] = []
        self._sleep = sleep
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    # -- background thread wiring (mirrors AutomationEngine/ReminderScheduler) --

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="jarvis-job-engine", daemon=True
        )
        self._thread.start()

    def is_running(self) -> bool:
        """True while the background thread is alive (read by the System health view - app/diagnostics)."""
        return self._thread is not None and self._thread.is_alive()

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - a failed poll must not kill the loop
                pass
            self._stop_event.wait(self.poll_interval_seconds)

    # -- the actual work --

    def tick(self) -> int:
        """Advance every currently-active Job once and return how many
        were processed. Never raises past a single Job: each one is
        isolated in its own try/except, the same failure-isolation
        philosophy ReminderScheduler/AutomationEngine already apply."""
        processed = 0
        for status in _ACTIVE_STATUSES:
            for job in self.store.list(status=status):
                try:
                    self._process(job)
                except StaleJobError:
                    # Someone else (a cancel/pause/confirm from chat, say)
                    # changed this Job after this tick read it, so this
                    # tick's copy is out of date and its write was refused:
                    # the newer state stands. Abandon the rest of this
                    # tick's work on the Job - no further step runs from a
                    # stale view, and no completion notification is sent for
                    # a state that was never persisted. The next tick
                    # re-reads the Job and carries on from what is really
                    # there (an interrupted step is recovered as usual).
                    continue
                except Exception:  # noqa: BLE001 - isolate one Job's failure
                    continue
                processed += 1
                # After `_process` returned, so the Job's final state is
                # already persisted; a Job listed here is never terminal
                # yet, so this fires at most once per Job.
                self._notify_if_finished(job)
        self._run_tick_hooks()
        return processed

    def add_tick_hook(self, hook: Callable[[], None]) -> None:
        """Run `hook()` at the end of every tick (after the Jobs were
        advanced). Each hook is isolated: one that raises never affects the
        tick, the Jobs or another hook. Used for the interaction lifecycle
        (reminders and expiry); it is housekeeping, never part of advancing a
        Job."""
        self._tick_hooks.append(hook)

    def _run_tick_hooks(self) -> None:
        for hook in list(self._tick_hooks):
            try:
                hook()
            except Exception:  # noqa: BLE001 - housekeeping must never break the engine
                pass

    # -- completion notification (optional, never critical) --

    @staticmethod
    def _clean_for_notification(text: str | None, limit: int) -> str:
        """Single line, no control/format characters, bounded - the text
        comes from goals and tool output (untrusted), and ends up in a
        desktop toast."""
        cleaned = "".join(
            " " if unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") else ch
            for ch in (text or "")
        )
        collapsed = " ".join(cleaned.split())
        return collapsed if len(collapsed) <= limit else collapsed[: limit - 3] + "..."

    def _notification_text(self, job: Job) -> tuple[str, str]:
        clean = self._clean_for_notification
        goal = clean(job.goal, 80)
        if job.status == JobStatus.COMPLETED:
            first_line = next(
                (line for line in (job.final_result or "").splitlines() if line.strip()), ""
            )
            outcome = clean(first_line, 200) or "It finished."
            # The message always starts with "Goal:" - never with the
            # untrusted text itself (a leading '-' could read as an option).
            return "Jarvis: task completed", f"Goal: {goal}. {outcome}"
        reason = clean(job.error, 200) or "It could not be completed."
        return "Jarvis: task failed", f"Goal: {goal}. {reason}"

    @staticmethod
    def _failure_was_users_decision(job: Job) -> bool:
        """True if every failed step failed only because the USER declined or
        never answered its confirmation - already announced (or chosen by
        the user), so not also reported as a failure."""
        failed = [s for s in (job.plan.steps if job.plan else []) if s.status == StepStatus.FAILED]
        return bool(failed) and all(
            s.error in (DECLINED_STEP_ERROR, EXPIRED_STEP_ERROR) for s in failed
        )

    def _route_finished(self, job: Job) -> None:
        """Announce a finished Job through the notification router. Never
        raises and never affects the Job (which is already persisted)."""
        try:
            if job.run_profile != INTERACTIVE_PROFILE:
                return  # a Job running under a capability profile is reported by whoever created it, in its own words (no generic duplicate)
            if job.status == JobStatus.FAILED:
                if self._failure_was_users_decision(job):
                    return
                event = failed_event(job)
            elif job.status == JobStatus.COMPLETED:
                # A Job that belongs to a conversation is reported by the
                # conversation itself; only a background Job earns a (LOW,
                # informational) notification.
                if not self.notify_on_completion or job.conversation_id is not None:
                    return
                event = completed_event(job)
            else:
                return
            self.notification_router.notify(DEFAULT_USER, event)
        except Exception:  # noqa: BLE001 - notifying is never part of the Job
            pass

    def _notify_if_finished(self, job: Job) -> None:
        if self.notification_router is not None:
            self._route_finished(job)
            return
        if self.notifier is None or job.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
            return
        try:
            title, message = self._notification_text(job)
            notifier = self.notifier

            def deliver() -> None:
                try:
                    notifier.notify(title, message)
                except Exception:  # noqa: BLE001 - a failed toast must never affect the Job
                    pass

            self._dispatch_notification(deliver)
        except Exception:  # noqa: BLE001 - nor may building/dispatching it
            pass

    def _process(self, job: Job) -> None:
        try:
            self._profile_of(job)
        except UnknownProfileError as exc:
            # A Job naming a profile this system does not define is refused, never run as if it had none.
            job.status = JobStatus.FAILED
            job.error = f"This run cannot start: {exc}"
            self._log(job, "profile_unknown", detail=job.error)
            job.updated_at = _now()
            self.store.save(job)
            return
        if self._enforce_deadline(job):
            job.updated_at = _now()
            self.store.save(job)
            return

        self._recover_interrupted_steps(job)

        if job.status in (JobStatus.PENDING, JobStatus.PLANNING):
            if job.status is JobStatus.PENDING and not self._continue_clarified_objective(job):
                return  # it waits again, or failed - already saved
            self._plan(job)
            if job.status != JobStatus.RUNNING:
                return  # planning failed - already saved as FAILED

        if job.status == JobStatus.RUNNING:
            self._advance(job)

    # -- a parked objective, continued once its question is answered ----------------------------
    #
    # A Job Handle This parked (AWAITING_INPUT, see CreateJobTool's
    # `clarification`) is resumed by the interaction resolver's coupled
    # transition: the SAME Job goes back to PENDING with the interaction
    # RESOLVED. Before it is planned, this folds the answer into the stored
    # ORIGINAL objective and re-checks it. Everything it reads is persisted (the
    # resolved interaction, the objective, the applied-ids record), so a restart
    # at any point simply repeats the step; recording the ids in the same write
    # that stores the result makes repeating it harmless.

    def _continue_clarified_objective(self, job: Job) -> bool:
        """True if the Job may go on to planning; False if it now waits on
        another question or has failed (either way already persisted)."""
        folded = self._fold_answers(job)
        if folded is None:
            return True  # an ordinary Job, or one whose answers are already folded in
        fresh, verdict, now_applied = folded
        if verdict is None:
            return self._fail_clarification(job, "the request could not be completed with that answer")

        if verdict.kind == "ready":
            self._stamp_folded(job, verdict, now_applied)
            self._log(job, "clarification_applied", detail=f"{len(fresh)} answer(s) folded into the objective")
            self.store.save(job)
            return True

        try:
            self._park_follow_up(job, verdict, now_applied)
        except InteractionLimitError:
            return self._fail_clarification(job, "it needed more information than the limit allows")
        return False

    def _fold_answers(self, job: Job):
        """What the user's new answers do to the Job's stored objective:
        None if there is nothing to fold in (no objective, or every answer is
        already folded), else (the fresh answered interactions, the
        continuer's verdict - None if it could not be used - and the full set
        of applied ids). Reads only persisted state, so repeating it after a
        restart gives the same result."""
        if self.objective_continuer is None or self.interactions is None:
            return None
        objective = job.context.get(OBJECTIVE_KEY)
        if not isinstance(objective, dict):
            return None
        state = job.context.get(CLARIFICATION_STATE_KEY)
        applied = set(state.get("applied", [])) if isinstance(state, dict) else set()
        revisions = state.get("revisions") if isinstance(state, dict) else None
        revisions = revisions if isinstance(revisions, dict) else {}
        fresh = [
            i
            for i in self.interactions.list_for_job(job.id)
            if i.type is InteractionType.INPUT
            and i.status is InteractionStatus.RESOLVED
            and i.detail.get("origin") == "handle_this"
            and i.id not in applied
        ]
        if not fresh:
            return None

        # A topic the user has since REVISED is folded in with its latest answer.
        answers = [
            {
                "topic": i.detail.get("topic"),
                "answer": revisions.get(i.detail.get("topic")) or i.response or "",
                "interaction_id": i.id,
            }
            for i in fresh
        ]
        now_applied = sorted(applied | {i.id for i in fresh})
        try:
            verdict = self.objective_continuer.continue_objective(objective, answers)
        except Exception:  # noqa: BLE001 - never plan from an objective that could not be re-checked
            return fresh, None, now_applied
        if verdict.kind not in ("ready", "ask") or not isinstance(verdict.objective, dict):
            return fresh, None, now_applied
        return fresh, verdict, now_applied

    @staticmethod
    def _stamp_folded(target: Job, verdict, now_applied: list[str]) -> None:
        state = target.context.get(CLARIFICATION_STATE_KEY)
        target.context = {
            **target.context,
            OBJECTIVE_KEY: verdict.objective,
            # applied ids move on; anything else recorded (revisions) is kept
            CLARIFICATION_STATE_KEY: {**(state if isinstance(state, dict) else {}), "applied": now_applied},
        }
        target.updated_at = _now()

    def _park_follow_up(self, job: Job, verdict, now_applied: list[str]):
        """Make the SAME Job wait on one more question, atomically. Raises
        InteractionLimitError (nothing changed) past the per-Job limits."""

        def park(target: Job) -> None:
            self._stamp_folded(target, verdict, now_applied)
            target.status = JobStatus.AWAITING_INPUT
            target.confirmation_message = None
            self._log(target, "clarification_requested", detail="one more answer is needed")

        return self.interactions.park_job_for_input(
            job.id,
            expected_job_version=job.version,
            mutate_job=park,
            prompt=verdict.question,
            detail=verdict.detail,
            ttl_seconds=default_ttl_seconds(InteractionType.INPUT),
            check=self._check_park_limits,
        )

    def open_follow_up(self, job_id: str) -> InteractionRequest | None:
        """The new PENDING question `follow_up` opened, or None."""
        return self.follow_up(job_id).question

    def follow_up(self, job_id: str) -> "FollowUp":
        """Called right after the user answered a question of a parked
        objective: if that answer leaves something ELSE materially missing (or
        the same thing, unusably answered), open the next question NOW, so the
        reply can put it to the user instead of the task appearing to carry on
        and asking a moment later. It is exactly the step the next tick takes
        (`_continue_clarified_objective`), taken early:

        - it would ask, and may       -> the next question is opened (`question`);
        - it would ask, but the per-Job limit is used up, or the objective can
          no longer be completed      -> the Job FAILS now, exactly as the tick
          would have failed it, and `stopped` is True - so the user is told the
          truth in the same reply instead of being told it is continuing;
        - it is ready                 -> nothing here: the tick plans it, as ever.

        Version-guarded like every write, so racing the tick (or another
        message) can neither open a second question nor fail a Job twice; if it
        loses, nothing changed and the tick decides."""
        try:
            job = self.store.get(job_id)
            if job is None or job.status is not JobStatus.PENDING:
                return FollowUp()
            folded = self._fold_answers(job)
            if folded is None:
                return FollowUp()
            _, verdict, now_applied = folded
            if verdict is None:
                self._fail_clarification(job, "the request could not be completed with that answer")
                return FollowUp(stopped=True)
            if verdict.kind != "ask":
                return FollowUp()
            try:
                _, interaction = self._park_follow_up(job, verdict, now_applied)
            except InteractionLimitError:
                self._fail_clarification(job, "it needed more information than the limit allows")
                return FollowUp(stopped=True)
            return FollowUp(question=interaction)
        except (StaleJobError, InteractionError):
            return FollowUp()  # someone else moved it: the tick decides, as before

    @staticmethod
    def _check_park_limits(job: Job, siblings: list[InteractionRequest]) -> None:
        """The per-Job limits, applied atomically when a NEW question is about
        to be opened: at most MAX_INPUT_QUESTIONS_PER_JOB INPUT questions and
        MAX_INTERACTIONS_PER_JOB interactions in total. Only questions that
        were asked and not withdrawn count (pending, resolved, declined): a
        cancelled or expired one never got an answer and does not use one up."""
        live = [
            s
            for s in siblings
            if s.status
            in (InteractionStatus.PENDING, InteractionStatus.RESOLVED, InteractionStatus.DECLINED)
        ]
        if len(live) >= MAX_INTERACTIONS_PER_JOB:
            raise InteractionLimitError(
                f"Job '{job.id}' has already used its {MAX_INTERACTIONS_PER_JOB} interactions."
            )
        if sum(1 for s in live if s.type is InteractionType.INPUT) >= MAX_INPUT_QUESTIONS_PER_JOB:
            raise InteractionLimitError(
                f"Job '{job.id}' has already used its {MAX_INPUT_QUESTIONS_PER_JOB} questions."
            )

    def _fail_clarification(self, job: Job, reason: str) -> bool:
        job.status = JobStatus.FAILED
        job.error = f"Could not continue the request: {reason}."
        self._log(job, "clarification_failed", detail=reason)
        job.updated_at = _now()
        self.store.save(job)
        return False

    # -- restart recovery --

    def _recover_interrupted_steps(self, job: Job) -> None:
        """Find any step this process left at RUNNING - only possible if
        a previous process died mid-`_run_step` (see module docstring) -
        and fail it without re-executing anything, then let the normal
        failure/replan path decide what happens next. A no-op on every
        ordinary tick."""
        if job.plan is None:
            return
        interrupted = [s for s in job.plan.steps if s.status == StepStatus.RUNNING]
        if not interrupted:
            return
        for step in interrupted:
            step.status = StepStatus.FAILED
            step.error = "Interrupted by a process restart - not automatically retried."
            self._log(job, "step_interrupted", step_id=step.id, detail=step.error)
            if job.status == JobStatus.RUNNING:
                # Same failure/replan decision a normal exhausted-retry
                # failure gets (see `_handle_step_failure`) - an
                # interrupted step's real outcome is unknown, so it's
                # never just silently retried in place.
                self._handle_step_failure(job, step)
        job.updated_at = _now()
        self.store.save(job)

    # -- planning --

    def _refresh_context(self, job: Job) -> None:
        """Refresh `job.context` from `self.context_provider` immediately
        before a planning call - never relying solely on the one-time
        snapshot CreateJobTool took when the Job was first created, which
        can go stale if a Job sits PENDING for a while, or is paused and
        resumed later. Only ever touches `job.context` - `job.goal`
        (the original goal) and `job.plan`/`job.events` (this Job's own
        execution history) are never altered here, so both survive a
        refresh unchanged, exactly as required.

        A no-op when `context_provider` wasn't given (every pre-existing
        test/caller) - preserves the prior "context is a fixed snapshot"
        behavior exactly. A provider that raises is swallowed: a failed
        context refresh must never block planning, since the alternative
        (a stale-but-present snapshot) is strictly safer than aborting the
        Job outright over a transient memory-store hiccup.
        """
        if self.context_provider is None:
            return
        try:
            fresh = self.context_provider()
        except Exception:  # noqa: BLE001 - a refresh failure must not block planning
            return
        if fresh:
            refreshed = dict(fresh)
            # The request-context handoff (see REQUEST_CONTEXT_KEY) is
            # captured once, when the goal was received, and is not part
            # of what `context_provider` returns - a refresh must not
            # discard it.
            if REQUEST_CONTEXT_KEY in job.context:
                refreshed[REQUEST_CONTEXT_KEY] = job.context[REQUEST_CONTEXT_KEY]
            # Likewise the structured objective Handle This understood: it is
            # captured once, when the request was received.
            if OBJECTIVE_KEY in job.context:
                refreshed[OBJECTIVE_KEY] = job.context[OBJECTIVE_KEY]
            if CLARIFICATION_STATE_KEY in job.context:
                refreshed[CLARIFICATION_STATE_KEY] = job.context[CLARIFICATION_STATE_KEY]
            job.context = refreshed

    # -- interactions: the coupled, engine-owned transition -------------------------------
    #
    # A user's reply reaches a Job only through these two methods. Each moves
    # the interaction to its terminal state AND applies the consequence to
    # the Job and its steps in ONE transaction (InteractionStore.
    # transition_with_job): both commit or neither does, both are
    # version-guarded, and nothing here executes a tool or grants a
    # permission - approving only clears the gate on the named steps, and
    # the steps still run later, through this engine's normal `_advance` ->
    # ToolRegistry.execute path (capability/permission checks unchanged).
    #
    # Every failure raises and leaves BOTH records exactly as they were:
    # StaleInteractionError (interaction changed), StaleJobError (Job
    # changed), InteractionJobStateError (Job/steps no longer waiting for
    # this), InteractionExpiredError / InvalidInteractionTransitionError
    # (interaction not answerable), InteractionNotFoundError.

    def resolve_interaction(
        self,
        interaction_id: str,
        *,
        expected_version: int,
        expected_job_version: int,
        response: str | None = None,
        via: str | None = None,
        now: datetime | None = None,
    ) -> AppliedInteraction:
        """RESOLVE an interaction and apply it. INPUT: `response` is the
        answer (required, untrusted text); the Job resumes through normal
        planning. CONFIRMATION: approved - ONLY the gated steps this
        interaction names are released; independent steps are untouched."""
        return self._apply_interaction(
            interaction_id,
            InteractionStatus.RESOLVED,
            expected_version=expected_version,
            expected_job_version=expected_job_version,
            response=response,
            reason=None,
            via=via,
            now=now,
        )

    def decline_interaction(
        self,
        interaction_id: str,
        *,
        expected_version: int,
        expected_job_version: int,
        response: str | None = None,
        reason: str | None = None,
        via: str | None = None,
        now: datetime | None = None,
    ) -> AppliedInteraction:
        """DECLINE an interaction and apply it. CONFIRMATION: ONLY the
        gated steps it names fail (dependents are skipped by the normal
        cascade; independent steps keep running; no replan is attempted -
        the user said no). INPUT: the user abandoned the question, so the
        Job is cancelled."""
        return self._apply_interaction(
            interaction_id,
            InteractionStatus.DECLINED,
            expected_version=expected_version,
            expected_job_version=expected_job_version,
            response=response,
            reason=reason,
            via=via,
            now=now,
        )

    def expire_interaction(
        self,
        interaction_id: str,
        *,
        expected_version: int,
        expected_job_version: int,
        now: datetime | None = None,
    ) -> AppliedInteraction:
        """EXPIRE an interaction that has run out of time and apply the
        consequence to its Job, in ONE transaction (same primitive as
        resolve/decline). It can only ever end an interaction whose
        `expires_at` has passed. Nothing is approved and nothing runs:

        - CONFIRMATION: only the gated steps it names fail ("expired - the
          step was not run"); dependents are skipped by the normal cascade,
          independent steps keep going, no replan is attempted.
        - INPUT: the Job cannot continue without the answer, so it FAILS.

        Every failure raises and leaves both records untouched (same errors
        as resolve/decline; `InteractionError` for "not due yet")."""
        if self.interactions is None:
            raise ValueError("This JobEngine has no interaction store configured.")

        def mutate(job: Job, interaction: InteractionRequest) -> None:
            self._apply_to_job(job, interaction, approved=False, expired=True)

        interaction, job = self.interactions.transition_with_job(
            interaction_id,
            InteractionStatus.EXPIRED,
            expected_version=expected_version,
            expected_job_version=expected_job_version,
            mutate_job=mutate,
            reason="expired",
            via="expiry",
            now=now,
        )
        return AppliedInteraction(interaction=interaction, job=job)

    def cancel_job(
        self,
        job_id: str,
        *,
        expected_job_version: int,
        reason: str = "cancelled by the user",
        via: str = "chat",
        now: datetime | None = None,
    ) -> Job:
        """CANCEL a Job the user has dropped, and withdraw every question or
        confirmation still waiting on it, in ONE transaction: both commit or
        neither does, so a cancelled Job is never left with a live interaction
        that could resume it (the same primitive `cancel_job` from chat uses).
        Nothing is planned or run and nothing is resolved - the user's words are
        not an answer. Refused (InteractionJobStateError, nothing changed) for
        a Job that has already finished; StaleJobError if it changed since it
        was read."""
        if self.interactions is None:
            raise ValueError("This JobEngine has no interaction store configured.")

        def mutate(fresh: Job) -> None:
            if fresh.status in TERMINAL_JOB_STATUSES:
                raise InteractionJobStateError(
                    f"Job '{fresh.id}' is already {fresh.status.value}; it cannot be cancelled."
                )
            fresh.status = JobStatus.CANCELLED
            fresh.confirmation_message = None
            self._log(fresh, "cancelled_by_user", detail="the user dropped this task")
            fresh.updated_at = _now()

        job, _withdrawn = self.interactions.cancel_job_with_interactions(
            job_id,
            expected_job_version=expected_job_version,
            mutate_job=mutate,
            reason=reason,
            via=via,
            now=now,
        )
        return job

    def revise_answer(self, job_id: str, topic: str, answer: str, *, expected_job_version: int) -> Job:
        """The user changed an answer they already gave ("actually, make that
        3000 AED"): the latest accepted answer becomes the constraint. The SAME
        Job keeps going - nothing is created, no second interaction is opened,
        and the earlier answer is superseded, not stored beside the new one.

        One version-guarded write of the Job (so a concurrent tick, cancel or
        second message either wins cleanly or this raises StaleJobError and
        nothing changed):

        - the stored objective's constraint for that topic is replaced (by the
          continuer, which also insists the new answer is usable);
        - the revision is recorded (`revisions`), so an answer the engine has
          not folded in yet is folded in as REVISED, and a later replan or the
          planner's clarification list sees only the latest answer;
        - a Job that has not started working is planned again with it: one
          that has no plan yet simply is (it reads the revised objective), and
          one whose plan has not run a single step has that plan dropped and
          goes back to PENDING. A Job already mid-way keeps its plan - work that
          may have had effects is never repeated - and uses the revised answer
          from here on (any replan).

        The answer is data: a short single line in a constraint, nothing that
        selects a tool or changes a permission or autonomy level."""
        if self.interactions is None or self.objective_continuer is None:
            raise InteractionJobStateError("This assistant cannot revise an answer.")
        job = self.store.get(job_id)
        if job is None:
            raise InteractionJobStateError(f"Job '{job_id}' does not exist.")
        if job.version != expected_job_version:
            raise StaleJobError(f"Job '{job.id}' changed since it was read; nothing was changed.")
        if job.status in TERMINAL_JOB_STATUSES:
            raise InteractionJobStateError(f"Job '{job.id}' is already {job.status.value}.")
        objective = job.context.get(OBJECTIVE_KEY)
        if not isinstance(objective, dict):
            raise InteractionJobStateError(f"Job '{job.id}' has no objective to revise.")
        revised = self.objective_continuer.revise_objective(objective, topic, answer)
        if not isinstance(revised, dict):
            raise InteractionJobStateError("That answer cannot be used in place of the earlier one.")

        state = job.context.get(CLARIFICATION_STATE_KEY)
        state = dict(state) if isinstance(state, dict) else {}
        constraint_line = next(
            (c["text"] for c in reversed(revised["constraints"]) if isinstance(c, dict)), answer
        )
        if (state.get("revisions") or {}).get(topic) == constraint_line:
            return job  # the same revision twice (a repeated message): already in force
        state["revisions"] = {**(state.get("revisions") or {}), topic: constraint_line}
        job.context = {**job.context, OBJECTIVE_KEY: revised, CLARIFICATION_STATE_KEY: state}

        restarted = False
        if (
            job.plan is not None
            and job.status is JobStatus.RUNNING
            and not job.pending_step_ids
            and all(s.status is StepStatus.PENDING for s in job.plan.steps)
        ):
            job.plan = None  # nothing has run: it is planned again with the revised answer
            job.total_step_count = 0
            job.status = JobStatus.PENDING
            restarted = True
        self._log(
            job,
            "clarification_revised",
            detail=f"{topic} answer changed" + ("; planning again" if restarted else ""),
        )
        job.updated_at = _now()
        self.store.save(job)  # compare-and-set on the version read above
        return job

    LIVE, PAUSED, STALE = "live", "paused", "stale"

    @staticmethod
    def interaction_state(interaction: InteractionRequest, job: Job) -> str:
        """Whether `job` is still waiting on `interaction`: LIVE (waiting),
        PAUSED (the Job is paused - nothing is being asked right now, but it
        may resume) or STALE (finished, cancelled, or moved on by another
        route - the interaction can never be applied). Pure."""
        if job.status in TERMINAL_JOB_STATUSES:
            return JobEngine.STALE
        if job.status is JobStatus.PAUSED:
            return JobEngine.PAUSED
        if interaction.type is InteractionType.INPUT:
            return JobEngine.LIVE if job.status is JobStatus.AWAITING_INPUT else JobEngine.STALE
        if job.status is not JobStatus.AWAITING_CONFIRMATION or job.plan is None:
            return JobEngine.STALE
        for step_id in interaction.step_ids:
            step = job.plan.get(step_id)
            if (
                step is None
                or step_id not in job.pending_step_ids
                or step.status is not StepStatus.PENDING
                or step.confirmed
            ):
                return JobEngine.STALE
        return JobEngine.LIVE

    def discard_stale_interaction(self, interaction_id: str, *, expected_version: int) -> bool:
        """Withdraw a PENDING interaction whose Job is no longer waiting on it
        (see `interaction_state`), so it can neither be answered nor nag. The
        staleness is re-checked INSIDE the transaction that cancels it, so a
        Job that started waiting again in between keeps its interaction. Only
        the interaction changes - there is no Job-side consequence, so the Job
        is never written. Returns False (changing nothing) if it is not
        stale, not pending, or changed since it was read."""
        if self.interactions is None:
            raise ValueError("This JobEngine has no interaction store configured.")

        def must_be_stale(job: Job, interaction: InteractionRequest) -> None:
            if self.interaction_state(interaction, job) != self.STALE:
                raise InteractionJobStateError(
                    f"Job '{job.id}' is still waiting on interaction '{interaction.id}'."
                )

        try:
            self.interactions.cancel_with_job_check(
                interaction_id,
                expected_version=expected_version,
                check=must_be_stale,
                reason="no longer waiting",
                via="lifecycle",
            )
        except InteractionError:
            return False
        return True

    def _apply_interaction(
        self,
        interaction_id: str,
        target: InteractionStatus,
        *,
        expected_version: int,
        expected_job_version: int,
        response: str | None,
        reason: str | None,
        via: str | None,
        now: datetime | None,
    ) -> AppliedInteraction:
        if self.interactions is None:
            raise ValueError("This JobEngine has no interaction store configured.")

        approved = target is InteractionStatus.RESOLVED

        def mutate(job: Job, interaction: InteractionRequest) -> None:
            self._apply_to_job(job, interaction, approved=approved)

        interaction, job = self.interactions.transition_with_job(
            interaction_id,
            target,
            expected_version=expected_version,
            expected_job_version=expected_job_version,
            mutate_job=mutate,
            check=self._check_interaction_limits,
            response=response,
            reason=reason,
            via=via,
            now=now,
        )
        return AppliedInteraction(interaction=interaction, job=job)

    @staticmethod
    def _check_interaction_limits(
        job: Job, interaction: InteractionRequest, siblings: list[InteractionRequest]
    ) -> None:
        """The agreed v1 limits, checked inside the coupled transaction
        against the Job's other interactions as that transaction sees them:
        at most MAX_INPUT_QUESTIONS_PER_JOB INPUT questions and
        MAX_INTERACTIONS_PER_JOB interactions in total may be ANSWERED
        (resolved or declined). Refusing leaves everything unchanged.

        REMAINING BOUNDARY: this bounds what is APPLIED. Refusing to open a
        third question in the first place belongs to whatever creates
        interactions (a later stage); nothing creates them yet."""
        answered = [
            s
            for s in siblings
            if s.id != interaction.id
            and s.status in (InteractionStatus.RESOLVED, InteractionStatus.DECLINED)
        ]
        if len(answered) >= MAX_INTERACTIONS_PER_JOB:
            raise InteractionLimitError(
                f"Job '{job.id}' has already used its {MAX_INTERACTIONS_PER_JOB} interactions."
            )
        if (
            interaction.type is InteractionType.INPUT
            and sum(1 for s in answered if s.type is InteractionType.INPUT)
            >= MAX_INPUT_QUESTIONS_PER_JOB
        ):
            raise InteractionLimitError(
                f"Job '{job.id}' has already used its {MAX_INPUT_QUESTIONS_PER_JOB} questions."
            )

    def _apply_to_job(
        self,
        job: Job,
        interaction: InteractionRequest,
        *,
        approved: bool,
        expired: bool = False,
    ) -> None:
        """The Job-side consequence of an interaction outcome. A PURE edit
        of the Job copy it is given (it runs inside the transaction, under
        the store's lock): it validates that the Job and steps are still in
        the state the interaction was waiting on - raising
        InteractionJobStateError if not - and changes nothing else."""
        if interaction.type is InteractionType.INPUT:
            self._apply_input(job, interaction, approved=approved, expired=expired)
        elif interaction.type is InteractionType.CONFIRMATION:
            self._apply_confirmation(job, interaction, approved=approved, expired=expired)
        else:  # a future kind with no handler yet: refuse, never guess
            raise InteractionJobStateError(
                f"No handler for interaction type '{interaction.type.value}'."
            )
        job.updated_at = _now()

    def _apply_input(
        self, job: Job, interaction: InteractionRequest, *, approved: bool, expired: bool = False
    ) -> None:
        if job.status is not JobStatus.AWAITING_INPUT:
            raise InteractionJobStateError(
                f"Job '{job.id}' is {job.status.value}, not waiting for input."
            )
        job.confirmation_message = None
        if approved:
            # Resume through the NORMAL pipeline: a Job with no plan yet is
            # planned again (its resolved answers are handed to the planner -
            # see `_planning_context`); one that already has a plan carries on.
            job.status = JobStatus.PENDING if job.plan is None else JobStatus.RUNNING
            self._log(job, "interaction_resolved", detail=f"input answered (code {interaction.code})")
        elif expired:
            # Nobody answered and the Job cannot go on without it.
            job.status = JobStatus.FAILED
            job.error = "Timed out waiting for the user's answer."
            self._log(job, "interaction_expired", detail=f"input expired - task failed (code {interaction.code})")
        else:
            job.status = JobStatus.CANCELLED  # the user abandoned the question
            self._log(job, "interaction_declined", detail=f"input declined - task cancelled (code {interaction.code})")

    def _apply_confirmation(
        self, job: Job, interaction: InteractionRequest, *, approved: bool, expired: bool = False
    ) -> None:
        if job.status is not JobStatus.AWAITING_CONFIRMATION or job.plan is None:
            raise InteractionJobStateError(
                f"Job '{job.id}' is {job.status.value}, not waiting for confirmation."
            )
        steps = []
        for step_id in interaction.step_ids:
            step = job.plan.get(step_id)
            if (
                step is None
                or step_id not in job.pending_step_ids
                or step.status is not StepStatus.PENDING
                or step.confirmed
            ):
                raise InteractionJobStateError(
                    f"Step '{step_id}' of Job '{job.id}' is no longer waiting for this confirmation."
                )
            steps.append(step)

        for step in steps:
            if approved:
                step.confirmed = True  # clears THIS step's gate only; it still runs via `_advance`
            else:
                step.status = StepStatus.FAILED
                step.error = EXPIRED_STEP_ERROR if expired else DECLINED_STEP_ERROR
            outcome = "approved" if approved else "expired" if expired else "declined"
            self._log(
                job,
                "interaction_resolved" if approved else f"interaction_{outcome}",
                step_id=step.id,
                detail=f"confirmation {outcome} (code {interaction.code})",
            )

        job.pending_step_ids = [s for s in job.pending_step_ids if s not in interaction.step_ids]
        if not job.pending_step_ids:  # nothing else waits: the Job is runnable again
            job.status = JobStatus.RUNNING
            job.confirmation_message = None

    def _planning_context(self, job: Job) -> dict:
        """`job.context`, plus - only when the Job has any - the user's
        answers to its resolved INPUT interactions under CLARIFICATIONS_KEY
        (a copy: never written back into the Job). The answers are the
        user's own text, stored verbatim and handed on as data; the planner
        prompt labels them as such."""
        if self.interactions is None:
            return job.context
        answered = [
            i
            for i in self.interactions.list_for_job(job.id)
            if i.type is InteractionType.INPUT
            and i.status is InteractionStatus.RESOLVED
            and i.response
        ]
        # A topic the user has since REVISED ("actually, make that 3000 AED") is
        # handed over once, with its latest answer - never with the superseded
        # one beside it.
        state = job.context.get(CLARIFICATION_STATE_KEY)
        revisions = state.get("revisions") if isinstance(state, dict) else None
        if isinstance(revisions, dict) and revisions:
            latest = {}
            for i in answered:
                latest[i.detail.get("topic")] = i
            answered = [
                i for i in answered if i.detail.get("topic") not in revisions or latest[i.detail.get("topic")] is i
            ]
        answers = [
            {"question": i.prompt, "answer": (revisions or {}).get(i.detail.get("topic")) or i.response}
            for i in answered
        ][-MAX_CLARIFICATIONS:]
        if not answers:
            return job.context
        return {**job.context, CLARIFICATIONS_KEY: answers}

    def _plannable_tools(self, job: Job | None = None):
        """The tools the planner may be offered: everything registered
        except Jarvis-control tools (`Tool.plannable is False`). Execution
        itself is untouched - `validate_plan` enforces the same rule on
        whatever the planner returns, so a hallucinated step naming a
        hidden tool is rejected rather than run.

        For a Job under a run profile (`job` given) this is NARROWED to the
        profile's tools that the live registry still has as plannable,
        read-only tools - the profile only ever removes tools from this list."""
        if job is None:
            return [tool for tool in self.tools.list_tools() if tool.plannable]
        try:
            profile = self._profile_of(job)
        except UnknownProfileError:
            return []  # fail closed: nothing is offered to a run whose profile is unknown
        if profile is None:
            return [tool for tool in self.tools.list_tools() if tool.plannable]
        return permitted_tools(profile, self.tools)

    # -- run profiles and the per-run deadline -------------------------------------------------
    #
    # Both only ever REDUCE what a Job may do or how long it may take. An ordinary Job (profile
    # "interactive", no deadline) goes through none of this.

    @staticmethod
    def _profile_of(job: Job) -> RunProfile | None:
        """The Job's run profile, or None for an ordinary Job. Raises UnknownProfileError for a name this system does not define."""
        return profile_for(job.run_profile)

    def _limits(self, job: Job) -> tuple[int, int, int]:
        """(max_steps, max_total_steps, max_replans) for this Job: the engine's own, narrowed by its profile - the smaller wins."""
        profile = self._profile_of(job)
        if profile is None:
            return self.max_steps, self.max_total_steps, self.max_replans
        return (
            min(self.max_steps, profile.max_steps),
            min(self.max_total_steps, profile.max_total_steps),
            min(self.max_replans, profile.max_replans),
        )

    def _profile_denial(self, job: Job, step: PlanStep) -> str | None:
        """Why this Job's profile forbids running `step` (checked against the LIVE registry), or None if it does not - which is
        always the case for an ordinary Job. A profile can only say no; it never makes a step runnable."""
        try:
            profile = self._profile_of(job)
        except UnknownProfileError:
            return "This run's profile is unknown."
        if profile is None:
            return None
        return profile.denial(step.tool_name, self.tools.get(step.tool_name))

    def _now_dt(self) -> datetime:
        return (self._clock() if self._clock is not None else datetime.now(timezone.utc)).astimezone(timezone.utc)

    def _past_deadline(self, job: Job) -> bool:
        if not job.deadline_at:
            return False
        try:
            deadline = datetime.fromisoformat(job.deadline_at)
        except ValueError:
            return True  # an unreadable deadline is treated as passed: fail closed
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        return self._now_dt() >= deadline

    def _enforce_deadline(self, job: Job) -> bool:
        """If this Job is still active and past its own `deadline_at`, mark it FAILED (TIMED_OUT_ERROR) in memory and return True -
        the caller persists. Called only BETWEEN steps, inside the existing loop: it cannot interrupt a tool call already in flight.
        Nothing is retried and nothing further is planned or run."""
        if job.status in TERMINAL_JOB_STATUSES or not self._past_deadline(job):
            return False
        for step in job.plan.steps if job.plan else []:
            if step.status == StepStatus.RUNNING:  # only possible from a process that died mid-step
                step.status = StepStatus.FAILED
                step.error = "Interrupted: the run timed out."
        job.status = JobStatus.FAILED
        job.error = TIMED_OUT_ERROR
        self._log(job, "job_timed_out", detail=f"deadline {job.deadline_at}")
        return True

    def _plan(self, job: Job) -> None:
        job.status = JobStatus.PLANNING
        job.updated_at = _now()
        self.store.save(job)

        self._refresh_context(job)
        available_tools = self._plannable_tools(job)
        profiled = job.run_profile != INTERACTIVE_PROFILE
        if profiled and not available_tools:
            # Nothing this kind of run may use is available right now: do not even ask the planner.
            job.status = JobStatus.FAILED
            job.error = "Could not create a plan: no capability is available for this kind of run."
            self._log(job, "planning_failed", detail=job.error)
            job.updated_at = _now()
            self.store.save(job)
            return
        max_steps = self._limits(job)[0]
        try:
            raw_steps = self.planner.plan(
                job.goal,
                available_tools=available_tools,
                context=self._planning_context(job),
                autonomy=job.autonomy,
            )
            steps = validate_plan(
                raw_steps,
                self.tools,
                max_steps=max_steps,
                default_retry=RetryPolicy(
                    max_attempts=self.max_retries_per_step,
                    delay_seconds=self.retry_delay_seconds,
                ),
                allowed_tool_names=frozenset(t.name for t in available_tools) if profiled else None,
            )
        except (PlanningError, PlanValidationError) as exc:
            job.status = JobStatus.FAILED
            job.error = f"Could not create a plan: {exc}"
            self._log(job, "planning_failed", detail=str(exc))
            job.updated_at = _now()
            self.store.save(job)
            return

        job.plan = Plan(steps=steps)
        job.total_step_count = len(steps)
        job.status = JobStatus.RUNNING
        self._log(job, "planned", detail=f"{len(steps)} step(s)")
        job.updated_at = _now()
        self.store.save(job)

    # -- execution --

    def _advance(self, job: Job) -> None:
        assert job.plan is not None
        progressed = True

        while progressed and job.status == JobStatus.RUNNING:
            progressed = False

            # The per-run deadline, checked between steps: past it the Job is FAILED (TIMED_OUT_ERROR) and nothing more runs.
            if self._enforce_deadline(job):
                break

            # A step that succeeded and explicitly asked to be evaluated
            # must get its verdict before anything can depend on it.
            if self._evaluate_pending(job):
                progressed = True
            if job.status != JobStatus.RUNNING:
                break

            if self._cascade_skips(job):
                progressed = True

            ready = [
                step
                for step in job.plan.steps
                if step.status == StepStatus.PENDING
                and all(self._dependency_satisfied(job.plan, dep) for dep in step.depends_on)
            ]
            if not ready:
                break

            # Run every ready step that doesn't need confirmation FIRST -
            # "Jarvis has prepared the email and selected the recipients;
            # sending it will contact 4 people - send?" (see this
            # project's own Confirmation example) means auto steps in the
            # same ready batch as a gated one still run, and only the
            # gated ones actually pause the Job - never the whole batch.
            # A step the Job's run profile forbids is never parked for a confirmation nobody is there to give: it goes straight
            # to _run_step, which refuses it. (For an ordinary Job `blocked_ids` is empty and these lists are exactly as before.)
            blocked_ids = {s.id for s in ready if self._profile_denial(job, s)}
            auto_ready = [
                s for s in ready if s.id in blocked_ids or not self._needs_confirmation(s, job)
            ]
            confirm_ready = [
                s for s in ready if s.id not in blocked_ids and self._needs_confirmation(s, job)
            ]

            for step in auto_ready:
                if self._enforce_deadline(job):
                    break
                self._run_step(job, step)
                progressed = True
                if step.status == StepStatus.FAILED:
                    self._handle_step_failure(job, step)
                    if job.status != JobStatus.RUNNING:
                        break

            if job.status != JobStatus.RUNNING:
                break

            if confirm_ready:
                job.status = JobStatus.AWAITING_CONFIRMATION
                job.pending_step_ids = [s.id for s in confirm_ready]
                job.confirmation_message = self._build_confirmation_message(job, confirm_ready)
                self._log(
                    job,
                    "confirmation_requested",
                    detail=job.confirmation_message,
                )
                break

        # One last cascade pass regardless of outcome, so a Job that ended
        # FAILED still has accurate SKIPPED bookkeeping for any step whose
        # dependency chain was never going to succeed - never left
        # ambiguously PENDING forever (see `_cascade_skips`).
        while self._cascade_skips(job):
            pass

        if job.status == JobStatus.RUNNING:
            self._maybe_finish(job)

        job.updated_at = _now()
        self.store.save(job)

    @staticmethod
    def _is_sufficient(step: PlanStep) -> bool:
        """A SUCCEEDED step counts as done only if it never asked to be
        evaluated, or its persisted verdict says "sufficient"."""
        if step.status != StepStatus.SUCCEEDED:
            return False
        if not step.expectation:
            return True
        return (step.evaluation or {}).get("verdict") == EVALUATION_SUFFICIENT

    @staticmethod
    def _dependency_satisfied(plan: Plan, dep_id: str) -> bool:
        dep_step = plan.get(dep_id)
        return dep_step is not None and JobEngine._is_sufficient(dep_step)

    # -- result evaluation (only for steps that explicitly ask for it) --

    @staticmethod
    def _needs_evaluation(step: PlanStep) -> bool:
        return (
            step.status == StepStatus.SUCCEEDED
            and bool(step.expectation)
            and step.evaluation is None
        )

    @staticmethod
    def _is_empty_result(result: dict | None) -> bool:
        """Deterministic pre-check, independent of any AI: a result with no
        message text and no data at all can't satisfy any expectation."""
        if not result:
            return True
        return not result.get("message") and result.get("data") in (None, "", [], {})

    def _evaluate_pending(self, job: Job) -> bool:
        """Evaluate every SUCCEEDED step that carries an `expectation` and
        has no verdict yet; a non-sufficient verdict fails the step and goes
        through the SAME bounded failure/replan path a tool error does
        (`_handle_step_failure`). Returns whether anything was evaluated.

        Safe to repeat after a crash: evaluation only reads the step's
        stored result (never re-runs the tool), and a persisted verdict
        makes a step ineligible, so it is evaluated at most once."""
        pending = [s for s in job.plan.steps if self._needs_evaluation(s)]
        for step in pending:
            self._evaluate_step(job, step)
            if step.status == StepStatus.FAILED:
                self._handle_step_failure(job, step)
                if job.status != JobStatus.RUNNING:
                    break
        return bool(pending)

    def _evaluate_step(self, job: Job, step: PlanStep) -> None:
        verdict = EVALUATION_UNAVAILABLE
        reason = ""
        truncated = False

        if self._is_empty_result(step.result):
            verdict, reason = EVALUATION_INSUFFICIENT, "The step returned no data."
        elif self.evaluator is None:
            reason = "no result evaluator is configured"
        else:
            try:
                evaluation = self.evaluator.evaluate(
                    step.expectation or "", step.tool_name, copy.deepcopy(step.result)
                )
            except EvaluationError as exc:
                reason = str(exc)
            except Exception as exc:  # noqa: BLE001 - any evaluator fault is "unavailable", never "sufficient"
                reason = f"the evaluator failed: {exc}"
            else:
                verdict = EVALUATION_SUFFICIENT if evaluation.sufficient else EVALUATION_INSUFFICIENT
                reason = evaluation.reason
                truncated = evaluation.input_truncated

        step.evaluation = {"verdict": verdict, "reason": reason, "input_truncated": truncated}
        self._log(
            job,
            "step_evaluated",
            step_id=step.id,
            detail=f"{verdict}: {reason}",
        )

        if verdict == EVALUATION_INSUFFICIENT:
            step.status = StepStatus.FAILED
            step.error = (
                f"Result did not meet the step's expectation ({step.expectation!r}): {reason}"
            )
        elif verdict == EVALUATION_UNAVAILABLE:
            step.status = StepStatus.FAILED
            step.error = (
                f"Result could not be evaluated ({reason}); it was not treated as sufficient."
            )

        # Persist the verdict (and any failure) immediately, so a restart
        # never re-evaluates or lets a dependent run on an unvetted result.
        job.updated_at = _now()
        self.store.save(job)

    def _cascade_skips(self, job: Job) -> bool:
        """Mark any PENDING step whose dependency terminally FAILED or
        was itself SKIPPED as SKIPPED too - it can never become eligible
        to run. Returns whether anything changed, so `_advance`'s loop
        knows to re-scan for newly-ready steps."""
        changed = False
        for step in job.plan.steps:
            if step.status != StepStatus.PENDING:
                continue
            blocking = [
                dep
                for dep in step.depends_on
                if (dep_step := job.plan.get(dep)) is not None
                and dep_step.status in (StepStatus.FAILED, StepStatus.SKIPPED)
            ]
            if blocking:
                step.status = StepStatus.SKIPPED
                step.error = f"Skipped: dependency '{blocking[0]}' did not succeed."
                self._log(job, "step_skipped", step_id=step.id, detail=step.error)
                changed = True
        return changed

    def _needs_confirmation(self, step: PlanStep, job: Job) -> bool:
        if step.confirmed:
            return False
        permission = PermissionLevel(step.permission) if step.permission else PermissionLevel.READ
        if permission in AUTOMATIC_PERMISSIONS:
            return False
        if permission is PermissionLevel.DANGEROUS:
            # Structural, not configurable: DANGEROUS always requires
            # confirmation, regardless of autonomy or `job.
            # autonomous_permissions` - see AutonomyLevel's docstring and
            # parse_autonomous_permissions (app/orchestration/base.py),
            # which makes it impossible to even construct a Job whose
            # autonomous_permissions contains "dangerous" in the first
            # place. This check is the second, independent layer of that
            # same guarantee.
            return True
        # permission is CONFIRM here: gated unless THIS Job's own
        # configured `autonomous_permissions` (not a hardcoded engine
        # constant) says AUTONOMOUS may auto-run it.
        if job.autonomy is AutonomyLevel.AUTONOMOUS and permission.value in job.autonomous_permissions:
            return False
        return True

    def _confirmed_value(self, step: PlanStep, job: Job) -> bool | None:
        permission = PermissionLevel(step.permission) if step.permission else PermissionLevel.READ
        if permission in AUTOMATIC_PERMISSIONS:
            return None  # ToolRegistry ignores `confirmed` for automatic permissions anyway
        if step.confirmed:
            return True
        if (
            permission is PermissionLevel.CONFIRM
            and job.autonomy is AutonomyLevel.AUTONOMOUS
            and permission.value in job.autonomous_permissions
        ):
            return True
        return None

    def _build_confirmation_message(self, job: Job, steps: list[PlanStep]) -> str:
        described = "; ".join(
            f"{s.tool_name}({self._describe_arguments(job, s)})" for s in steps
        )
        plural = "s" if len(steps) != 1 else ""
        return (
            f"Jarvis is ready to run {len(steps)} step{plural} that "
            f"need{'s' if len(steps) == 1 else ''} your confirmation: "
            f"{described}. Confirm to proceed?"
        )

    # -- step references --

    def _resolve_step_arguments(self, job: Job, step: PlanStep) -> dict:
        """`step.arguments` with every step reference replaced by the value
        it points at in the (already SUCCEEDED) source step's stored
        result - see app/orchestration/references.py. Returns
        the resolved arguments. Raises
        ReferenceError for anything that can't be resolved exactly, and,
        when the step contained references, for a resolved argument set
        the target tool's own schema (types/required/unexpected) or
        `validate_arguments` rejects. A step with no references is
        returned unchanged and unchecked here - plan validation already
        covered it, exactly as before references existed."""
        resolved: dict = {}
        used_reference = False
        for name, value in step.arguments.items():
            if not looks_like_reference(value):
                resolved[name] = value
                continue
            used_reference = True
            from_step, _segments = validate_reference(value)
            if from_step not in step.depends_on:
                raise ReferenceError(
                    f"argument '{name}' references '{from_step}', which is not one of "
                    "this step's declared dependencies"
                )
            source = job.plan.get(from_step) if job.plan else None
            if source is None or not self._is_sufficient(source):
                raise ReferenceError(
                    f"argument '{name}' references step '{from_step}', which has not "
                    "succeeded (or its result was not judged sufficient)"
                )
            try:
                resolved[name] = resolve_reference(value, source.result)
            except ReferenceError as exc:
                raise ReferenceError(f"argument '{name}': {exc}") from exc

        if used_reference:
            tool = self.tools.get(step.tool_name)
            if tool is None:
                raise ReferenceError(f"unknown tool '{step.tool_name}'")
            try:
                validate_resolved_arguments(tool, step.tool_name, resolved)
                tool.validate_arguments(**resolved)
            except (PlanValidationError, ToolArgumentError) as exc:
                raise ReferenceError(f"resolved arguments are not acceptable: {exc}") from exc
        return resolved

    def _describe_arguments(self, job: Job, step: PlanStep, limit: int = 200) -> str:
        """A human-readable, bounded view of what a step WILL be called
        with, for a confirmation message: references are shown resolved,
        so the user approves the actual content rather than an opaque
        pointer. Falls back to the raw arguments if they can't be
        resolved (the step would then fail rather than run)."""
        try:
            arguments = self._resolve_step_arguments(job, step)
        except ReferenceError:
            arguments = step.arguments
        return ", ".join(
            f"{name}={self._shorten(value, limit)}" for name, value in arguments.items()
        )

    @staticmethod
    def _shorten(value, limit: int) -> str:
        rendered = repr(value) if isinstance(value, str) else json.dumps(value, default=str)
        return rendered if len(rendered) <= limit else rendered[:limit] + "...[truncated]"

    def _run_step(self, job: Job, step: PlanStep) -> None:
        denial = self._profile_denial(job, step)
        if denial is not None:
            # The run profile forbids this tool (checked against the live registry, whatever the plan says): the tool is never
            # called. It fails like any other step and the bounded failure/replan path decides what happens next.
            step.status = StepStatus.FAILED
            step.error = denial
            self._log(job, "step_blocked_by_profile", step_id=step.id, detail=denial)
            job.updated_at = _now()
            self.store.save(job)
            return

        try:
            arguments = self._resolve_step_arguments(job, step)
        except ReferenceError as exc:
            # Deterministic, so never retried and the tool is never called:
            # the step simply fails and the existing bounded failure/replan
            # path decides what happens next.
            step.status = StepStatus.FAILED
            step.error = f"Could not resolve a step reference: {exc}"
            self._log(job, "step_reference_failed", step_id=step.id, detail=step.error)
            job.updated_at = _now()
            self.store.save(job)
            return

        tool = self.tools.get(step.tool_name)
        if tool is not None and tool.wants_job_provenance:
            # Set by the engine, never by a plan: the tool's advertised
            # parameters don't include these, so a plan can't supply them.
            arguments = {**arguments, "source_job_id": job.id, "source_step_id": step.id}

        step.status = StepStatus.RUNNING
        step.attempts += 1
        job.updated_at = _now()
        self.store.save(job)  # persist BEFORE the (possibly irreversible) call - see module docstring
        self._log(job, "step_started", step_id=step.id, detail=f"attempt {step.attempts}")
        try:
            result = self.tools.execute(
                step.tool_name,
                params=arguments,
                confirmed=self._confirmed_value(step, job),
            )
        except Exception as exc:  # noqa: BLE001 - a safety net; ToolRegistry already isolates tool failures
            result = ToolResult.error(f"Executor error running '{step.tool_name}': {exc}")

        step.result = _serialize_tool_result(result)

        if result.status is ToolResultStatus.SUCCESS:
            step.status = StepStatus.SUCCEEDED
            step.error = None
            self._log(job, "step_succeeded", step_id=step.id, detail=result.message)
        elif result.status is ToolResultStatus.DENIED:
            step.status = StepStatus.FAILED
            step.error = result.message or "Permission denied."
            self._log(job, "step_denied", step_id=step.id, detail=step.error)
        elif result.status is ToolResultStatus.CONFIRMATION_REQUIRED:
            # Shouldn't happen given `_needs_confirmation` gates this
            # before `_run_step` is ever called - handled defensively
            # anyway: revert to PENDING so the next `_advance` pass's
            # confirmation gate picks it up cleanly rather than
            # recording a spurious failure.
            step.status = StepStatus.PENDING
            step.attempts -= 1
            self._log(job, "step_confirmation_unexpected", step_id=step.id)
        else:  # ERROR
            if step.attempts < step.retry_policy.max_attempts:
                step.status = StepStatus.PENDING
                self._log(
                    job,
                    "step_retry_scheduled",
                    step_id=step.id,
                    detail=result.message,
                )
                if step.retry_policy.delay_seconds > 0:
                    self._sleep(step.retry_policy.delay_seconds)
            else:
                step.status = StepStatus.FAILED
                step.error = result.message or "Tool execution failed."
                self._log(job, "step_failed", step_id=step.id, detail=step.error)

        # Persist this step's OUTCOME immediately, not just its RUNNING
        # start (see the save above and this module's docstring). Without
        # this, a step that actually succeeded stays recorded as RUNNING
        # on disk until either the next step's own pre-call save or
        # `_advance`'s trailing save - a crash inside that window would
        # make `_recover_interrupted_steps` treat an already-SUCCEEDED
        # step as interrupted and potentially replan an alternative
        # action that duplicates one that already really happened.
        # Saving here closes that window: at most the single step
        # actually mid-flight at crash time can ever be found RUNNING on
        # restart.
        job.updated_at = _now()
        self.store.save(job)

    def _handle_step_failure(self, job: Job, step: PlanStep) -> None:
        """A step exhausted its retries. Decide whether replanning is
        worth attempting (bounded by max_replans and max_total_steps) or
        the Job should simply fail - never an uncontrolled replanning
        loop."""
        if self._enforce_deadline(job):
            return  # out of time: no replanning call is made
        max_steps, max_total_steps, max_replans = self._limits(job)
        if job.replan_count >= max_replans:
            job.status = JobStatus.FAILED
            job.error = (
                f"Step '{step.id}' ({step.tool_name}) failed: {step.error} "
                f"(replan limit of {max_replans} reached)."
            )
            self._log(job, "job_failed", detail=job.error)
            return

        completed = [s for s in job.plan.steps if s.status == StepStatus.SUCCEEDED]
        remaining = [
            s
            for s in job.plan.steps
            if s.status == StepStatus.PENDING and s.id != step.id
        ]
        known_ids = frozenset(s.id for s in completed)

        self._refresh_context(job)  # a replan is a planning call too - see _refresh_context
        available_tools = self._plannable_tools(job)
        profiled = job.run_profile != INTERACTIVE_PROFILE
        if profiled and not available_tools:
            job.status = JobStatus.FAILED
            job.error = (
                f"Step '{step.id}' ({step.tool_name}) failed: {step.error}. "
                "Replanning is not possible: no capability is available for this kind of run."
            )
            self._log(job, "replan_failed", detail=job.error)
            return
        try:
            raw_new_steps = self.planner.plan(
                job.goal,
                available_tools=available_tools,
                context=self._planning_context(job),
                autonomy=job.autonomy,
                replan_context=ReplanContext(
                    completed_steps=completed,
                    failed_step=step,
                    failure_reason=step.error or "unknown error",
                    remaining_steps=remaining,
                ),
            )
            if len(completed) + len(raw_new_steps) > max_total_steps:
                raise PlanValidationError(
                    f"Replan would exceed the total step safety limit of "
                    f"{max_total_steps}."
                )
            new_steps = validate_plan(
                raw_new_steps,
                self.tools,
                max_steps=max_steps,
                default_retry=RetryPolicy(
                    max_attempts=self.max_retries_per_step,
                    delay_seconds=self.retry_delay_seconds,
                ),
                known_step_ids=known_ids,
                allowed_tool_names=frozenset(t.name for t in available_tools) if profiled else None,
            )
        except (PlanningError, PlanValidationError) as exc:
            job.status = JobStatus.FAILED
            job.error = (
                f"Step '{step.id}' ({step.tool_name}) failed: {step.error}. "
                f"Replanning also failed: {exc}"
            )
            self._log(job, "replan_failed", detail=str(exc))
            return

        job.plan.steps = completed + new_steps
        job.total_step_count += len(new_steps)
        job.replan_count += 1
        self._log(
            job,
            "replanned",
            step_id=step.id,
            detail=f"{len(new_steps)} replacement step(s) after '{step.id}' failed",
        )
        # job.status stays RUNNING - `_advance`'s own loop re-scans for
        # newly-ready steps on its next iteration.

    def _maybe_finish(self, job: Job) -> None:
        statuses = {step.status for step in job.plan.steps}
        if not statuses <= TERMINAL_STEP_STATUSES:
            return  # something is still PENDING/RUNNING - not done yet
        if any(self._needs_evaluation(step) for step in job.plan.steps):
            return  # a step is still waiting for its verdict - not done yet

        failed = [s for s in job.plan.steps if s.status == StepStatus.FAILED]
        if failed:
            job.status = JobStatus.FAILED
            names = ", ".join(f"{s.id} ({s.tool_name})" for s in failed)
            job.error = f"Job failed - unresolved failed step(s): {names}."
            self._log(job, "job_failed", detail=job.error)
            return

        job.status = JobStatus.COMPLETED
        # The user-facing result is built from the plan's TERMINAL steps
        # (plus a report-producing step's synthesis), not a transcript of
        # every tool message - see app/orchestration/report.py.
        summary = build_final_report(job.plan.steps, produces_report=self._step_produces_report)
        job.final_result = summary
        self._log(job, "job_completed", detail=" ".join(summary.split())[:300])

    def _step_produces_report(self, step: PlanStep) -> bool:
        tool = self.tools.get(step.tool_name)
        return tool is not None and tool.produces_report

    def _log(self, job: Job, kind: str, *, step_id: str | None = None, detail: str = "") -> None:
        job.events.append(
            JobEvent(timestamp=_now(), kind=kind, detail=detail or "", step_id=step_id)
        )
