"""
Automation Engine (M8): a lightweight background poller that turns due
Automations into ToolRegistry executions - the direct sibling of
ReminderScheduler (app/reminders/scheduler.py), reusing the exact same
"poll on a daemon thread, isolate every per-item failure, isolate the
poll itself too" shape, including the two-layer exception isolation
ReminderScheduler was hardened with (see its own `_run`/`tick`) - a
single misbehaving automation, or even a store-level failure, can never
kill the background thread (see `_run`/`tick` below).

Deliberately not a general task-automation/agent engine: `_execute` only
ever calls `ToolRegistry.execute(tool_name, params=..., confirmed=None)`
for an already-validated action (a plain `ToolCall`, app/tools/
decision.py) - it never passes `confirmed=True`, never runs Python/shell
directly, and never invents a tool call from anything other than the
Automation's own persisted action. Permission enforcement is entirely
ToolRegistry's, unchanged - this milestone adds no second gate and no
bypass; CreateAutomationTool (app/automation/tools.py) additionally
rejects a CONFIRM/DANGEROUS tool at creation time, so an automation whose
target tool somehow isn't automatic just fails its execution (recorded,
never silently approved) rather than ever being force-run.
"""

import threading
from datetime import datetime, timezone

from app.automation.base import (
    FORBIDDEN_ACTION_TOOL_NAMES,
    Automation,
    AutomationStatus,
    AutomationStore,
    ConditionContext,
    ExecutionOutcome,
    TriggerType,
)
from app.memory.base import MemoryStore
from app.reminders.base import ReminderStatus, ReminderStore, advance_by_recurrence
from app.tools.base import ToolResultStatus
from app.tools.registry import ToolRegistry

#: How often AutomationEngine polls by default when none is configured -
#: matches ReminderScheduler's own default (app/reminders/scheduler.py),
#: frequent enough that a due automation runs within about half a minute,
#: without hammering AutomationStore's SQLite connection.
DEFAULT_POLL_INTERVAL_SECONDS = 30.0

#: Reminder statuses that count as "the linked task completed" for a
#: REMINDER_COMPLETED trigger - CANCELLED is deliberately excluded, since
#: cancelling a reminder means "never mind," not "done."
_REMINDER_DONE_STATUSES = frozenset({ReminderStatus.TRIGGERED, ReminderStatus.COMPLETED})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AutomationRunError(RuntimeError):
    """Raised by `run_now` for an automation that can't be run right now
    - unknown ID, or not ENABLED. Never raised by the background poller
    itself, which simply skips anything not due."""


class AutomationEngine:
    """Polls `store` for due automations and executes each one's action
    through `tools`, on its own background thread.

    `tick()` is the actual check-and-run step, fully synchronous and with
    all its side effects visible before it returns - tests call it
    directly instead of racing a background thread; `start()`/`stop()`
    just wrap it in a daemon polling loop for the running application.
    `run_now()` executes a single automation immediately regardless of
    its trigger type - the only way a MANUAL-trigger automation ever
    runs, and also usable to force an early run of a TIME/
    REMINDER_COMPLETED one (see app/automation/tools.py's
    run_automation).
    """

    def __init__(
        self,
        store: AutomationStore,
        tools: ToolRegistry,
        reminder_store: ReminderStore,
        memory: MemoryStore,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ):
        self.store = store
        self.tools = tools
        self.reminder_store = reminder_store
        self.memory = memory
        self.poll_interval_seconds = poll_interval_seconds
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the background polling thread. No-op if already running."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="jarvis-automation-engine", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float | None = 5.0) -> None:
        """Signal the background thread to stop and wait for it to exit.
        No-op if it isn't running."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - a failed poll must not kill the loop
                # E.g. store.list() itself failing (a transient SQLite
                # error, a closed connection, ...) - outside tick()'s own
                # per-automation isolation below. Swallowed rather than
                # logged since this codebase has no logging setup yet;
                # the next poll simply retries.
                pass
            self._stop_event.wait(self.poll_interval_seconds)

    def tick(self) -> int:
        """Execute every currently-due automation once and return how
        many were successfully processed (regardless of whether the
        action itself succeeded, failed, or was skipped by a condition).

        Never raises past a single automation: each one is isolated in
        its own try/except, the same failure-isolation philosophy
        ToolRegistry.execute (app/tools/registry.py) already applies to a
        single misbehaving tool, and ReminderScheduler.tick
        (app/reminders/scheduler.py) already applies to a single
        misbehaving reminder - so one automation failing in an
        unexpected way can't stop the rest of this poll, and can't kill
        the background thread (see `_run`)."""
        due = self._collect_due(_now())
        processed = 0
        for automation in due:
            try:
                self._execute(automation)
            except Exception:  # noqa: BLE001 - isolate one automation's failure
                continue
            processed += 1
        return processed

    def run_now(self, automation_id: str) -> None:
        """Execute one automation immediately, regardless of its trigger
        type or whether it's currently "due." Raises AutomationRunError
        if the ID is unknown or the automation isn't ENABLED (DISABLED/
        CANCELLED) - the caller (run_automation, app/automation/tools.py)
        turns that into a ToolResult.error rather than a 502; any failure
        *during* the run itself is still recorded as a normal execution,
        exactly like a poller-triggered run."""
        automation = self.store.get(automation_id)
        if automation is None:
            raise AutomationRunError(f"Unknown automation: '{automation_id}'")
        if automation.status is not AutomationStatus.ENABLED:
            raise AutomationRunError(
                f"Automation '{automation_id}' is {automation.status.value}, "
                "not enabled."
            )
        self._execute(automation)

    def _collect_due(self, now: str) -> list[Automation]:
        """Every ENABLED automation whose trigger currently holds.
        MANUAL automations are never included - see `run_now`.

        Each automation's own due-check is isolated in its own
        try/except - the same per-item failure-isolation philosophy
        `tick()` already applies to *executing* a due automation, applied
        here to *checking* one: a single automation with an unexpected
        problem (e.g. `reminder_store.get()` raising for a corrupt
        trigger) can't stop every other automation from being collected
        and run this poll - it's simply skipped for this cycle, same as
        `tick()` skips one that fails during execution."""
        due = []
        for automation in self.store.list(status=AutomationStatus.ENABLED):
            try:
                is_due = self._is_due(automation, now)
            except Exception:  # noqa: BLE001 - isolate one automation's due-check
                continue
            if is_due:
                due.append(automation)
        return due

    def _is_due(self, automation: Automation, now: str) -> bool:
        trigger = automation.trigger
        if trigger.type is TriggerType.TIME:
            return automation.next_run_at is not None and automation.next_run_at <= now
        if trigger.type is TriggerType.REMINDER_COMPLETED:
            reminder = self.reminder_store.get(trigger.reminder_id)
            return reminder is not None and reminder.status in _REMINDER_DONE_STATUSES
        return False  # TriggerType.MANUAL: never polled.

    def _execute(self, automation: Automation) -> None:
        context = ConditionContext(
            now=datetime.now(timezone.utc), facts=self.memory.get_facts()
        )
        conditions_met = all(c.evaluate(context) for c in automation.conditions)

        if automation.action.tool_name in FORBIDDEN_ACTION_TOOL_NAMES:
            # Defence in depth (creation already refuses these): an automation
            # must never answer a confirmation for the user, even one that got
            # into the store some other way. Recorded as a failure; nothing runs.
            outcome = ExecutionOutcome.FAILURE
            detail = "this action can't be run by an automation"
        elif conditions_met:
            result = self.tools.execute(
                automation.action.tool_name, params=automation.action.parameters
            )
            outcome = (
                ExecutionOutcome.SUCCESS
                if result.status is ToolResultStatus.SUCCESS
                else ExecutionOutcome.FAILURE
            )
            detail = result.message or result.status.value
        else:
            outcome = ExecutionOutcome.SKIPPED
            detail = "a condition was not met"

        next_run_at, disable = self._advance_trigger(automation)

        self.store.record_execution(
            automation.id,
            executed_at=_now(),
            outcome=outcome,
            detail=detail,
            next_run_at=next_run_at,
            disable=disable,
        )

    @staticmethod
    def _advance_trigger(automation: Automation) -> tuple[str | None, bool]:
        """Decide how this run affects the trigger's own future
        scheduling - entirely independent of whether a condition allowed
        the action to actually run this time (see `_execute`): a skipped
        occurrence still advances/retires the trigger exactly as a run
        one would, so a permanently-false condition can't spam an
        occurrence every single poll interval forever.

        Returns (next_run_at, disable) for
        AutomationStore.record_execution: a recurring TIME trigger
        advances; a one-time TIME trigger or a REMINDER_COMPLETED trigger
        (both inherently one-shot) disables; MANUAL does neither - it
        stays ENABLED for as many manual runs as the user likes."""
        trigger = automation.trigger
        if trigger.type is TriggerType.TIME:
            if trigger.recurrence is not None:
                return advance_by_recurrence(automation.next_run_at, trigger.recurrence), False
            return None, True
        if trigger.type is TriggerType.REMINDER_COMPLETED:
            return None, True
        return None, False
