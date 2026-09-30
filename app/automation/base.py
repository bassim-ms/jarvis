"""
Automation abstraction (M8): persistent, tool-backed automations -
"when TRIGGER (and CONDITIONS hold), run ACTION" - built entirely on
architecture M1-M7 already established, not a parallel system:

- Persistence follows the exact SQLiteReminderStore/SQLiteMemoryStore
  pattern (see app/automation/sqlite_store.py).
- A recurring TIME trigger reuses Recurrence/RecurrenceUnit/
  parse_recurrence/advance_by_recurrence (app/reminders/base.py)
  verbatim - the same fixed-interval schedule a recurring Reminder
  already uses, not a second copy of that interval math.
- An Automation's action is a plain `ToolCall` (app/tools/decision.py) -
  the same type the Tool Decision Layer already uses to represent "run
  this registered tool with these parameters." AutomationEngine
  (app/automation/engine.py) only ever dispatches it through
  `ToolRegistry.execute(tool_name, params=...)`, never `confirmed=True` -
  so an automation can only ever run a tool whose permission is already
  automatic (READ/SAFE_ACTION; see app/tools/base.py's
  AUTOMATIC_PERMISSIONS). CreateAutomationTool (app/automation/tools.py)
  rejects a CONFIRM/DANGEROUS tool at creation time, and the engine
  never overrides that by passing `confirmed=True` - the whole security
  boundary this milestone adds is that persisted automation state can
  only ever mean "call this already-permitted tool with these
  parameters," never arbitrary Python/shell, and never a tool the user
  hasn't already granted unattended trust to.

An Automation is deliberately NOT an agent: it has no planning loop, no
multi-step reasoning, and its action is a single, fixed, pre-validated
ToolCall chosen at creation time - never one an AI decides at run time.
Nothing here blocks a future milestone from registering new kinds of
Tools (e.g. for writing code, running tests, or driving hardware) that an
Automation could then invoke like any other tool, once such a tool
exists and is deliberately given an automatic permission level - this
module just doesn't add one itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from app.reminders.base import Recurrence
from app.tools.decision import ToolCall


#: Tools that must NEVER run as an automation's action, however the
#: automation came to exist: they answer a question or confirmation a Job is
#: waiting on the USER for. An unattended trigger approving (or declining)
#: something on the user's behalf would defeat the confirmation. Checked when
#: an automation is created (CreateAutomationTool) AND again when one runs
#: (AutomationEngine), so a stored automation can't get around it. Interaction
#: replies themselves are not tools at all - they only arrive through the chat
#: resolver - so this list is the whole boundary on the tool side.
FORBIDDEN_ACTION_TOOL_NAMES: frozenset[str] = frozenset({"confirm_job"})


class AutomationStatus(Enum):
    """Lifecycle state of an Automation.

    ENABLED - eligible to run: due (TIME), observed (REMINDER_COMPLETED),
      or manually invoked (any trigger type, via run_automation).
    DISABLED - paused; AutomationEngine never considers it due, but it
      can be re-enabled. Also the state a one-shot trigger (a
      non-recurring TIME trigger, or REMINDER_COMPLETED) puts itself
      into after it fires once, so it never fires twice - the same
      "retire after the one occurrence" idea as a one-time Reminder
      becoming TRIGGERED (app/reminders/base.py), reusing DISABLED
      rather than inventing a parallel status for it.
    CANCELLED - terminal; the user explicitly cancelled it. Can never be
      re-enabled.
    """

    ENABLED = "enabled"
    DISABLED = "disabled"
    CANCELLED = "cancelled"


#: Terminal states an automation can never leave.
TERMINAL_AUTOMATION_STATUSES = frozenset({AutomationStatus.CANCELLED})


class TriggerType(Enum):
    """What makes an Automation eligible to run."""

    TIME = "time"
    MANUAL = "manual"
    REMINDER_COMPLETED = "reminder_completed"


@dataclass
class Trigger:
    """What makes an Automation due. Only the fields relevant to `type`
    are populated - the same "one small dataclass, per-variant optional
    fields" shape CapabilityDecision (app/capability.py) already uses for
    its own optional `tool_name`, rather than a class hierarchy for three
    trigger kinds.

    TIME: `recurrence` given -> the automation repeats on that fixed
      interval, exactly like a recurring Reminder, advancing forever
      until disabled/cancelled. `recurrence` omitted -> fires once (at
      the Automation's own `next_run_at`), then self-disables.
    MANUAL: no fields used. AutomationEngine's background poll never
      treats a MANUAL-trigger automation as due - it only ever runs
      through the `run_automation` tool (app/automation/tools.py).
    REMINDER_COMPLETED: `reminder_id` names a Reminder
      (app.reminders.base.Reminder). The automation is due once that
      reminder's status becomes TRIGGERED or COMPLETED - never
      CANCELLED, which means "never mind," not "done." Since a Reminder
      can only leave PENDING once (see app/reminders/base.py), this is
      inherently a one-shot check; the automation self-disables the
      first time it's observed.
    """

    type: TriggerType
    recurrence: Recurrence | None = None
    reminder_id: str | None = None


@dataclass
class ConditionContext:
    """Read-only snapshot AutomationEngine builds once per check and
    hands to every Condition - the same "compute shared context once,
    pass it down" shape JarvisCore._format_context (app/core.py) already
    uses for facts/history. A Condition never fetches anything itself."""

    now: datetime
    facts: dict[str, str]


class Condition(ABC):
    """A basic, deterministic (no AI, no network) yes/no check gating
    whether an already-due Automation's action actually runs this time.
    Never gates the trigger's own scheduling - a skipped occurrence still
    advances/retires the trigger exactly as a run one would (see
    AutomationEngine._advance_trigger, app/automation/engine.py)."""

    @abstractmethod
    def evaluate(self, context: ConditionContext) -> bool:
        """Return whether this condition currently holds."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a small, JSON-friendly {"type": ..., ...params}
        dict - AutomationStore persists an Automation's conditions as
        JSON (see app/automation/sqlite_store.py). `condition_from_dict`
        below is the inverse."""


class ConditionError(ValueError):
    """Raised by `condition_from_dict` for an unknown type or malformed
    params - never silently ignored or dropped."""


@dataclass
class FactEqualsCondition(Condition):
    """True only if a stored user fact (MemoryStore.get_facts(),
    app/memory/base.py) equals a fixed value exactly - deterministic
    string comparison, no fuzzy matching. False (not an error) if the key
    was never set."""

    key: str
    value: str

    def evaluate(self, context: ConditionContext) -> bool:
        return context.facts.get(self.key) == self.value

    def to_dict(self) -> dict[str, Any]:
        return {"type": "fact_equals", "key": self.key, "value": self.value}


@dataclass
class TimeWindowCondition(Condition):
    """True only while the current UTC hour-of-day is within
    [start_hour, end_hour) - e.g. start_hour=9, end_hour=18 means
    "only between 09:00 and 17:59 UTC". Wraps past midnight when
    start_hour > end_hour (e.g. 22 -> 6 means 22:00-23:59 or
    00:00-05:59)."""

    start_hour: int
    end_hour: int

    def evaluate(self, context: ConditionContext) -> bool:
        hour = context.now.hour
        if self.start_hour <= self.end_hour:
            return self.start_hour <= hour < self.end_hour
        return hour >= self.start_hour or hour < self.end_hour

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "time_window",
            "start_hour": self.start_hour,
            "end_hour": self.end_hour,
        }


def condition_from_dict(data: dict[str, Any]) -> Condition:
    """Reconstruct a Condition from its `to_dict()` form - the inverse,
    used only by AutomationStore when loading a persisted Automation.
    Raises ConditionError for an unrecognized type or missing params,
    never a guess."""
    condition_type = data.get("type")
    if condition_type == "fact_equals":
        return FactEqualsCondition(key=data["key"], value=data["value"])
    if condition_type == "time_window":
        return TimeWindowCondition(
            start_hour=data["start_hour"], end_hour=data["end_hour"]
        )
    raise ConditionError(f"Unknown condition type: {condition_type!r}")


class ExecutionOutcome(Enum):
    """Result of one AutomationEngine execution attempt."""

    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"  # a condition didn't hold; the action never ran


@dataclass
class AutomationExecution:
    """One record of an Automation actually being checked/run - the
    persisted execution history this milestone requires, independent of
    the Automation's own current state (mirrors ReminderExecution,
    app/reminders/base.py)."""

    automation_id: str
    executed_at: str
    outcome: ExecutionOutcome
    detail: str


@dataclass
class Automation:
    """A single persisted automation: trigger + conditions + action.

    `next_run_at` is only meaningful for a TIME trigger (the next time
    it's due, ISO 8601 UTC) - always None for MANUAL/REMINDER_COMPLETED,
    which are never due "at" a time."""

    id: str
    name: str
    status: AutomationStatus
    trigger: Trigger
    conditions: list[Condition]
    action: ToolCall
    next_run_at: str | None
    created_at: str
    updated_at: str


class AutomationStore(ABC):
    """Persistent storage for automations and their execution history."""

    @abstractmethod
    def create(
        self,
        name: str,
        trigger: Trigger,
        conditions: list[Condition],
        action: ToolCall,
        next_run_at: str | None = None,
    ) -> Automation:
        """Create and persist a new ENABLED automation."""

    @abstractmethod
    def get(self, automation_id: str) -> Automation | None:
        """Look up an automation by ID, or None if it doesn't exist."""

    @abstractmethod
    def list(self, status: AutomationStatus | None = None) -> list[Automation]:
        """Return automations, optionally filtered to one status, most
        recently created first."""

    @abstractmethod
    def record_execution(
        self,
        automation_id: str,
        executed_at: str,
        outcome: ExecutionOutcome,
        detail: str = "",
        next_run_at: str | None = None,
        disable: bool = False,
    ) -> None:
        """Always appends an AutomationExecution row, regardless of
        `outcome`.

        `next_run_at` given advances the automation's own schedule (a
        recurring TIME trigger, whether this run's outcome was SUCCESS,
        FAILURE, or SKIPPED - a condition gates the action, never the
        trigger's own cadence). `disable` True instead flips it to
        DISABLED (a one-shot trigger retiring itself after being
        checked). AutomationEngine never passes both for the same call.
        """

    @abstractmethod
    def get_executions(self, automation_id: str) -> list[AutomationExecution]:
        """Return every recorded execution for `automation_id`, oldest
        first."""

    @abstractmethod
    def enable(self, automation_id: str) -> bool:
        """Move an automation to ENABLED (from DISABLED, or a no-op if
        already ENABLED). Returns False if the ID doesn't exist or is
        CANCELLED (terminal)."""

    @abstractmethod
    def disable(self, automation_id: str) -> bool:
        """Move an automation to DISABLED (from ENABLED, or a no-op if
        already DISABLED). Same return-value contract as `enable`."""

    @abstractmethod
    def cancel(self, automation_id: str) -> bool:
        """Terminally cancel an automation. Returns False if the ID
        doesn't exist or is already CANCELLED."""
