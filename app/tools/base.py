"""
Tool abstraction.

The Tool Registry and Jarvis Core depend only on this interface, never on
any specific tool's implementation, so new tools (web, email, calendar,
smart home, robotics, printers, etc.) can be added later without touching
the Registry or Core.

This module is architecture only - no real tools are implemented here.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PermissionLevel(Enum):
    """How much user trust a tool's execution requires.

    Enforcement lives in the Tool Registry (see app/tools/registry.py), not
    in any AI provider, so a tool stays safe regardless of what's calling
    it - Gemini, a future provider, or nothing at all.
    """

    READ = "read"                # automatic - read-only, no side effects
    SAFE_ACTION = "safe_action"  # automatic - low-risk side effects
    CONFIRM = "confirm"          # requires user confirmation
    DANGEROUS = "dangerous"      # requires explicit/strong user confirmation


# Permission levels that never require confirmation before running.
AUTOMATIC_PERMISSIONS = frozenset({PermissionLevel.READ, PermissionLevel.SAFE_ACTION})


class ToolResultStatus(Enum):
    """Outcome categories for a tool execution attempt."""

    SUCCESS = "success"
    ERROR = "error"
    DENIED = "denied"
    CONFIRMATION_REQUIRED = "confirmation_required"


@dataclass
class ToolResult:
    """Structured outcome of a tool execution attempt.

    Callers (eventually Jarvis Core, an AI provider's tool-calling loop, or
    a UI) branch on `status` rather than parsing free-form strings.
    """

    status: ToolResultStatus
    data: Any = None
    message: str = ""
    permission: PermissionLevel | None = None

    @classmethod
    def success(cls, data: Any = None, message: str = "") -> "ToolResult":
        return cls(status=ToolResultStatus.SUCCESS, data=data, message=message)

    @classmethod
    def error(cls, message: str) -> "ToolResult":
        return cls(status=ToolResultStatus.ERROR, message=message)

    @classmethod
    def denied(cls, message: str) -> "ToolResult":
        return cls(status=ToolResultStatus.DENIED, message=message)

    @classmethod
    def confirmation_required(
        cls, permission: PermissionLevel, message: str = ""
    ) -> "ToolResult":
        return cls(
            status=ToolResultStatus.CONFIRMATION_REQUIRED,
            message=message,
            permission=permission,
        )


@dataclass
class ToolParameter:
    """Describes one argument a tool's `execute` accepts.

    Deliberately simple and provider-agnostic (plain strings for types)
    rather than tied to any one AI provider's function-calling schema -
    AI tool-calling itself is a later milestone.
    """

    name: str
    type: str  # e.g. "string", "number", "boolean", "object", "structured" (object or list)
    description: str = ""
    required: bool = True
    default: Any = None
    #: Only meaningful for `type == "string"`. False (the default, fail-
    #: closed): in an explicit tool command (app/tools/command.py) an
    #: UNQUOTED value is exactly one whitespace-delimited token - anything
    #: longer must be quoted, so trailing words/clauses can never be
    #: silently absorbed into the value (an `open_url` URL must never
    #: swallow ", wait for 5 seconds"). True: the value is prose (a task's
    #: text, a search query), so an unquoted value may span every
    #: remaining word of its step, up to the next step separator.
    free_text: bool = False


@dataclass
class ToolRequirements:
    """Optional metadata about what a tool needs in order to actually run.

    Purely descriptive for now - nothing enforces these automatically yet.
    Future tools declare what they depend on here, so a future "is this
    tool usable right now" check has a standard place to look, without
    Core or the Registry needing to know about any specific dependency.
    """

    dependencies: list[str] = field(default_factory=list)  # e.g. python packages
    env_vars: list[str] = field(default_factory=list)  # e.g. required settings/secrets
    network: bool = False


class ToolArgumentError(ValueError):
    """Raised by `Tool.validate_arguments` when an argument VALUE is
    unacceptable (disallowed URL scheme, off-screen coordinate, ...) -
    distinct from a schema/type problem, which the explicit-command layer
    (app/tools/command.py) already checks against `Tool.parameters`."""


class Tool(ABC):
    """A single capability Rhasspy can invoke.

    Implementations declare their own name, description, parameters, and
    permission level, and provide `execute`. Nothing outside a Tool's own
    module needs to know its concrete type - see app/tools/registry.py.
    """

    #: Tools with no external dependencies can leave this at the default.
    requirements: ToolRequirements = ToolRequirements()

    #: True only for a tool whose arguments are genuinely prose that an AI
    #: ToolCallProvider is allowed to extract (e.g. "remind me to stretch
    #: in 10 minutes" -> text + due_in_minutes). Default False, fail-
    #: closed: an explicit command (app/tools/command.py) resolves this
    #: tool's arguments deterministically from `parameters` alone, and
    #: anything it can't resolve is rejected - never handed to an LLM.
    natural_language_arguments: bool = False

    #: False for a tool that manages Jarvis's own autonomy machinery - Job
    #: control (create/confirm/pause/resume/cancel) and automation
    #: management. Such a tool stays fully usable from chat and by an
    #: AutomationEngine the user configured, but M15's planner is never
    #: offered it and `validate_plan` rejects a step naming it: a goal-
    #: driven plan must not be able to approve its own confirmations,
    #: spawn Jobs with a higher autonomy, or schedule further automation.
    plannable: bool = True

    #: False for a tool an AUTOMATION may never be configured to run, nor run
    #: unattended, even though it is registered and even if it is READ or
    #: SAFE_ACTION: `plannable` keeps a goal-driven plan away from a tool;
    #: this keeps a schedule or trigger away from it. Both are read from the tool
    #: itself, so the registry stays the one place that says what a tool is. Used
    #: for the memory-write tools: persistent memory is written on the user's own
    #: authority (through the write policy), never by something running on its own.
    automatable: bool = True

    #: True for a tool that records where its output came from. JobEngine
    #: then adds `source_job_id`/`source_step_id` to the arguments it
    #: executes the tool with - values only the engine knows and a plan
    #: can never supply (they are not among the tool's declared
    #: `parameters`, so plan validation rejects them). Outside a Job
    #: (chat, an automation) they're simply absent.
    wants_job_provenance: bool = False

    #: True for a tool whose output is itself a user-facing answer (e.g.
    #: `analyze`'s synthesis). When such a step succeeds in a Job, its output
    #: leads the Job's final report even if later steps (save a note, set a
    #: reminder) came after it - see app/orchestration/report.py.
    produces_report: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier used to register and look up this tool."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human/AI-readable summary of what the tool does."""

    @property
    @abstractmethod
    def permission(self) -> PermissionLevel:
        """Trust level required before this tool is allowed to run."""

    @property
    def parameters(self) -> list[ToolParameter]:
        """Structured description of the keyword arguments `execute` takes.

        Defaults to none; tools that take input override this.
        """
        return []

    def validate_arguments(self, **kwargs: Any) -> None:
        """Side-effect-free check of argument VALUES, called for every
        step of an explicit tool sequence BEFORE any step executes (see
        app/tools/command.py). Raise ToolArgumentError to reject. Must
        never act on the world - at most read (e.g. a screen size).
        Default: accept; `execute` remains free to validate again.
        """

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool and return a structured ToolResult.

        Only called once the Registry has cleared permission checks.
        """
