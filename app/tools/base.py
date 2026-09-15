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
    type: str  # e.g. "string", "number", "boolean", "object"
    description: str = ""
    required: bool = True
    default: Any = None


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


class Tool(ABC):
    """A single capability Jarvis can invoke.

    Implementations declare their own name, description, parameters, and
    permission level, and provide `execute`. Nothing outside a Tool's own
    module needs to know its concrete type - see app/tools/registry.py.
    """

    #: Tools with no external dependencies can leave this at the default.
    requirements: ToolRequirements = ToolRequirements()

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

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool and return a structured ToolResult.

        Only called once the Registry has cleared permission checks.
        """
