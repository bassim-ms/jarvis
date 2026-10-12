"""
The Handle This v1 capability boundary: ONE authoritative statement of which
KINDS of objective Handle This may attempt.

    RESEARCH           finding/gathering current or external information
    ANALYSIS           reasoning over information already available
    ORGANIZATION       tasks, reminders, notes, calendar organization, and listing / reading / CREATING new text files in the
                       one Jarvis workspace folder (never changing, replacing or deleting an existing file)
    COMPUTER_WORKFLOW  approved browser / Windows computer-control workflows

For each family this module lists the EXISTING tools that family is built on.
That list is documentation the rest of the system can check, not a permission
system:

- It says "this class of objective is supported", never "this specific action
  is allowed". Whether a particular tool call may run - its permission level,
  the confirmation it needs, the autonomy of the Job - is decided, later and
  unchanged, by the Planner's validator, the capability checks and
  ToolRegistry. Nothing here can grant, widen or bypass any of that, and this
  module deliberately knows nothing about permission levels.
- It can only REMOVE support, never add it: `assess` compares the families an
  objective needs with the tools the eventual Planner would actually be OFFERED
  (passed in by the caller). A family with none of its tools on offer is
  reported unavailable, and an objective needing it is treated as unsupported
  for now. In particular the computer-control tools live in a separate registry
  that Jobs cannot reach (see tests/test_computer_m15_isolation.py); handing
  this boundary the Job-facing registry therefore reports COMPUTER_WORKFLOW as
  unavailable, and Handle This does not quietly route around that isolation.

Tool names are static, application-controlled data. Nothing a user or a model
says can add to them.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from app.handle_this.base import SUPPORTED_FAMILIES, GoalFamily

_DECLARED_TOOLS: dict[GoalFamily, frozenset[str]] = {
    GoalFamily.RESEARCH: frozenset({"web_search"}),
    GoalFamily.ANALYSIS: frozenset({"analyze"}),
    GoalFamily.ORGANIZATION: frozenset(
        {
            "create_task",
            "list_tasks",
            "complete_task",
            "cancel_task",
            "create_reminder",
            "list_reminders",
            "complete_reminder",
            "cancel_reminder",
            "save_note",
            "list_notes",
            "list_workspace_files",
            "read_workspace_file",
            "write_workspace_file",
            "list_events",
            "create_event",
            "update_event",
            "delete_event",
        }
    ),
    GoalFamily.COMPUTER_WORKFLOW: frozenset(
        {
            "get_active_window",
            "switch_window",
            "observe_page",
            "verify_observation",
            "observe_screen",
            "get_screen_info",
            "get_screen_size",
            "open_application",
            "open_url",
            "type_text",
            "press_key",
            "move_mouse",
            "click_mouse",
            "double_click",
            "scroll",
        }
    ),
}


#: For each kind of computer action (app/handle_this/signals.py `detect_computer_actions`), the tools a Job needs to do
#: it - ALL of them. The only way an action is "available" is that every one of its tools is actually offered to the
#: Planner, so wording alone (or a recognised COMPUTER_WORKFLOW) can never make a click, a typed word or an
#: application launch possible: `form` needs click_mouse AND type_text, and those are not offered. `tab` and `close`
#: name tools that do not exist yet, so they are never available.
COMPUTER_ACTION_TOOLS: dict[str, tuple[str, ...]] = {
    "navigate": ("open_url",),
    "scroll": ("scroll",),
    "switch_window": ("switch_window",),
    "click": ("click_mouse",),
    "type": ("type_text",),
    "key": ("press_key",),
    "mouse": ("move_mouse",),
    "launch_app": ("open_application",),
    "form": ("click_mouse", "type_text"),
    "tab": ("switch_tab",),
    "close": ("close_window",),
}


@dataclass(frozen=True)
class BoundaryAssessment:
    """Whether every family an objective needs is currently available.
    `unavailable` lists the families with none of their tools on offer."""

    unavailable: tuple[GoalFamily, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.unavailable


class CapabilityBoundary:
    """See the module docstring. Immutable: the families and their declared
    tools are fixed at construction from the static table above."""

    def __init__(self) -> None:
        self._tools = dict(_DECLARED_TOOLS)

    @property
    def families(self) -> tuple[GoalFamily, ...]:
        return SUPPORTED_FAMILIES

    def is_supported_family(self, family: object) -> bool:
        return family in SUPPORTED_FAMILIES

    def declared_tools(self, family: GoalFamily) -> frozenset[str]:
        """The existing tools `family` is built on (empty for MIXED,
        UNSUPPORTED, or anything else that is not a supported family)."""
        return self._tools.get(family, frozenset())

    def assess(
        self, required: Iterable[GoalFamily], available_tools: Iterable[str] | None
    ) -> BoundaryAssessment:
        """Which of `required` families cannot currently be attempted.
        `available_tools` is the set of tool names the eventual Planner would
        be offered; `None` means "no availability information": only the
        static family support is checked. A family that is not a supported
        family at all is always unavailable (fail closed)."""
        offered = None if available_tools is None else frozenset(available_tools)
        unavailable = []
        for family in required:
            if not self.is_supported_family(family):
                unavailable.append(family)
            elif offered is not None and not (self._tools[family] & offered):
                unavailable.append(family)
        return BoundaryAssessment(unavailable=tuple(unavailable))


    def assess_actions(self, actions: Iterable[str], available_tools: Iterable[str] | None) -> tuple[str, ...]:
        """The computer actions that cannot be done with the tools the Planner would be offered. An unknown kind is
        always missing (fail closed); `None` (no availability information) checks nothing."""
        if available_tools is None:
            return ()
        offered = frozenset(available_tools)
        return tuple(
            action
            for action in actions
            if not (action in COMPUTER_ACTION_TOOLS and set(COMPUTER_ACTION_TOOLS[action]) <= offered)
        )


DEFAULT_BOUNDARY = CapabilityBoundary()
