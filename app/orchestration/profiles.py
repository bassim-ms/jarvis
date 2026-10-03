"""
Run profiles: a REDUCTION of what one Job may do. Never a grant.

An ordinary Job (`INTERACTIVE_PROFILE`) has no profile and is exactly as it always was. A Job that names a profile - a Job created for a
Scheduled Job's occurrence, which runs while nobody is at the keyboard - gets the profile applied on top of everything that already
applies to every Job. It can only take capability away:

  * the Planner is OFFERED only the profile's tools (and only those the live registry still has, still plannable, still READ);
  * `validate_plan` REJECTS any other tool, whatever the planner returned;
  * JobEngine REFUSES, just before running, any step whose tool is not permitted - a third, independent check against the live
    registry - so a stale or hand-built plan cannot run it either;
  * a step that would need confirmation is never parked (nobody is there to answer): if the tool were not permitted it simply fails;
  * the plan is bounded tighter than the engine's own limits (fewer steps, fewer replans) - the smaller of the two always wins.

What it does NOT do: it grants no permission, adds no tool, lowers no check, and bypasses neither the ToolRegistry nor the permission
levels, argument validation, plan validation or audit trail. The ToolRegistry stays the only authority over whether a call runs; a profile
is checked BEFORE it, and can only say no. There is no second permission system: "permitted" here means "on the list AND the registry
says it is a plannable, read-only tool right now".

`unattended_v1` is the READ-only surface approved for Scheduled Jobs V1: research, web observation of PUBLIC pages, extraction and
analysis, and reading the user's own organizer state. Writes (tasks, notes, files, calendar, reminders, memory), messages, computer
control, screen capture and every CONFIRM/DANGEROUS tool are outside it - and adding a tool to the registry never adds it here.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.orchestration.base import INTERACTIVE_PROFILE
from app.tools.base import PermissionLevel


class UnknownProfileError(ValueError):
    """A Job names a run profile this system does not define. Treated as a refusal to run (fail closed), never as 'no profile'."""


@dataclass(frozen=True)
class RunProfile:
    name: str
    allowed_tools: frozenset[str]
    max_steps: int
    max_total_steps: int
    max_replans: int

    def denial(self, tool_name: str, tool) -> str | None:
        """Why `tool` (the live registered tool named `tool_name`, or None) may not be used under this profile; None if it may."""
        if tool is None:
            return f"'{tool_name}' is not available."
        if tool_name not in self.allowed_tools:
            return f"'{tool_name}' is not permitted in an unattended run."
        if not tool.plannable:
            return f"'{tool_name}' cannot be used in a plan."
        if tool.permission is not PermissionLevel.READ:
            return f"'{tool_name}' is not read-only, so it is not permitted in an unattended run."
        return None


#: The approved V1 unattended surface. Names only; each must still be registered, plannable and READ to be offered or run.
UNATTENDED_V1 = RunProfile(
    name="unattended_v1",
    allowed_tools=frozenset(
        {
            "web_search",  # research
            "observe_page",  # web observation of a public page (an anonymous HTTP fetch, not the user's browser or screen)
            "verify_observation",  # a deterministic check of an observation
            "analyze",  # analysis and synthesis
            "list_tasks",  # reading the user's own organizer state
            "list_notes",
            "list_events",
            "list_reminders",
        }
    ),
    max_steps=6,
    max_total_steps=9,
    max_replans=1,
)

PROFILES: dict[str, RunProfile] = {UNATTENDED_V1.name: UNATTENDED_V1}


def profile_for(name: str | None) -> RunProfile | None:
    """The profile a Job runs under: None for an ordinary Job. An unknown name raises - it is never quietly treated as ordinary."""
    if name == INTERACTIVE_PROFILE:
        return None
    profile = PROFILES.get(name) if isinstance(name, str) else None
    if profile is None:
        raise UnknownProfileError(f"Unknown run profile: {name!r}.")
    return profile


def permitted_tools(profile: RunProfile, registry) -> list:
    """The live tools `profile` allows right now: on its list, registered, plannable and still read-only."""
    return [tool for tool in registry.list_tools() if profile.denial(tool.name, tool) is None]
