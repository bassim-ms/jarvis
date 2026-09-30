"""
The `morning_brief` tool: the read-only brief, reachable through the ordinary ToolRegistry like any other
tool ("use tool morning_brief"). READ permission (no side effects), and NOT plannable / NOT automatable - a
goal-driven plan is never offered it and no automation may run it, because scheduled or autonomous delivery of
the brief is a separate, later decision. The tool has no parameters: nothing a caller passes can change what
is read.
"""

from __future__ import annotations

from app.tools.base import PermissionLevel, Tool, ToolResult


class MorningBriefTool(Tool):
    name = "morning_brief"
    description = "Give the user a short, prioritised brief of today: schedule, tasks, reminders and what needs attention. Read-only."
    permission = PermissionLevel.READ
    plannable = False
    automatable = False

    def __init__(self, service):
        self.service = service

    def execute(self) -> ToolResult:
        try:
            return ToolResult.success(data={"brief": self.service.brief()})
        except Exception:  # noqa: BLE001
            return ToolResult.error("The morning brief could not be put together.")
