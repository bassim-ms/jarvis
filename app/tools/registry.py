"""
Tool Registry.

A generic lookup/dispatch table for Tool instances. It knows nothing about
any specific tool - registration is the only way tools become known to it -
so it scales from a couple of fake test tools to hundreds of real ones
without ever needing an `if tool_name == ...` branch or a code change here.

Permission enforcement (see app/tools/base.py PermissionLevel) happens here,
inside Jarvis, so it applies no matter what eventually decides to call a
tool - an AI provider's tool-calling loop, a script, or a human.
"""

from typing import Any

from app.tools.base import AUTOMATIC_PERMISSIONS, Tool, ToolResult


class ToolAlreadyRegisteredError(ValueError):
    """Raised when registering a tool whose name is already taken."""


class ToolRegistry:
    """Holds registered tools and mediates permission-checked execution."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Add a tool, keyed by its own declared name.

        Raises ToolAlreadyRegisteredError on a duplicate name rather than
        silently overwriting, since that usually indicates a bug (two
        tools accidentally sharing a name) rather than intent.
        """
        if tool.name in self._tools:
            raise ToolAlreadyRegisteredError(
                f"Tool '{tool.name}' is already registered."
            )
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Remove a tool by name, if present. No-op if it isn't registered."""
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        """Look up a registered tool by name, or None if it's not known."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Return whether a tool with this name is registered."""
        return name in self._tools

    def list_tools(self) -> list[Tool]:
        """Return all registered tools."""
        return list(self._tools.values())

    def execute(
        self,
        name: str,
        params: dict[str, Any] | None = None,
        confirmed: bool | None = None,
    ) -> ToolResult:
        """Run a registered tool by name, enforcing its permission level.

        `confirmed` is tri-state and only matters for tools whose
        permission is CONFIRM or DANGEROUS (READ and SAFE_ACTION always
        run immediately):
          - None  -> confirmation hasn't been asked for yet; returns
                     CONFIRMATION_REQUIRED (carrying the tool's permission
                     level, so a caller can show a stronger warning for
                     DANGEROUS than for CONFIRM).
          - False -> the user was asked and declined; returns DENIED.
          - True  -> the user approved; the tool runs.

        Unknown tool names and exceptions raised by a tool's `execute` are
        both reported as ToolResult errors rather than propagating, so a
        single misbehaving tool can't crash the caller.
        """
        tool = self.get(name)
        if tool is None:
            return ToolResult.error(f"Unknown tool: '{name}'")

        if tool.permission not in AUTOMATIC_PERMISSIONS:
            if confirmed is None:
                return ToolResult.confirmation_required(
                    tool.permission,
                    message=f"Tool '{name}' requires confirmation before running.",
                )
            if confirmed is False:
                return ToolResult.denied(f"Execution of '{name}' was denied.")

        try:
            return tool.execute(**(params or {}))
        except Exception as exc:  # noqa: BLE001 - isolate tool failures
            return ToolResult.error(f"Tool '{name}' raised an error: {exc}")
