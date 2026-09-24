"""
Self-Development Engine V1: a controlled subsystem that lets Jarvis
propose, and - only with explicit human approval at each gate - apply,
bounded changes to its own codebase.

This is NOT a second M15 (app/orchestration/): there is no competing
Plan/PlanStep/Job execution system here. Every actual action (reading a
file, editing a file, running a command, running tests, creating a
branch, committing, rolling back) is a plain app.tools.base.Tool
registered on this subsystem's OWN, separate ToolRegistry
(app/self_development/engine.py) and gated by that registry's existing
PermissionLevel/confirmation mechanism (app/tools/registry.py) - never
a second permission system. See app/self_development/base.py for the
DevelopmentCapability -> PermissionLevel mapping.

Module layout:
    base.py           - types/validation only (DevelopmentPlan, session
                         state, results) - no I/O.
    safety.py          - path-boundary and command-allowlist enforcement
                         - the security choke points everything else
                         depends on.
    command_runner.py  - the ONE place a subprocess is ever started
                         (shell=False, always), plus RunTestsTool/
                         RunCommandTool.
    git.py             - GitClient (status/diff/branch/commit/rollback),
                         built entirely on command_runner - never a
                         second subprocess call site.
    inspector.py       - read-only repo tools (list/read/search).
    editor.py          - EditFileTool (the one write path).
    planner.py          - LLM-backed DevelopmentPlan/fix proposal,
                         reusing the existing AIProvider abstraction.
    engine.py           - SelfDevelopmentEngine: the session/approval/
                         repair-loop orchestrator - the only thing that
                         calls this subsystem's ToolRegistry.
    factory.py           - builds the application's default engine.
"""
