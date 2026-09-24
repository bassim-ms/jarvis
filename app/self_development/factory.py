"""
Builds the application's default SelfDevelopmentEngine - the one place
construction happens, mirroring every other app/*/factory.py in this
codebase. Takes the AIProvider to route the planner through rather than
constructing one itself (see app/context_reasoning/factory.py/
app/context_planning/factory.py for the identical convention).
"""

import shlex

from app.self_development.command_runner import RunCommandTool, RunTestsTool
from app.self_development.editor import EditFileTool, SecurityEditFileTool
from app.self_development.engine import SelfDevelopmentEngine
from app.self_development.git import (
    CommitTool,
    CreateBranchTool,
    GitClient,
    GitDiffTool,
    RollbackTool,
)
from app.self_development.inspector import ListRepoFilesTool, ReadFileTool, SearchRepoTool
from app.self_development.planner import DevelopmentPlanner
from app.providers.base import AIProvider
from app.tools.registry import ToolRegistry


def create_default_self_development_registry(
    project_root: str, git_client: GitClient, *, command_timeout_seconds: float, test_command: list[str]
) -> ToolRegistry:
    """A fresh, dedicated ToolRegistry for self-development tools only
    - deliberately SEPARATE from JarvisCore's own `core.tools`
    (app/core.py), so these privileged development tools are never
    reachable through the ordinary TOOL capability/ToolCallBuilder
    path (app/tools/decision.py), and JarvisCore's own tools are never
    reachable from here."""
    registry = ToolRegistry()
    registry.register(ListRepoFilesTool(project_root))
    registry.register(ReadFileTool(project_root))
    registry.register(SearchRepoTool(project_root))
    registry.register(EditFileTool(project_root))
    registry.register(SecurityEditFileTool(project_root))
    registry.register(RunTestsTool(project_root, test_command, command_timeout_seconds))
    registry.register(RunCommandTool(project_root, command_timeout_seconds))
    registry.register(GitDiffTool(git_client))
    registry.register(CreateBranchTool(git_client))
    registry.register(CommitTool(git_client))
    registry.register(RollbackTool(git_client))
    return registry


def create_default_self_development_engine(
    provider: AIProvider,
    *,
    project_root: str,
    max_repair_iterations: int,
    command_timeout_seconds: float,
    test_command: str,
) -> SelfDevelopmentEngine:
    git_client = GitClient(project_root, timeout=command_timeout_seconds)
    registry = create_default_self_development_registry(
        project_root,
        git_client,
        command_timeout_seconds=command_timeout_seconds,
        test_command=shlex.split(test_command),
    )
    planner = DevelopmentPlanner(provider)
    return SelfDevelopmentEngine(
        project_root,
        registry,
        git_client,
        planner,
        max_repair_iterations=max_repair_iterations,
    )
