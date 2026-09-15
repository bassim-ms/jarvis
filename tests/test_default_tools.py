"""
Tests for default tool registration: proves `web_search` is part of the
default/application tool registry with READ permission, without changing
JarvisCore's own no-args default, which stays empty - see
test_tools.py::test_core_exposes_a_tool_registry_by_default.
"""

from app.tools.base import PermissionLevel
from app.tools.defaults import create_default_registry, register_default_tools
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool


def test_create_default_registry_contains_web_search():
    registry = create_default_registry()

    assert registry.has("web_search")


def test_default_web_search_tool_keeps_read_permission():
    registry = create_default_registry()

    tool = registry.get("web_search")

    assert tool.permission == PermissionLevel.READ


def test_register_default_tools_adds_web_search_to_a_given_registry():
    registry = ToolRegistry()

    register_default_tools(registry)

    assert registry.has("web_search")
    assert isinstance(registry.get("web_search"), WebSearchTool)


# --- The actual running application ---------------------------------------


def test_application_core_has_web_search_registered():
    """Proves the app's real JarvisCore singleton (app.main.core) - what
    /chat and any future tool-calling would actually use - has web_search
    registered, not just that the factory function works in isolation."""
    from app.main import core as app_core

    assert app_core.tools.has("web_search")
    assert app_core.tools.get("web_search").permission == PermissionLevel.READ
