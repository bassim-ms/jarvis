"""
Default tool registration for the running application.

This is the one place that knows the concrete set of tools Jarvis ships
with by default (currently just `web_search`). It exists so the actual
application (app/main.py) can get a fully-populated ToolRegistry, while
JarvisCore itself (app/core.py) stays unaware of any specific tool - its
own default, when no registry is injected, stays an empty ToolRegistry
(see test_tools.py::test_core_exposes_a_tool_registry_by_default).
"""

from app.search.cached import CachedSearchProvider
from app.search.factory import (
    create_default_search_cache,
    create_default_search_provider,
)
from app.tools.registry import ToolRegistry
from app.tools.web_search import WebSearchTool


def register_default_tools(registry: ToolRegistry) -> None:
    """Register every tool Jarvis ships with by default onto `registry`.

    Takes a registry rather than owning one, so callers - the application
    singleton, tests, or a future dashboard - decide what registry gets
    the defaults, instead of this module hardcoding where they end up.

    web_search is wired as WebSearchTool -> CachedSearchProvider ->
    SearchProviderManager -> (SearXNGProvider, ...), built from
    app/search/factory.py, so JARVIS_SEARCH_PROVIDER_ORDER and
    JARVIS_SEARCH_CACHE_TTL both take effect without WebSearchTool,
    SearchProviderManager, or JarvisCore knowing anything about SQLite or
    caching.
    """
    provider = CachedSearchProvider(
        create_default_search_provider(), create_default_search_cache()
    )
    registry.register(WebSearchTool(provider=provider))


def create_default_registry() -> ToolRegistry:
    """Build a fresh ToolRegistry populated with the default tools."""
    registry = ToolRegistry()
    register_default_tools(registry)
    return registry
