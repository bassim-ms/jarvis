"""
Jarvis Core - the orchestrator for handling a chat request.

Core depends only on the AIProvider interface, not on any specific AI
vendor, so providers can be added or swapped without touching this module.
"""

from app.providers.base import AIProvider
from app.providers.gemini import GeminiProvider


class JarvisCore:
    """Handles chat requests by delegating to a configured AI provider."""

    def __init__(self, provider: AIProvider | None = None):
        self.provider = provider or GeminiProvider()

    def handle_chat(self, message: str) -> str:
        """Route a user message to the AI provider and return its reply."""
        return self.provider.generate_reply(message)


core = JarvisCore()
