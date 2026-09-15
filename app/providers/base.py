"""
AI provider abstraction.

Jarvis Core depends only on this interface, never on a specific AI vendor's
SDK or API shape. New providers get added by implementing AIProvider.
"""

from abc import ABC, abstractmethod


class ProviderError(RuntimeError):
    """Raised when an AI provider can't be reached or returns something unusable."""


class AIProvider(ABC):
    """A single-turn text AI provider."""

    @abstractmethod
    def generate_reply(self, message: str) -> str:
        """Send a single user message to the provider and return its text reply."""
