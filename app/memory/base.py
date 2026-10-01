"""
Memory abstraction.

Jarvis Core depends only on this interface, never on SQLite (or any other
storage engine) directly, so the persistence backend can be replaced later
without touching Core.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Message:
    """A single turn in a conversation's history."""

    role: str
    content: str


class MemoryStore(ABC):
    """Persistent storage for conversation history and simple user facts."""

    @abstractmethod
    def create_conversation(self) -> str:
        """Create a new conversation and return its ID."""

    @abstractmethod
    def conversation_exists(self, conversation_id: str) -> bool:
        """Return whether a conversation with this ID has been created."""

    @abstractmethod
    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """Append a message to a conversation's history."""

    @abstractmethod
    def get_messages(self, conversation_id: str) -> list[Message]:
        """Return a conversation's messages in chronological order."""

    @abstractmethod
    def save_fact(self, key: str, value: str) -> None:
        """Store (or update) a simple fact about the user."""

    @abstractmethod
    def get_facts(self) -> dict[str, str]:
        """Return all stored facts about the user."""

    @abstractmethod
    def delete_fact(self, key: str) -> bool:
        """Forget one fact. True if it existed. Only ever called for an explicit, unambiguous request
        from the user (see app/memory_policy/service.py)."""
