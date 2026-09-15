"""
Jarvis Core - the orchestrator for handling a chat request.

Core depends only on the AIProvider and MemoryStore interfaces, not on any
specific AI vendor or storage backend, so providers can be added or swapped
without touching this module.
"""

from app.config import settings
from app.memory.base import MemoryStore, Message
from app.memory.sqlite_store import SQLiteMemoryStore
from app.providers.base import AIProvider
from app.providers.gemini import GeminiProvider
from app.tools.registry import ToolRegistry


class ConversationNotFoundError(RuntimeError):
    """Raised when a client supplies a conversation_id that doesn't exist."""


class JarvisCore:
    """Handles chat requests: builds AI context from conversation history and
    stored facts, delegates to an AI provider, then persists the exchange."""

    def __init__(
        self,
        provider: AIProvider | None = None,
        memory: MemoryStore | None = None,
        tools: ToolRegistry | None = None,
    ):
        self.provider = provider or GeminiProvider()
        self.memory = memory or SQLiteMemoryStore(settings.memory_db_path)
        # Empty by default - no real tools exist yet (M5 is architecture
        # only). AI tool-calling isn't wired up here either; Core just has
        # a place to reach the registry without knowing any tool's type.
        self.tools = tools or ToolRegistry()

    def handle_chat(
        self, message: str, conversation_id: str | None = None
    ) -> tuple[str, str]:
        """Route a user message to the AI provider and return its reply.

        If conversation_id is None, a new conversation is started. If it's
        given but unknown, ConversationNotFoundError is raised rather than
        silently starting a new conversation under that ID. Otherwise the
        conversation's prior messages are loaded and combined with any
        stored user facts to give the provider context for this turn.
        """
        if conversation_id is None:
            conversation_id = self.memory.create_conversation()
        elif not self.memory.conversation_exists(conversation_id):
            raise ConversationNotFoundError(
                f"Conversation '{conversation_id}' does not exist."
            )

        history = self.memory.get_messages(conversation_id)
        facts = self.memory.get_facts()
        prompt = self._build_prompt(message, history, facts)

        reply = self.provider.generate_reply(prompt)

        self.memory.add_message(conversation_id, "user", message)
        self.memory.add_message(conversation_id, "assistant", reply)

        return reply, conversation_id

    def save_memory(self, key: str, value: str) -> None:
        """Programmatic hook to save a user fact. Manual for now; automatic
        memory extraction can call this later without changing Core."""
        self.memory.save_fact(key, value)

    def get_memories(self) -> dict[str, str]:
        """Programmatic hook to retrieve all stored user facts."""
        return self.memory.get_facts()

    @staticmethod
    def _build_prompt(
        message: str, history: list[Message], facts: dict[str, str]
    ) -> str:
        """Combine stored facts and conversation history into context for
        the provider. With no facts or history, the raw message is sent
        unchanged, matching the original single-turn behavior."""
        parts = []
        if facts:
            facts_text = "\n".join(f"- {key}: {value}" for key, value in facts.items())
            parts.append(f"Known facts about the user:\n{facts_text}")
        if history:
            history_text = "\n".join(f"{m.role}: {m.content}" for m in history)
            parts.append(f"Conversation so far:\n{history_text}")

        if not parts:
            return message
        return "\n\n".join(parts) + f"\n\nUser: {message}"


core = JarvisCore()
