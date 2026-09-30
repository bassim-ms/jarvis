"""
Notes abstraction: small, durable pieces of text Jarvis saves on the user's
behalf - typically the useful findings of a goal-driven Job ("what I found
out about X") - with enough metadata to say where each one came from.

Deliberately its OWN store, separate from everything else that persists
things about the user:

  * NOT memory facts (app/memory): facts are injected into every prompt, so
    anything saved there changes every future conversation. A note is only
    ever read when someone asks for notes.
  * NOT the World Model (app/world_model): that is structured, curated
    knowledge (projects, goals, commitments...). A note is free text and
    unvetted - it may contain whatever a web page said.
  * NOT Rhasspy Tasks (app/tasks): those are todo items.

Nothing here (or anywhere) promotes a note into memory or the World Model
automatically; that would have to be a separate, deliberate, user-approved
feature. Note content should be treated as untrusted data by anything that
reads it back: it is stored verbatim.

Depends only on the types defined here, never on SQLite directly - the same
"Core/Registry never knows the storage engine" split as every other store.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

#: Bounds enforced by the tools (app/notes/tools.py) before anything is
#: stored, so a note can't be used to park unbounded data.
MAX_TITLE_CHARS = 200
MAX_CONTENT_CHARS = 5000


@dataclass
class Note:
    """One saved note. `source_job_id`/`source_step_id` are set only when a
    note was saved by a step of a goal-driven Job (JobEngine supplies them -
    a plan can't); they are `None` for a note saved any other way."""

    id: str
    title: str
    content: str
    created_at: str
    source_job_id: str | None = None
    source_step_id: str | None = None


class NoteStore(ABC):
    """Persistent storage for Notes."""

    @abstractmethod
    def create(
        self,
        title: str,
        content: str,
        source_job_id: str | None = None,
        source_step_id: str | None = None,
    ) -> Note:
        """Create and persist a new Note. Length limits are the caller's
        (the tool's) responsibility - the store stores what it is given."""

    @abstractmethod
    def get(self, note_id: str) -> Note | None:
        """Look up a Note by ID, or None."""

    @abstractmethod
    def list(self, job_id: str | None = None, limit: int = 20) -> list[Note]:
        """Return Notes, most recently created first - at most `limit`,
        optionally only those saved by the Job `job_id`."""
