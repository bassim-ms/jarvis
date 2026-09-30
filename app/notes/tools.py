"""
Note tools: adapt NoteStore into the Tool interface, so saving and listing
notes go through the same ToolRegistry/permission machinery as everything
else - including as steps of a goal-driven Job (M15), the reason they exist.

`save_note` is SAFE_ACTION: it only ever appends one row to the notes
table (its own store, separate from memory/facts and the World Model - see
app/notes/base.py) and can't affect anything else. `list_notes` is READ.

When `save_note` runs as a Job step, JobEngine adds `source_job_id`/
`source_step_id` (`wants_job_provenance`); they are deliberately NOT among
the tool's declared parameters, so a plan can never claim a different
source. Everything a note contains is stored verbatim and must be treated as
untrusted data by whatever reads it back.
"""

from __future__ import annotations

from typing import Any

from app.notes.base import MAX_CONTENT_CHARS, MAX_TITLE_CHARS, Note, NoteStore
from app.tools.base import PermissionLevel, Tool, ToolArgumentError, ToolParameter, ToolResult

DEFAULT_LIST_LIMIT = 20
MAX_LIST_LIMIT = 50


def _note_to_dict(note: Note) -> dict[str, Any]:
    return {
        "id": note.id,
        "title": note.title,
        "content": note.content,
        "created_at": note.created_at,
        "source_job_id": note.source_job_id,
        "source_step_id": note.source_step_id,
    }


class SaveNoteTool(Tool):
    """Saves one note. SAFE_ACTION - see this module's docstring."""

    name = "save_note"
    description = (
        "Save a note (a short title and its text content) for the user to "
        "read later. Notes are kept separately from memory and are never "
        "used as facts automatically."
    )
    permission = PermissionLevel.SAFE_ACTION
    wants_job_provenance = True

    def __init__(self, store: NoteStore):
        self.store = store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="title",
                type="string",
                description=f"A short title (at most {MAX_TITLE_CHARS} characters)",
            ),
            ToolParameter(
                name="content",
                type="string",
                description=f"The note's text (at most {MAX_CONTENT_CHARS} characters)",
                free_text=True,
            ),
        ]

    def validate_arguments(
        self,
        title: str,
        content: str,
        source_job_id: str | None = None,
        source_step_id: str | None = None,
    ) -> None:
        if not isinstance(title, str) or not title.strip():
            raise ToolArgumentError("A note needs a non-empty title.")
        if len(title) > MAX_TITLE_CHARS:
            raise ToolArgumentError(f"A note title can be at most {MAX_TITLE_CHARS} characters.")
        if not isinstance(content, str) or not content.strip():
            raise ToolArgumentError("A note needs non-empty content.")
        if len(content) > MAX_CONTENT_CHARS:
            raise ToolArgumentError(f"A note's content can be at most {MAX_CONTENT_CHARS} characters.")

    def execute(
        self,
        title: str,
        content: str,
        source_job_id: str | None = None,
        source_step_id: str | None = None,
    ) -> ToolResult:
        try:
            self.validate_arguments(title, content)
        except ToolArgumentError as exc:
            return ToolResult.error(str(exc))

        note = self.store.create(
            title=title.strip(),
            content=content,
            source_job_id=source_job_id,
            source_step_id=source_step_id,
        )
        return ToolResult.success(
            data=_note_to_dict(note), message=f"Saved note '{note.id}': {note.title}"
        )


class ListNotesTool(Tool):
    """Lists saved notes, newest first. READ: no side effects."""

    name = "list_notes"
    description = (
        "List saved notes (newest first), optionally only those saved by one "
        "task. Result data is a list of {id, title, content, created_at, "
        "source_job_id, source_step_id}."
    )
    permission = PermissionLevel.READ

    def __init__(self, store: NoteStore):
        self.store = store

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="job_id",
                type="string",
                description="Only notes saved by this task (job) id",
                required=False,
                default=None,
            ),
            ToolParameter(
                name="limit",
                type="number",
                description=f"How many notes to return (1-{MAX_LIST_LIMIT}, default {DEFAULT_LIST_LIMIT})",
                required=False,
                default=DEFAULT_LIST_LIMIT,
            ),
        ]

    def validate_arguments(self, job_id: str | None = None, limit: float = DEFAULT_LIST_LIMIT) -> None:
        if isinstance(limit, bool) or not isinstance(limit, (int, float)) or limit != int(limit):
            raise ToolArgumentError("limit must be a whole number.")
        if not 1 <= int(limit) <= MAX_LIST_LIMIT:
            raise ToolArgumentError(f"limit must be between 1 and {MAX_LIST_LIMIT}.")

    def execute(self, job_id: str | None = None, limit: float = DEFAULT_LIST_LIMIT) -> ToolResult:
        try:
            self.validate_arguments(job_id, limit)
        except ToolArgumentError as exc:
            return ToolResult.error(str(exc))

        notes = self.store.list(job_id=job_id, limit=int(limit))
        return ToolResult.success(
            data=[_note_to_dict(n) for n in notes],
            message=f"{len(notes)} note(s).",
        )
