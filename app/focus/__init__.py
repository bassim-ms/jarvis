"""
Focus (Productivity Action Loop, Slice 1): "what am I trying to get done, and what should I do first?"

A read-only answer built from what the existing systems already hold - Tasks, the calendar, the World Model's
projects and goals, the user's stated activity - ranked by a small deterministic formula and worded from fixed
templates. It starts nothing, writes nothing, creates no Job and calls no model.
"""

from app.focus.service import FocusService

__all__ = ["FocusService"]
