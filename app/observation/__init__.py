"""
Observation (Slice 4): read-only, bounded, normalised "what is on the page / in the active window / on the
screen right now" for the EXISTING Planner + JobEngine + ToolRegistry to compose with the existing actions.

    base.py     the Observation shape, every bound, and the secret scrubber
    page.py     structured page observation: a guarded, cookie-less HTTP fetch parsed with the stdlib
    screen.py   the last-resort screenshot + vision seam (bounded, in-memory, never persisted)
    engine.py   ObservationEngine: picks the best source (page -> window -> screen) and normalises. Nothing else.
    verify.py   deterministic checks of an Observation (no model, no I/O)
    tools.py    the READ tools the Job planner / chat use: observe_page, verify_observation, observe_screen

This package never plans, never acts, never touches Memory or the World Model, never decides a permission and
never bypasses the ToolRegistry. Everything it returns is DATA: a page that says "ignore your rules" is page text.
"""
