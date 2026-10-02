"""
Scheduled Jobs V1 - Slice 1: the foundation (model, state machine, storage, ledger, pure policy). NOTHING RUNS from this package yet:
it creates no Job, calls no Planner or tool, sends no notification, and is not wired into the application. Later slices add the
unattended Job profile, the scheduler, creation through Handle This, and management. See base.py for the concepts and the lifecycle.
"""
