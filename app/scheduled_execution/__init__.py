"""
Scheduled Jobs V1 - Slice 2: the execution adapter.

    Approved Scheduled Job -> Scheduled Occurrence -> Execution Adapter -> fresh context -> READ-only unattended profile
        -> an ordinary, PENDING Job -> (the existing) JobEngine -> Planner -> ToolRegistry -> Tools -> Verify

This package turns ONE due occurrence of ONE approved Scheduled Job into ONE ordinary Job and persists it atomically with its ledger
row (the Slice 1 invariant). It does not wake up, discover due schedules, poll, plan, run a tool, notify, or write memory: it is
invoked explicitly (by a test, or by the scheduler a later slice adds), and everything after the Job exists is the existing machinery.
It is a separate package from app/scheduled_jobs on purpose: that one stays a pure foundation that can import nothing able to run,
plan or notify; this one is the single place a Scheduled Job meets the orchestration layer.
"""
