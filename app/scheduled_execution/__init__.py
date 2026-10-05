"""
Scheduled Jobs V1 - Slice 2 (the execution adapter) and Slice 3 (the scheduler that drives it).

    Approved Scheduled Job -> Scheduled Occurrence -> Execution Adapter -> fresh context -> READ-only unattended profile
        -> an ordinary, PENDING Job -> (the existing) JobEngine -> Planner -> ToolRegistry -> Tools -> Verify

This package turns ONE due occurrence of ONE approved Scheduled Job into ONE ordinary Job and persists it atomically with its ledger
row (the Slice 1 invariant). It does not wake up, discover due schedules, poll, plan, run a tool, notify, or write memory: it is
invoked explicitly (by a test, or by the scheduler a later slice adds), and everything after the Job exists is the existing machinery.
It is a separate package from app/scheduled_jobs on purpose: that one stays a pure foundation that can import nothing able to run,
plan or notify; this one is the single place a Scheduled Job meets the orchestration layer.

Slice 3 adds `scheduler.py`: `ScheduledJobScheduler.tick()` discovers due occurrences, applies the runtime gates (kill switch, expiry,
overlap, failure breaker, execution cap, device trust, the capability boundary), launches through the adapter, and settles finished runs.
It is hung on the JobEngine's own tick - no second loop, thread or timer - and it still never executes a tool, plans, or notifies.
"""
