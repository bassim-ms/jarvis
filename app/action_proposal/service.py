"""
The Action Proposal service: build a small plan from what Focus decided, show it, and wait for the user's answer - without creating a
Job, a task, a reminder or anything else. Nothing is executed here, and nothing is stored: a proposal lives in memory for a short time,
bound to ONE conversation, and is gone on a restart.

    "make a plan" -> Focus decides -> a proposal (the user's steps, and at most a step of Jarvis's) -> shown -> WAITING
    WAITING + a reply:   yes ........ pop it (exactly once) -> revalidate -> hand off (Slice 3) -> only THEN may a Job exist
                         no ......... dropped
                         "ok"/"sure"  a nudge for a clear yes; stays waiting (the lifetime is not extended)
                         a question about the plan, or "skip step N" ... answered here; stays waiting (never extended)
                         anything else .... the proposal is dropped, and the message carries on as ordinary

THE ADJACENCY LOCK. A proposal is only ever answered by the very next message. `note_message` is called for EVERY message the chat core
receives, before anything else decides who answers it, and counts the messages that have arrived since the proposal was last shown or
answered. The proposal is consulted only while that count is exactly one - the message in front of it. Any other message - a reminder, a
search, a tool call, a question another door took, even one that FAILED and so left nothing in the conversation history - moves the count,
so a later "yes" can never approve an old proposal. (A count of stored messages would miss a failed exchange.)

WHAT AN APPROVAL COVERS. Exactly the steps that were shown: the plan handed on is rebuilt from those same stored steps, the digest is
recomputed from live data (a task that changed or was finished makes it stale), and every tool call is validated again against the live
registry. Any doubt - stale, expired, wrong conversation, already consumed - creates nothing. This module has no model, no Planner, no
JobStore and no way to create a Job: the handoff is a callback (`on_approved`, see handoff.py) - absent, the approval says nothing was started.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta

from app.action_proposal import render as words
from app.action_proposal.base import (
    MAX_PENDING,
    TTL_SECONDS,
    ActionProposal,
    ApprovedProposal,
    ProposalStep,
    proposal_digest,
    task_stamp,
)
from app.action_proposal.builder import ProposalBuilder, ProposalError, is_note_step, plan_steps_for, reminder_zone
from app.action_proposal.recognition import (
    ACK,
    APPROVE,
    CHANGE,
    EXPLAIN_AFTER,
    EXPLAIN_STEP,
    EXPLAIN_WHY,
    LAST,
    REJECT,
    SHOW,
    SKIP,
    FollowUp,
    classify_follow_up,
    parse_plan_request,
)
from app.focus.base import PICK
from app.focus.recognition import QUESTION, FocusRequest, parse_focus_request
from app.focus.render import quoted, render_decision
from app.tasks.base import TaskStatus

_CLOCK_SLACK = timedelta(minutes=1)


@dataclass
class _Pending:
    proposal: ActionProposal
    expires_at: float  # on the injected monotonic clock; a follow-up never moves it
    messages_since: int = 0  # messages that arrived since this was last shown or answered; the lock requires exactly 1 when it is consulted


class ActionProposalService:
    """`focus` needs only `decide(FocusRequest)`; `tasks` is read-only (`get`); `tools` is the live registry (validated against, never
    executed). `on_approved(ApprovedProposal) -> str` is the handoff that starts the approved steps (handoff.py) - `None` (the default) means nothing can be started, and an approval says so."""

    def __init__(
        self,
        *,
        focus,
        tasks,
        tools,
        clock: Callable[[], datetime],
        monotonic: Callable[[], float] = time.monotonic,
        on_approved: Callable[[ApprovedProposal], str] | None = None,
        ttl_seconds: float = TTL_SECONDS,
    ):
        self.focus = focus
        self.tasks = tasks
        self.tools = tools
        self._clock = clock
        self._monotonic = monotonic
        self.on_approved = on_approved
        self._ttl = ttl_seconds
        self._builder = ProposalBuilder(tasks=tasks, tools=tools, clock=clock)
        self._pending: dict[str, _Pending] = {}
        self._lock = threading.Lock()

    # -- the lock -----------------------------------------------------------------------------------------------------------------------

    def note_message(self, conversation_id: str | None) -> None:
        """Called by the chat core for every message, first of all: one more message has arrived in this conversation. Never raises."""
        if not isinstance(conversation_id, str):
            return
        with self._lock:
            entry = self._pending.get(conversation_id)
            if entry is not None:
                entry.messages_since += 1

    # -- the two doors ----------------------------------------------------------------------------------------------------------------

    def command_reply(self, message: str, conversation_id: str | None = None) -> str | None:
        """The ordinary-conversation door: a reply to a waiting proposal, else a request for a plan, else None. Never raises."""
        if not isinstance(conversation_id, str) or not conversation_id:
            return None
        try:
            reply = self._respond_pending(message, conversation_id)
            if reply is not None:
                return reply
            request = parse_plan_request(message)
            return None if request is None else self._create(request.subject, conversation_id, note=request.save_note)
        except Exception:  # noqa: BLE001 - a fault here is "not a proposal message": nothing was created
            return None

    def confirmation_reply(self, message: str, conversation_id: str | None = None) -> str | None:
        """Offered a yes/no-shaped message only AFTER every real confirmation has declined it: a reply to a waiting proposal, else None."""
        if not isinstance(conversation_id, str) or not conversation_id:
            return None
        try:
            return self._respond_pending(message, conversation_id)
        except Exception:  # noqa: BLE001
            return None

    def pending_count(self) -> int:
        with self._lock:
            self._purge_locked()
            return len(self._pending)

    # -- creating -----------------------------------------------------------------------------------------------------------------------

    def _create(self, subject: str, conversation_id: str, *, note: bool = False) -> str | None:
        if subject:
            # The same reading of "my <X> project" Focus itself uses: a subject with a clause in it is not ours to plan.
            request = parse_focus_request(f"I need to finish my {subject}")
            if request is None:
                return None
        else:
            request = FocusRequest(QUESTION)
        decision = self.focus.decide(request)
        if decision is None:
            return None  # "make a plan for my coffee": nothing is known about it - ordinary conversation
        if decision.outcome != PICK:
            return render_decision(decision)  # ambiguous / unknown / nothing to rank: Focus's own question - no proposal is built
        proposal = self._builder.build(decision, conversation_id, note=note)
        if proposal is None:
            return render_decision(decision)
        # A note the user asked for but that could not be offered is SAID, never silently left out.
        notice = words.NOTE_UNAVAILABLE + "\n" if note and not any(is_note_step(step) for step in proposal.steps) else ""
        if not proposal.my_steps:  # nothing of Jarvis's to approve: shown, never pending
            return notice + words.render_proposal(proposal, approvable=False)
        self._hold(conversation_id, _Pending(proposal, self._monotonic() + self._ttl))
        return notice + words.render_proposal(proposal)

    # -- answering a waiting proposal --------------------------------------------------------------------------------------------------------

    def _respond_pending(self, message: str, conversation_id: str) -> str | None:
        with self._lock:
            entry = self._pending.get(conversation_id)
        if entry is None:
            return None
        if self._monotonic() >= entry.expires_at:
            self._drop(conversation_id, entry)
            return None
        if entry.messages_since != 1:  # not the very next message: something else arrived since the proposal was shown
            self._drop(conversation_id, entry)
            return None

        follow = classify_follow_up(message)
        kind = follow.kind
        if kind == APPROVE:
            return self._approve(conversation_id, entry)
        if kind == REJECT:
            self._drop(conversation_id, entry)
            return words.DROPPED
        proposal = entry.proposal
        if kind == ACK:
            return self._answered(entry, words.nudge(proposal))
        if kind == SHOW:
            return self._answered(entry, words.render_proposal(proposal))
        if kind == CHANGE:
            return self._answered(entry, words.cannot_change(proposal))
        if kind in (EXPLAIN_STEP, EXPLAIN_AFTER, EXPLAIN_WHY, SKIP):
            return self._about_steps(conversation_id, entry, follow)
        self._drop(conversation_id, entry)  # unrelated: the proposal goes, the message carries on
        return None

    def _answered(self, entry: _Pending, reply: str) -> str:
        """The proposal stays waiting - for the very next message again. The lifetime is NOT extended."""
        with self._lock:
            entry.messages_since = 0
        return reply

    def _about_steps(self, conversation_id: str, entry: _Pending, follow: FollowUp) -> str:
        proposal = entry.proposal
        count = len(proposal.steps)
        number = None
        if follow.step is not None:
            number = count if follow.step == LAST else follow.step
            if not 1 <= number <= count:
                return self._answered(entry, words.no_such_step(proposal))
        if follow.kind == EXPLAIN_STEP:
            return self._answered(entry, words.explain_step(proposal, number))
        if follow.kind == EXPLAIN_AFTER:
            return self._answered(entry, words.explain_after(proposal, number))
        if follow.kind == EXPLAIN_WHY:
            return self._answered(
                entry, words.explain_why_plan(proposal) if number is None else words.explain_why_step(proposal, number)
            )
        return self._skip(conversation_id, entry, number)

    def _skip(self, conversation_id: str, entry: _Pending, number: int) -> str:
        """Skipping DISCARDS the proposal and builds a new one from the remaining steps (new id, new digest) - no step is edited in place."""
        old = entry.proposal
        remaining: tuple[ProposalStep, ...] = tuple(step for i, step in enumerate(old.steps, start=1) if i != number)
        remaining = self._builder.refresh_note(old.subject, old.made_on, remaining)  # a saved plan must not list a step that was skipped
        if not remaining:
            self._drop(conversation_id, entry)
            return words.NOTHING_TO_DO
        rebuilt = replace(old, id=uuid.uuid4().hex, steps=remaining, digest=proposal_digest(old.subject, remaining))
        if not rebuilt.my_steps:  # nothing of Jarvis's left: shown for the user, but there is nothing to approve
            self._drop(conversation_id, entry)
            return words.NOTHING_TO_DO + "\n" + "\n".join(words.numbered(rebuilt))
        replacement = _Pending(rebuilt, entry.expires_at)  # the original deadline, not a new one
        with self._lock:
            if self._pending.get(conversation_id) is not entry:
                return words.DROPPED
            self._pending[conversation_id] = replacement
        return f"Okay - without step {number}:\n" + words.render_proposal(rebuilt)

    # -- approving ------------------------------------------------------------------------------------------------------------------------

    def _approve(self, conversation_id: str, entry: _Pending) -> str | None:
        with self._lock:  # POP exactly once: a second yes (or a racing one) finds nothing
            if self._pending.get(conversation_id) is not entry:
                return None
            del self._pending[conversation_id]
        proposal = entry.proposal
        problems, plan_steps = self._revalidate(proposal)
        if problems:
            return words.changed(problems)
        approved = ApprovedProposal(
            proposal_id=proposal.id,
            conversation_id=proposal.conversation_id,
            subject=proposal.subject,
            steps=proposal.steps,
            plan_steps=tuple(plan_steps),
            digest=proposal.digest,
            approved_at=self._now(),
        )
        if self.on_approved is None:
            return words.UNWIRED
        try:
            result = self.on_approved(approved)
        except Exception:  # noqa: BLE001 - fail closed: the proposal is already consumed
            return words.FAILED
        return result if isinstance(result, str) and result.strip() else words.UNWIRED

    def _revalidate(self, proposal: ActionProposal) -> tuple[list[str], list]:
        """(problems, validated plan steps). A non-empty problem list means NOTHING may be handed on."""
        if proposal_digest(proposal.subject, proposal.steps) != proposal.digest:
            return ["the plan itself no longer matches what I showed you"], []
        now = self._now()
        zone = reminder_zone(self.tools)
        local_now = now.astimezone(zone) if zone is not None else now  # the zone the reminder tool itself reads `at` in
        problems: list[str] = []
        if local_now.date().isoformat() != proposal.made_on:
            problems.append("it's a new day, so the plan's timing no longer holds")

        seen: dict[str, object] = {}
        for step in proposal.steps:
            for ref in step.refs:
                seen.setdefault(ref.ref, ref)
        live_stamps: dict[str, str] = {}
        reasons: list[str] = []  # what to tell the user IF the plan turns out stale
        for ref in seen.values():
            try:
                task = self.tasks.get(ref.ref[len("task:") :])
            except Exception:  # noqa: BLE001
                reasons.append(f"I couldn't re-check {quoted(ref.label)}")
                continue
            if task is None:
                reasons.append(f"{quoted(ref.label)} is no longer open")
                continue
            live_stamps[ref.ref] = task_stamp(task)
            if task.status is not TaskStatus.TODO:
                reasons.append(f"{quoted(ref.label)} is no longer open")
            elif live_stamps[ref.ref] != ref.stamp:
                reasons.append(f"{quoted(ref.label)} has changed")
        # THE test of staleness: the digest recomputed from LIVE data. It equals the stored one only if every referent is exactly as it
        # was when the plan was shown; the reasons above only say WHAT changed.
        live_steps = tuple(
            replace(step, refs=tuple(replace(ref, stamp=live_stamps.get(ref.ref, "")) for ref in step.refs)) for step in proposal.steps
        )
        if proposal_digest(proposal.subject, live_steps) != proposal.digest:
            problems.extend(reasons or ["something I based the plan on has changed"])

        actions = [step.action for step in proposal.steps if step.action is not None]
        plan_steps: list = []
        if actions:
            try:
                plan_steps = plan_steps_for(self.tools, actions)
            except ProposalError as exc:
                problems.append(f"I can't do my part any more ({exc})")
            if zone is None and any(action.tool_name == "create_reminder" for action in actions):  # only a reminder needs the reminder zone
                problems.append("I can't tell which time zone my reminders use")
            for action in actions:
                if self._reminder_has_passed(action, local_now):
                    problems.append("the reminder time has already passed")
        return problems, plan_steps

    @staticmethod
    def _reminder_has_passed(action, now: datetime) -> bool:
        args = action.as_arguments()
        if action.tool_name != "create_reminder" or args.get("day") != "today":
            return False
        try:
            hour, minute = (int(part) for part in str(args.get("at", "")).split(":"))
        except ValueError:
            return True  # a time that cannot be read is not one to act on
        return now.replace(hour=hour, minute=minute, second=0, microsecond=0) < now + _CLOCK_SLACK

    # -- the pending map ------------------------------------------------------------------------------------------------------------------------

    def _hold(self, conversation_id: str, entry: _Pending) -> None:
        with self._lock:
            self._purge_locked()
            while len(self._pending) >= MAX_PENDING and conversation_id not in self._pending:
                del self._pending[min(self._pending, key=lambda k: self._pending[k].expires_at)]
            self._pending[conversation_id] = entry  # one per conversation: a new proposal replaces the old

    def _drop(self, conversation_id: str, entry: _Pending) -> None:
        with self._lock:
            if self._pending.get(conversation_id) is entry:
                del self._pending[conversation_id]

    def _purge_locked(self) -> None:
        now = self._monotonic()
        for key in [k for k, e in self._pending.items() if now >= e.expires_at]:
            del self._pending[key]

    def _now(self) -> datetime:
        now = self._clock()
        return now if now.tzinfo is not None else now.astimezone()
