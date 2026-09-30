"""
MemoryWriteService: applies a WritePolicy decision through the EXISTING stores.

    proposal --WritePolicy.decide--> decision --> WorldModelService / MemoryStore
                                                  / candidate staging / a confirmation / nothing
                                       \\--> write_journal (always: what, why, from what)

This is the only code that turns a proposal into persisted state, and it never
writes a row itself: a fact goes through `MemoryStore.save_fact`, a preference /
project / goal through `WorldModelService` - the same calls everything else uses.
A model can reach it only through the tools in tools.py (registered on the
ToolRegistry, so the permission layer applies; non-plannable and non-automatable,
so neither a Job's plan nor an automation can name them); the user's own message
reaches it through `command_reply` / `observe`, which read the USER'S sentence
deterministically (extractor.py).

Sensitive information is the user's to confirm - through the EXISTING confirmation
machinery (confirmation.py): the request is held, a confirmation is asked, and only
the approved step (`commit_pending`, via the CONFIRM-level `commit_memory_write`
tool) persists it. "Remember anyway" is not a bypass. Secrets are never stored,
confirmed or not.

Contradictions: the new value replaces the current one (as the user asked) and
the journal keeps the previous value, its source and the time - history is
never destroyed. Candidates are staged, corroborated and only ever promoted by
the user stating the thing plainly; they are never read as facts (the Context
Engine may show a confident one, labelled unconfirmed - see
`candidates_for_context`).
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import time
import uuid

from app.memory.base import MemoryStore
from app.memory_policy.base import (
    Destination,
    Kind,
    Origin,
    Sensitivity,
    WriteDecision,
    WriteOutcome,
    WriteProposal,
)
from app.memory_policy.extractor import extract_statements, parse_command
from app.memory_policy.policy import WritePolicy, classify_sensitivity, normalize
from app.memory_policy.store import WriteStore
from app.world_model.base import EntitySource, UnconfirmedCandidate
from app.world_model.service import WorldModelService

logger = logging.getLogger("jarvis.memory_policy")

_NOTED = "Noted - I'll remember that."
_HAVE_IT = "I already have that."
_KEEP_IN_MIND = "I'll keep that in mind for now, but I haven't saved it as a fact."
_SECRET = "That looks like a credential or a secret, so I won't store it."
_SENSITIVE_ASK = "That looks sensitive, so I won't keep it unless you confirm. Shall I save it? (yes or no)"
_SENSITIVE_PASSIVE = (
    "That looks sensitive, so I haven't saved it. If you'd like me to keep it, ask me to remember it."
)
_SENSITIVE_NO_WAY = "That looks sensitive, so I haven't saved it."
_EXTERNAL = "I can't save something that came from an outside source."
_CANT = "I can't save that."

#: How long, and how many, held sensitive requests wait in memory for their confirmation.
PENDING_TTL_SECONDS = 86_400.0
MAX_PENDING = 20


@dataclasses.dataclass
class _Pending:
    proposal: WriteProposal
    created: float
    job_id: str | None = None


class MemoryWriteService:
    def __init__(
        self,
        world_model: WorldModelService,
        memory: MemoryStore,
        store: WriteStore,
        policy: WritePolicy | None = None,
        confirmer=None,
    ):
        self.world_model = world_model
        self.memory = memory
        self.store = store
        self.policy = policy or WritePolicy()
        # Duck-typed (see confirmation.py): `ask(pending_id, conversation_id) -> job_id`.
        # None: sensitive requests are simply not saved (there is nobody to ask through).
        self.confirmer = confirmer
        # Sensitive requests waiting for their confirmation. Held in memory ONLY - never
        # on disk - so an unconfirmed sensitive line leaves no trace, restart included
        # (after a restart the confirmation finds nothing to save and says so).
        self._pending: dict[str, _Pending] = {}
        self._pending_lock = threading.Lock()

    # -- the doors from the user's own words ------------------------------------------------------------------

    def command_reply(self, message: str, conversation_id: str | None = None) -> str | None:
        """If `message` is an explicit memory command ("remember that ...", "set
        my goal to ..."), apply it and return what to tell the user; else None."""
        proposals = parse_command(message)
        if not proposals:
            return None
        return self._reply([self.apply(p, conversation_id) for p in proposals], always=True)

    def observe(self, message: str, conversation_id: str | None = None) -> str | None:
        """Plain first-person statements in the user's message ("I prefer dark
        mode"): applied under the policy. Returns a short note when something was
        saved, asked about or refused; None when it was just conversation, a
        candidate, or temporary."""
        proposals = extract_statements(message)
        if not proposals:
            return None
        return self._reply([self.apply(p, conversation_id) for p in proposals], always=False)

    @staticmethod
    def redact(message: str) -> str:
        """The form of a user's message that may be kept in the conversation
        history: itself, unless it looks like it contains a credential or a
        secret - then a placeholder, so the history (which is fed back to the
        model as context) never holds one. The reply to the message is
        unaffected: it is produced from the original."""
        if isinstance(message, str) and classify_sensitivity(message) is Sensitivity.SECRET:
            return "[message withheld: it looked like it contained a secret]"
        return message

    def candidates_for_context(self) -> list[UnconfirmedCandidate]:
        """The staged candidates, as the read-only, labelled-unconfirmed shape the
        Context Engine may show reasoning (`ContextEngine.candidate_source`). Nothing
        is promoted or written by reading them."""
        return [
            UnconfirmedCandidate(
                id=c.id, kind=c.kind, key=c.key, value=c.value, confidence=c.confidence, origin=c.origin,
                created_at=c.created_at, updated_at=c.updated_at,
            )
            for c in self.store.list_candidates()
        ]

    @staticmethod
    def _reply(outcomes: list[WriteOutcome], *, always: bool) -> str | None:
        lines = []
        for outcome in outcomes:
            quiet = outcome.decision.destination in (Destination.CANDIDATE, Destination.TEMPORARY)
            if (always or not quiet) and outcome.message and outcome.message not in lines:
                lines.append(outcome.message)
        return " ".join(lines[:2]) or None

    # -- the one door to persistence -------------------------------------------------------------------------------------

    def apply(self, proposal: WriteProposal, conversation_id: str | None = None) -> WriteOutcome:
        """Decide, then act. Never raises for a policy outcome; a store fault
        propagates (nothing half-applied: each write is one call). Whatever the
        proposal says about being confirmed is ignored here: only an APPROVED
        confirmation (`commit_pending`) can lift the sensitive hold."""
        return self._apply(proposal, conversation_id, confirmed=False)

    def _apply(self, proposal: WriteProposal, conversation_id: str | None, *, confirmed: bool) -> WriteOutcome:
        proposal = dataclasses.replace(proposal, acknowledged_sensitive=confirmed)
        goal_project = self._resolve_project(proposal) if proposal.kind is Kind.GOAL else None
        if (
            proposal.kind is Kind.GOAL
            and goal_project is None
            and self.policy.decide(proposal).destination in (Destination.WORLD_MODEL, Destination.MEMORY)
        ):
            # (a goal the policy would refuse or ask about anyway is answered by the
            # policy below - the project question is only for one that would persist)
            decision = WriteDecision(Destination.ASK, "no single project to add the goal to")
            self._journal("asked", proposal, decision, value=None)
            return WriteOutcome(
                decision, False, "Which project should I add that goal to? Tell me the project's name."
            )

        existing = self._existing(proposal, goal_project)
        decision = self.policy.decide(proposal, existing)
        d = decision.destination

        if d is Destination.REJECT:
            self._journal("rejected", proposal, decision, value=None)
            return WriteOutcome(decision, False, self._reject_message(proposal, decision))
        if d is Destination.ASK:
            self._journal("asked", proposal, decision, value=None)
            return WriteOutcome(decision, False, self._ask(proposal, decision, conversation_id))
        if d is Destination.TEMPORARY:
            self._journal("temporary", proposal, decision, value=None)
            return WriteOutcome(decision, False, "")
        if d is Destination.CANDIDATE:
            return self._stage(proposal, decision)
        if confirmed:
            decision = dataclasses.replace(decision, reason=f"{decision.reason}; confirmed by the user")
        return self._persist(proposal, decision, goal_project)

    # -- the confirmation of a sensitive request ------------------------------------------------------------------------

    def _ask(self, proposal: WriteProposal, decision: WriteDecision, conversation_id: str | None) -> str:
        if decision.sensitivity is Sensitivity.SENSITIVE:
            if proposal.origin is not Origin.USER_COMMAND:
                return _SENSITIVE_PASSIVE  # a passing statement: not worth a question - just not saved
            return _SENSITIVE_ASK if self._hold_and_ask(proposal, conversation_id) else _SENSITIVE_NO_WAY
        if decision.existing_value is not None:
            return (
                f"Earlier you told me \"{decision.existing_value}\". If that has changed, tell me the new "
                "value and say it changed (for example \"my budget is now ...\"), or say \"remember that ...\" "
                "to replace it."
            )
        return _CANT

    def _hold_and_ask(self, proposal: WriteProposal, conversation_id: str | None) -> bool:
        """Hold the request (in memory) and ask the user through the existing
        confirmation. False - and nothing held - if it cannot be asked."""
        if self.confirmer is None:
            return False
        pending_id = uuid.uuid4().hex
        with self._pending_lock:
            self._prune_locked()
            self._pending[pending_id] = _Pending(proposal, time.monotonic())
        try:
            job_id = self.confirmer.ask(pending_id, conversation_id)
        except Exception:  # noqa: BLE001 - it cannot be asked: nothing is saved, nothing stays held
            logger.warning("memory_policy confirmation_failed")
            with self._pending_lock:
                self._pending.pop(pending_id, None)
            return False
        with self._pending_lock:
            held = self._pending.get(pending_id)
            if held is not None:
                held.job_id = job_id
        return True

    def _prune_locked(self) -> None:
        now = time.monotonic()
        for key in [k for k, p in self._pending.items() if now - p.created > PENDING_TTL_SECONDS]:
            del self._pending[key]
        while len(self._pending) >= MAX_PENDING:
            del self._pending[min(self._pending, key=lambda k: self._pending[k].created)]

    def commit_pending(self, pending_id: str, job_id: str | None) -> WriteOutcome:
        """Persist a held sensitive request - called ONLY by the CONFIRM-level tool, i.e.
        only when the user approved the confirmation of THAT Job (`job_id` is supplied
        by the engine, never by a plan). The policy still runs (a secret is still
        refused); what the confirmation lifts is only the sensitive hold."""
        with self._pending_lock:
            self._prune_locked()
            held = self._pending.get(pending_id)
            if held is None:
                decision = WriteDecision(Destination.ASK, "nothing is waiting for that confirmation")
                return WriteOutcome(decision, False, "That request has expired, so nothing was saved.")
            approved = getattr(self.confirmer, "approved", None)
            if not job_id or held.job_id != job_id or approved is None or not approved(job_id):
                # not the Job that asked, or its confirmation was not APPROVED (declined,
                # expired, withdrawn, or not answered yet): nothing is saved
                decision = WriteDecision(Destination.REJECT, "not an approved confirmation")
                return WriteOutcome(decision, False, "That can only be saved by the confirmation that was asked.")
            del self._pending[pending_id]
        return self._apply(held.proposal, None, confirmed=True)

    def pending_count(self) -> int:
        with self._pending_lock:
            return len(self._pending)

    # -- persistence, through the existing services ------------------------------------------------------------------

    def _persist(self, proposal: WriteProposal, decision: WriteDecision, goal_project) -> WriteOutcome:
        if decision.existing_value is not None and not decision.updates_existing:
            self._journal("unchanged", proposal, decision, value=proposal.value)
            return WriteOutcome(decision, False, _HAVE_IT)

        source = decision.source or EntitySource.USER
        ref: str | None = None
        if proposal.kind is Kind.FACT:
            self.memory.save_fact(proposal.key, proposal.value)
        elif proposal.kind is Kind.PREFERENCE:
            current = self._current_preference(proposal.key)
            if current is None:
                ref = self.world_model.create_preference(
                    proposal.key, proposal.value, proposal.category, source, decision.confidence
                ).id
            else:
                ref = self.world_model.update_preference(
                    current.id, value=proposal.value, category=proposal.category, source=source,
                    confidence=decision.confidence,
                ).id
        elif proposal.kind is Kind.PROJECT:
            ref = self.world_model.create_project(
                proposal.value, metadata={"source": source.value, "origin": proposal.origin.value}
            ).id
        else:  # GOAL
            ref = self.world_model.create_goal(goal_project.id, proposal.value).id

        action = "updated" if decision.updates_existing else "persisted"
        self._journal(
            action, proposal, decision, value=proposal.value, previous=decision.existing_value, ref=ref
        )
        candidate = self.store.get_candidate(proposal.kind.value, proposal.key)
        if candidate is not None:  # the user has now said it plainly: the candidate is validated
            self.store.promote_candidate(candidate.id)
            self._journal("promoted", proposal, decision, value=proposal.value, ref=candidate.id)
        message = (
            f"Updated - I had \"{decision.existing_value}\", now \"{proposal.value}\"."
            if decision.updates_existing
            else _NOTED
        )
        return WriteOutcome(decision, True, message, ref)

    def _stage(self, proposal: WriteProposal, decision: WriteDecision) -> WriteOutcome:
        candidate, corroborated = self.store.stage_candidate(
            kind=proposal.kind.value,
            key=proposal.key,
            value=proposal.value,
            category=proposal.category,
            origin=proposal.origin.value,
            confidence=decision.confidence,
            evidence=proposal.evidence,
        )
        self._journal(
            "corroborated" if corroborated else "candidate", proposal, decision, value=proposal.value,
            ref=candidate.id,
        )
        return WriteOutcome(decision, False, _KEEP_IN_MIND, candidate.id)

    # -- lookups (reads through the existing services) -----------------------------------------------------------------------

    def _existing(self, proposal: WriteProposal, goal_project) -> str | None:
        if proposal.kind is Kind.FACT:
            return self.memory.get_facts().get(proposal.key)
        if proposal.kind is Kind.PREFERENCE:
            current = self._current_preference(proposal.key)
            return current.value if current is not None else None
        if proposal.kind is Kind.PROJECT:
            for project in self.world_model.get_active_projects():
                if normalize(project.name) == normalize(proposal.value):
                    return project.name
            return None
        if goal_project is None:
            return None
        for goal in self.world_model.get_active_goals(goal_project.id):
            if normalize(goal.title) == normalize(proposal.value):
                return goal.title
        return None

    def _current_preference(self, key: str):
        found = self.world_model.get_preferences(key=key)
        return found[0] if found else None  # newest first

    def _resolve_project(self, proposal: WriteProposal):
        active = self.world_model.get_active_projects()
        if proposal.project:
            named = [p for p in active if normalize(p.name) == normalize(proposal.project)]
            return named[0] if len(named) == 1 else None
        return active[0] if len(active) == 1 else None

    # -- what the user is told / what is journalled --------------------------------------------------------------------------

    @staticmethod
    def _reject_message(proposal: WriteProposal, decision: WriteDecision) -> str:
        if decision.sensitivity is Sensitivity.SECRET:
            return _SECRET
        if proposal.origin is Origin.EXTERNAL:
            return _EXTERNAL
        return _CANT

    def _journal(
        self,
        action: str,
        proposal: WriteProposal,
        decision: WriteDecision,
        *,
        value: str | None,
        previous: str | None = None,
        ref: str | None = None,
    ) -> None:
        """Values of anything not persisted or staged are never recorded (`value`
        is None for those) - see store.py."""
        try:
            self.store.record(
                action=action,
                kind=proposal.kind.value,
                key=(
                    "(withheld)"
                    if decision.sensitivity is Sensitivity.SECRET
                    or (
                        decision.sensitivity is Sensitivity.SENSITIVE
                        and not decision.persists
                        and decision.destination is not Destination.CANDIDATE
                    )
                    else proposal.key
                ),  # a key is made from the user's words: nothing sensitive that was not kept may leave a trace
                value=value,
                previous_value=previous,
                origin=proposal.origin.value,
                destination=decision.destination.value,
                source=decision.source.value if decision.source else None,
                confidence=decision.confidence if decision.persists or decision.destination is Destination.CANDIDATE else None,
                reason=decision.reason,
                ref=ref,
            )
        except Exception:  # noqa: BLE001 - an audit fault must not undo or block the user's write
            logger.warning("memory_policy journal_failed")
