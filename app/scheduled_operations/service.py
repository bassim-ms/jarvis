"""
ScheduledJobService: the ONE place a user's request becomes a PROPOSED Scheduled Job, an approval makes it ACTIVE, and the user's own
controls (pause / resume / cancel / look) act on it. The chat door (operations.py) and the approval tool (tools.py) both call it, so
every path gets identical rules.

It contains no rule of its own that the Slice 1-3 layers already have. Timing is built by the SHARED schedule builder (the one
Reminder V2 uses); every limit - the minimum gap, the horizon, the 90-day lifetime, the cap on how many exist - is the existing policy
(app/scheduled_jobs/policy.py); every state change is the store's compare-and-set (`create_proposal` / `activate` / `transition` /
`resume`). It adds no scheduler, no Job, no tool, no store and no second approval system.

WHAT IT STORES. The user's own words for what to do and the stable, validated understanding of it (Handle This's objective, handed in
as a callable so this package never imports Handle This). Never a plan, never a tool, and never a snapshot of the world: whatever the
understanding picked up from the user's current context (`source: context` constraints) is dropped, because context is built FRESH for
every run.

APPROVAL GRANTS AN INTENT, NOT AUTHORITY. Before anything is stored - and again at approval, when the world may have changed - the
Slice 2 adapter's own check is run (`check_objective`): the objective must be something an unattended run under the reduce-only READ
profile could do right now, or the request is REFUSED aloud. It is never trimmed to something smaller than was asked. Approving it
adds no capability: every run still goes through the profile, the ToolRegistry, the Planner's validation, device trust and audit.

Errors are `ScheduledJobRequestError` with a plain, user-facing message. Nothing here runs a tool, makes a Job, calls a Planner or a
model, notifies, or writes memory.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.reminders.language import When
from app.scheduled_jobs.base import (
    OPEN_STATUSES,
    PROFILE_UNATTENDED_V1,
    InvalidTransitionError,
    ScheduledJob,
    ScheduledJobError,
    ScheduledJobRun,
    ScheduledJobStatus,
    ScheduledJobStore,
    StaleScheduledJobError,
    StatusReason,
)
from app.scheduled_jobs.policy import (
    Limits,
    Lifetime,
    ScheduledJobPolicyError,
    check_proposal_cap,
    clean_intent,
    lifetime_for,
    next_future_due,
    validate_new_schedule,
)
from app.scheduling.builder import ScheduleSpec, ScheduleSpecError, Wording, build_schedule
from app.scheduling.schedule import Schedule, ScheduleError, ScheduleKind, next_due
from app.scheduling.timezones import UTC, TimezoneUnavailableError, ZoneChoice, resolve_timezone

logger = logging.getLogger("jarvis.scheduled_operations")

_S, _R = ScheduledJobStatus, StatusReason

WORDING = Wording(subject="scheduled job", when_question="When should I do it? For example: every morning at 8 AM, or tomorrow at 5 PM.")
MAX_TITLE_WORDS_CHARS = 60
#: What the user is told when the request needs more than an unattended run may do. Deliberately free of internal terms and of any list of
#: what exists; what V1 CAN do is stated in the approval prompt.
UNATTENDED_REFUSAL = (
    "I can only schedule read-only work for now - researching, reading public web pages and looking at your tasks, notes, events and "
    "reminders. I can't schedule something that changes things, sends anything or controls your computer, so I haven't set that up."
)


class ScheduledJobRequestError(ValueError):
    """A request that cannot be honoured, with a plain message safe to say to the user."""


@dataclass(frozen=True)
class Proposed:
    """A stored PROPOSED Scheduled Job and what approving it would mean. `duplicate`: an identical one already exists (open), so none was
    added - `scheduled_job` is that one."""

    scheduled_job: ScheduledJob
    first_due: datetime
    lifetime: Lifetime
    rolled: bool = False  # a bare "at 7pm" that had passed today was set for tomorrow
    duplicate: bool = False
    warning: str = ""  # JARVIS_TIMEZONE was set but unusable and the system zone was used instead
    renews: str | None = None  # the id of the Scheduled Job this proposal would RENEW (and replace) if approved


@dataclass(frozen=True)
class Approved:
    scheduled_job: ScheduledJob
    lifetime: Lifetime
    already: bool = False  # it was approved by this very confirmation before: nothing was done again
    renewed: bool = False  # it replaced the Scheduled Job it renews


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _short_title(text: str) -> str:
    words, title = text.split(), ""
    for word in words:
        candidate = f"{title} {word}".strip()
        if len(candidate) > MAX_TITLE_WORDS_CHARS:
            return title.rstrip(" ,;:-") + "…"
        title = candidate
    return title


def stable_objective(objective: dict[str, Any]) -> dict[str, Any]:
    """The objective as it is STORED: what was understood from the request itself. A constraint that came from the user's current
    context is a fact about the world as it was just now - it is dropped, because every run builds its context afresh."""
    kept = dict(objective)
    kept["constraints"] = [c for c in objective.get("constraints", []) if isinstance(c, dict) and c.get("source") != "context"]
    return kept


def understander(handle_this: Callable[[], Any]) -> Callable[[str, str | None], dict[str, Any]]:
    """Adapts the EXISTING Handle This (`handle_this()` returns it, or None) to the service's `understand`. The application wiring
    supplies it - this package never imports Handle This - and the outcome is read by its public shape alone (`kind.value`, `reply`,
    `fallback_reply`, `job_params`). Only a READY, supported request yields an objective; "something material is missing" is
    answered with Handle This's own one-line ask (send it again with that included): no clarification Job is created, no state kept."""

    def understand(text: str, conversation_id: str | None) -> dict[str, Any]:
        service = handle_this()
        outcome = service.handle(text, None) if service is not None else None  # no conversation: nothing waits on a question
        if outcome is None:
            raise ScheduledJobRequestError("I can't work out what to schedule right now, so nothing was set up.")
        kind = getattr(getattr(outcome, "kind", None), "value", None)
        if kind == "start_job" and isinstance(getattr(outcome, "job_params", None), dict) and isinstance(outcome.job_params.get("objective"), dict):
            return outcome.job_params["objective"]  # the understood request ONLY: its context snapshot is never kept
        if kind == "unsupported":
            raise ScheduledJobRequestError(UNATTENDED_REFUSAL)
        raise ScheduledJobRequestError(getattr(outcome, "fallback_reply", None) or getattr(outcome, "reply", None) or "I couldn't work out what to schedule.")

    return understand


class ScheduledJobService:
    """`understand(text, conversation_id) -> objective` turns the instruction into Handle This's objective (or raises
    ScheduledJobRequestError carrying what to tell the user); `check_objective(objective, profile)` is the Slice 2 adapter's check
    (raises a ScheduledJobError when an unattended run could not be made). Both are injected by the application wiring."""

    def __init__(
        self,
        store: ScheduledJobStore,
        *,
        limits: Limits,
        understand: Callable[[str, str | None], dict[str, Any]],
        check_objective: Callable[[dict[str, Any], str], object],
        clock: Callable[[], datetime] | None = None,
        zone_resolver: Callable[[], ZoneChoice] | None = None,
        proposal_ttl: timedelta = timedelta(hours=1),
        run_now: Callable[[str], Any] | None = None,
        in_flight: Callable[[str], bool] | None = None,
    ):
        self.store = store
        self._run_now = run_now
        self._in_flight = in_flight
        self.limits = limits
        self._understand = understand
        self._check_objective = check_objective
        self._clock = clock or _utc_now
        self._zone_resolver = zone_resolver
        self.proposal_ttl = proposal_ttl

    # -- the environment --------------------------------------------------------------------------------------------------------

    def now(self) -> datetime:
        return self._clock().astimezone(UTC)

    def zone(self) -> ZoneChoice:
        """The zone a NEW Scheduled Job is created in: JARVIS_TIMEZONE, else the system's when it can be determined reliably, else a clear
        failure (nothing is created that might run at the wrong time). Stored in the Schedule and never changed afterwards."""
        try:
            if self._zone_resolver is not None:
                return self._zone_resolver()
            from app.config import settings
            from app.system_timezone import detect_system_timezone

            return resolve_timezone(settings.timezone, detect_system_timezone())
        except TimezoneUnavailableError as exc:
            raise ScheduledJobRequestError(str(exc)) from None

    def sweep(self) -> None:
        """Proposals nobody approved in time end (EXPIRED), so they never hold one of the user's places or look approvable."""
        self.store.expire_proposals(self.now())

    # -- proposing ---------------------------------------------------------------------------------------------------------------

    def propose(self, text: str, when: When, *, conversation_id: str | None = None, device_id: str | None = None) -> Proposed:
        """Validate everything and store a PROPOSED Scheduled Job. Nothing runs from it: it is not due, and the scheduler only ever
        looks at ACTIVE ones."""
        self.sweep()
        now = self.now()
        try:
            intent = clean_intent(text)
        except ScheduledJobPolicyError as exc:
            raise ScheduledJobRequestError(str(exc)) from None
        if when.relative_minutes is not None or when.relative_days is not None:
            raise ScheduledJobRequestError("I can schedule something for a time of day or a date - for example \"tomorrow at 5 PM\" - not \"in a while\".")

        spec = ScheduleSpec(when.clock, when.day, when.repeat, when.weekdays, when.every, None)
        try:
            choice = self.zone()
        except ScheduledJobRequestError:
            if when.every is None:
                raise
            choice = ZoneChoice("UTC", "system")  # an elapsed interval ("every 2 hours") never depended on a zone
        try:
            schedule, first_due, rolled = build_schedule(spec, choice.name, now, min_interval=self.limits.min_interval, wording=WORDING)
            validate_new_schedule(schedule, now=now, first_due=first_due, limits=self.limits)
        except ScheduleSpecError as exc:
            raise ScheduledJobRequestError(str(exc)) from None
        except (ScheduledJobPolicyError, TimezoneUnavailableError) as exc:
            raise ScheduledJobRequestError(str(exc)) from None
        except ScheduleError as exc:
            raise ScheduledJobRequestError(f"I couldn't set that up: {exc}.") from None
        lifetime = lifetime_for(schedule, now, None, self.limits)

        return self._store_proposal(
            intent=intent, title=_short_title(intent), schedule=schedule, first_due=first_due, lifetime=lifetime, rolled=rolled,
            warning=choice.warning, now=now, profile=PROFILE_UNATTENDED_V1,
            make_objective=lambda: stable_objective(self._understand(intent, conversation_id)),
            conversation_id=conversation_id, device_id=device_id,
        )

    def _store_proposal(
        self, *, intent: str, title: str, schedule: Schedule, first_due: datetime, lifetime: Lifetime, rolled: bool, warning: str,
        now: datetime, profile: str, make_objective: Callable[[], dict[str, Any]], conversation_id: str | None, device_id: str | None,
        renews: str | None = None, open_discount: int = 0,
    ) -> Proposed:
        """The part a NEW request and a RENEWAL share: no duplicate, a place free, the objective something an unattended run could do,
        stored as a PROPOSAL. `renews` names the Scheduled Job a renewal replaces (it is not a duplicate of its own renewal, and while it
        is ACTIVE it does not use up a second place: `open_discount`)."""
        key = schedule.to_dict()
        for existing in self.store.list(OPEN_STATUSES):
            if existing.id != renews and existing.intent_text.casefold() == intent.casefold() and existing.schedule.to_dict() == key:
                return Proposed(existing, existing.due_at or first_due, lifetime, rolled, duplicate=True, warning=warning, renews=renews)

        try:  # (after the duplicate check: repeating something you already have is not "too many")
            check_proposal_cap(max(0, self.store.count(OPEN_STATUSES) - open_discount), self.limits)
        except ScheduledJobPolicyError as exc:
            raise ScheduledJobRequestError(str(exc)) from None
        objective = make_objective()
        self._require_unattended(objective, profile)
        try:
            proposal = self.store.create_proposal(
                title=title,
                intent_text=intent,
                objective=objective,
                schedule=schedule,
                proposal_expires_at=now + self.proposal_ttl,
                profile=profile,
                created_via_conversation_id=conversation_id,
                created_via_device_id=device_id,
                now=now,
            )
        except ScheduledJobError as exc:
            raise ScheduledJobRequestError(str(exc)) from None
        # The check above and the insert are not one atomic step, so two identical requests arriving together could both get here. Both then
        # look again and the EARLIEST (created time, then id) wins: the loser withdraws its own row, so exactly one proposal is ever asked.
        earlier = [
            sj for sj in self.store.list(OPEN_STATUSES)
            if sj.id not in (proposal.id, renews) and sj.intent_text.casefold() == intent.casefold() and sj.schedule.to_dict() == key
            and (sj.created_at, sj.id) < (proposal.created_at, proposal.id)
        ]
        if earlier:
            self.discard(proposal.id)
            winner = min(earlier, key=lambda sj: (sj.created_at, sj.id))
            return Proposed(winner, winner.due_at or first_due, lifetime, rolled, duplicate=True, warning=warning, renews=renews)
        return Proposed(proposal, first_due, lifetime, rolled, warning=warning, renews=renews)

    # -- renewing ----------------------------------------------------------------------------------------------------------------------

    def propose_renewal(self, scheduled_job_id: str, *, conversation_id: str | None = None, device_id: str | None = None) -> Proposed:
        """Propose RENEWING a repeating Scheduled Job: a NEW Scheduled Job - same instruction, same understanding, same schedule - that the
        user must approve exactly as they approved the first (the same confirmation, the same commit tool). Nothing is renewed by asking, and
        nothing renews itself. It is a fresh scheduling decision, so it meets every rule a new one does (minimum gap, the 90-day bound from
        ITS approval, the unattended check against what is available now): approving it extends nothing - it starts one new bounded period.

        Only a Scheduled Job that is running (ACTIVE) or whose approved period has just ended can be renewed. A paused one is not - the user
        decides to resume it first - and neither is one the user cancelled, nor a one-time one (it has no period to renew)."""
        self.sweep()
        now = self.now()
        old = self._require(scheduled_job_id)
        self._require_renewable(old)
        self._require_not_running(old)
        if old.consecutive_failures >= self.limits.breaker_failures:
            # Its failure streak is at the limit where the breaker stops a Scheduled Job (and the streak would come with a renewal), so
            # renewing it would only be paused again: say why, and leave a NEW request - a fresh decision - as the way forward.
            raise ScheduledJobRequestError(
                f"'{old.title}' had failed {old.consecutive_failures} runs in a row, which is where I stop running a scheduled job, so I won't renew it as it is. "
                "Look at why it was failing; if you still want it, ask me to schedule it again as a new one."
            )
        first_due = next_due(old.schedule, now)
        try:
            validate_new_schedule(old.schedule, now=now, first_due=first_due, limits=self.limits)
        except ScheduledJobPolicyError as exc:
            raise ScheduledJobRequestError(f"{exc} I can't renew it as it is.") from None
        lifetime = lifetime_for(old.schedule, now, None, self.limits)
        return self._store_proposal(
            intent=old.intent_text, title=old.title, schedule=old.schedule, first_due=first_due, lifetime=lifetime, rolled=False,
            warning="", now=now, profile=old.profile, make_objective=lambda: old.objective, conversation_id=conversation_id,
            device_id=device_id, renews=old.id, open_discount=1 if old.status is _S.ACTIVE else 0,
        )

    def _require_not_running(self, old: ScheduledJob) -> None:
        """A run still in flight settles into THIS record - which a renewal ends - so its result (a failure that should count toward the
        breaker, say) would be lost to the renewed one. So nothing is renewed while a run is going: the scheduler's own definition of "in
        flight" (see `ScheduledJobScheduler.in_flight`), asked fail-closed - if it cannot be asked, the answer is no."""
        try:
            running = self._in_flight is None or self._in_flight(old.id)
        except Exception:  # noqa: BLE001 - a question that cannot be answered is not a "no"
            running = True
        if running:
            raise ScheduledJobRequestError(
                f"'{old.title}' is running right now. Let it finish first - I'll tell you when it's done - and then ask me to renew it."
            )

    @staticmethod
    def _require_renewable(old: ScheduledJob) -> None:
        name = f"'{old.title}'"
        if old.schedule.kind is ScheduleKind.ONCE:
            raise ScheduledJobRequestError(f"{name} is a one-time scheduled job, so it has no period to renew.")
        if old.status is _S.ACTIVE or (old.status is _S.COMPLETED and old.status_reason is _R.LIFETIME_ENDED):
            return
        if old.status is _S.PAUSED and old.status_reason is _R.BREAKER:
            raise ScheduledJobRequestError(f"{name} was paused after several failed runs in a row. Resume it first (that clears the failure count), then renew it if you still need to.")
        if old.status is _S.PAUSED:
            raise ScheduledJobRequestError(f"{name} is paused. Resume it first, then renew it if you still need to.")
        if old.status is _S.CANCELLED:
            raise ScheduledJobRequestError(f"You cancelled {name}, so there is nothing to renew. Ask me to schedule it again if you want it back.")
        if old.status is _S.COMPLETED:
            raise ScheduledJobRequestError(f"{name} has finished, so there is nothing to renew.")
        raise ScheduledJobRequestError(f"{name} isn't an approved scheduled job, so there is nothing to renew.")

    # -- running one now ---------------------------------------------------------------------------------------------------------------

    def run_now(self, scheduled_job_id: str) -> tuple[ScheduledJob, Any]:
        """Ask the scheduler to start ONE run of an approved Scheduled Job now. All the rules are the scheduler's (the same gates, the same
        launch as a due occurrence); this only hands the request over and returns what it said. Nothing here launches anything."""
        current = self._require(scheduled_job_id)
        if self._run_now is None:
            raise ScheduledJobRequestError("I can't start scheduled runs right now, so nothing was run.")
        try:
            result = self._run_now(current.id)
        except Exception:  # noqa: BLE001 - whatever went wrong, nothing is half-started: the launch is one transaction
            logger.warning("scheduled job run-now failed", exc_info=True)
            raise ScheduledJobRequestError("I couldn't start that just now, so nothing was run.") from None
        return current, result

    def _require_unattended(self, objective: dict[str, Any], profile: str) -> None:
        try:
            self._check_objective(objective, profile)
        except ScheduledJobError as exc:
            logger.info("scheduled_job refused at creation: %s", type(exc).__name__)
            raise ScheduledJobRequestError(UNATTENDED_REFUSAL) from None

    def discard(self, scheduled_job_id: str) -> None:
        """Drop a proposal that could not be put to the user (best effort: an approved or changed one is left as it is)."""
        current = self.store.get(scheduled_job_id)
        if current is None or current.status is not _S.PROPOSED:
            return
        try:
            self.store.transition(
                scheduled_job_id, _S.CANCELLED, _R.DECLINED, expected_version=current.version, via="chat", detail="could not be asked"
            )
        except ScheduledJobError:
            pass

    # -- approving -----------------------------------------------------------------------------------------------------------------

    def approve(self, scheduled_job_id: str, approval_interaction_id: str, renews: str | None = None) -> Approved:
        """PROPOSED -> ACTIVE, atomically (the store's compare-and-set records the approval, the first due time and the lifetime in ONE
        update, so the scheduler can never see a half-approved row). Called only after the user's confirmation was approved. The request
        is validated AGAIN against now: the time may have passed, the limits or the tools may have changed since it was proposed."""
        current = self._require(scheduled_job_id)
        if current.status is not _S.PROPOSED:
            return self._already(current, approval_interaction_id)
        now = self.now()
        if current.proposal_expires_at is not None and now > current.proposal_expires_at:
            self.sweep()
            raise ScheduledJobRequestError("That request expired before it was approved, so nothing was scheduled. Please ask me again.")
        due = next_due(current.schedule, now)
        try:
            validate_new_schedule(current.schedule, now=now, first_due=due, limits=self.limits)
            lifetime = lifetime_for(current.schedule, now, current.requested_until, self.limits)
        except ScheduledJobPolicyError as exc:
            raise ScheduledJobRequestError(f"{exc} Nothing was scheduled - please ask me again.") from None
        self._require_unattended(current.objective, current.profile)
        replaces_version = None
        if renews:  # a renewal may only replace the very thing it renews: the same instruction on the same schedule
            old = self.store.get(renews)
            if old is None or old.id == current.id or old.intent_text != current.intent_text or old.schedule.to_dict() != current.schedule.to_dict():
                raise ScheduledJobRequestError("That isn't a renewal of the scheduled job it names, so nothing was changed.")
            self._require_not_running(old)  # the user may have started a run since they were asked
            old = self.store.get(renews)  # read AFTER that check (it settles finished runs)...
            if old is None:
                raise ScheduledJobRequestError("The scheduled job to renew is gone, so nothing was changed.")
            self._require_not_running(old)  # ...and asked AGAIN: a run launched in between shows up here
            replaces_version = old.version  # and one launched after this changes the version, which the store holds the renewal to
        arguments = dict(
            expected_version=current.version, approved_at=now, approval_interaction_id=approval_interaction_id,
            due_at=due, expires_at=lifetime.expires_at, via="chat",
        )
        try:
            if renews:  # the renewed Scheduled Job ends in the SAME transaction: the two can never both be live
                activated = self.store.activate_renewal(current.id, replaces=renews, replaces_version=replaces_version, **arguments)
            else:
                activated = self.store.activate(current.id, **arguments)
        except (StaleScheduledJobError, InvalidTransitionError) as exc:
            again = self._require(scheduled_job_id)
            if renews and isinstance(exc, InvalidTransitionError) and again.status is _S.PROPOSED:
                raise ScheduledJobRequestError(f"{exc} Ask me again if you still want it renewed.") from None  # the one to renew changed
            return self._already(again, approval_interaction_id)  # raced: see what the winner did
        except ScheduledJobError as exc:
            raise ScheduledJobRequestError(f"{exc} Nothing was scheduled.") from None
        return Approved(activated, lifetime, renewed=bool(renews))

    def _already(self, current: ScheduledJob, approval_interaction_id: str) -> Approved:
        if current.approval_interaction_id == approval_interaction_id and current.status in (_S.ACTIVE, _S.PAUSED, _S.COMPLETED):
            return Approved(current, Lifetime(current.expires_at, False), already=True)  # the same approval, seen twice: harmless
        raise ScheduledJobRequestError("That request changed (or was already dealt with), so nothing was scheduled. Ask me again if you still want it.")

    # -- looking ---------------------------------------------------------------------------------------------------------------------

    def _require(self, scheduled_job_id: str) -> ScheduledJob:
        current = self.store.get(scheduled_job_id) if isinstance(scheduled_job_id, str) and scheduled_job_id else None
        if current is None:
            raise ScheduledJobRequestError("I couldn't find that scheduled job.")
        return current

    def listing(self, limit: int = 15) -> list[ScheduledJob]:
        """What the user has: live and paused ones, then (most recent first) ones that ran their course. Cancelled and expired
        proposals are gone from view."""
        self.sweep()
        shown = self.store.list(frozenset({_S.ACTIVE, _S.PAUSED, _S.COMPLETED}))
        order = {_S.ACTIVE: 0, _S.PAUSED: 1, _S.COMPLETED: 2}
        return sorted(shown, key=lambda sj: (order[sj.status], sj.due_at or sj.updated_at))[:limit]

    def last_run(self, scheduled_job_id: str) -> ScheduledJobRun | None:
        runs = self.store.runs_for(scheduled_job_id, limit=1)
        return runs[0] if runs else None

    # -- the user's own controls --------------------------------------------------------------------------------------------------------

    def pause(self, scheduled_job_id: str) -> ScheduledJob:
        current = self._require(scheduled_job_id)
        if current.status is _S.PAUSED:
            raise ScheduledJobRequestError(f"'{current.title}' is already paused.")
        if current.status is not _S.ACTIVE:
            raise ScheduledJobRequestError(f"'{current.title}' isn't running, so there is nothing to pause.")
        return self._move(current, _S.PAUSED, _R.USER)

    def resume(self, scheduled_job_id: str) -> tuple[ScheduledJob, datetime]:
        """Make a paused one live again at its next FUTURE occurrence (whatever came due while paused is never replayed). Pausing never
        stopped the clock on the approved period: once it has ended only a fresh approval can start another."""
        current = self._require(scheduled_job_id)
        if current.status is not _S.PAUSED:
            raise ScheduledJobRequestError(f"'{current.title}' isn't paused.")
        now = self.now()
        if current.expires_at is not None and now >= current.expires_at:
            raise ScheduledJobRequestError(
                f"The period you approved for '{current.title}' has ended, so it can't be resumed. Ask me to schedule it again if you still want it."
            )
        upcoming = next_future_due(current.schedule, now, current.expires_at)
        if upcoming is None:
            raise ScheduledJobRequestError(f"'{current.title}' has nothing left to run (its time has passed, or its approved period ends first).")
        try:
            return self.store.resume(current.id, next_due_at=upcoming, expected_version=current.version, via="chat", now=now), upcoming
        except (StaleScheduledJobError, InvalidTransitionError):
            raise ScheduledJobRequestError(f"'{current.title}' changed while I was resuming it - please try again.") from None
        except ScheduledJobError as exc:
            raise ScheduledJobRequestError(str(exc)) from None

    def cancel(self, scheduled_job_id: str) -> ScheduledJob:
        current = self._require(scheduled_job_id)
        if current.status in (_S.ACTIVE, _S.PAUSED):
            return self._move(current, _S.CANCELLED, _R.USER)
        if current.status is _S.PROPOSED:
            return self._move(current, _S.CANCELLED, _R.DECLINED)
        raise ScheduledJobRequestError(f"'{current.title}' isn't running, so there is nothing to cancel.")

    def _move(self, current: ScheduledJob, target: ScheduledJobStatus, reason: StatusReason) -> ScheduledJob:
        try:
            return self.store.transition(current.id, target, reason, expected_version=current.version, via="chat", now=self.now())
        except (StaleScheduledJobError, InvalidTransitionError):
            raise ScheduledJobRequestError(f"'{current.title}' changed while I was doing that - please look again and retry.") from None
