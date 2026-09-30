"""
Handle This -> existing Job: what happens to a goal-driven request once it has
been understood.

`HandleThisService.handle(text, conversation_id)` runs the intake (Slice 1)
and turns the result into exactly ONE of four outcomes:

    START_JOB            the request is READY and supported. The outcome carries
                         the ONLY arguments the existing `create_job` tool may
                         be called with, plus what to tell the user.
    NEEDS_CLARIFICATION  something material is missing. The outcome carries the
                         arguments for a Job that is created already WAITING on
                         ONE fixed question (the same `create_job` tool, its
                         `clarification` argument): no planner, no tools, no
                         execution until the user answers. The answer resumes the
                         SAME Job (see app/handle_this/continuation.py).
    UNSUPPORTED          outside the boundary (or not available to Jobs right
                         now). No Job, no planner, no tools, and nothing about
                         capabilities or internals is revealed.
    FAILED               it could not be understood (or its context could not
                         be looked up). No Job, no tools, a safe message.

THE ARCHITECTURE STAYS EXACTLY AS IT WAS. This service holds no ToolRegistry,
no JobStore and no planner. It cannot create a Job, run a tool or call the
Planner: it returns DATA, and the existing chat path (JarvisCore._handle_task)
creates the Job through the existing `create_job` tool - a PENDING Job that
JobEngine later plans with the EXISTING Planner, validates with the EXISTING
plan validator, and runs through the EXISTING ToolRegistry, permission and
autonomy checks. What Handle This adds is an understood objective attached to
that Job (see OBJECTIVE_KEY), which the planner receives as data.

What `job_params` deliberately never contains: a tool name, a step, a
permission, an autonomy level, an autonomous-permissions set, or a priority.
The Job therefore gets the same defaults (ASSISTED autonomy, confirmation
required for anything that needs it) as any Job started from chat. "Handle
this" means "do what you are already allowed to do" - it authorises nothing.

Computer workflows are not handed to Jobs: any request that needs that family
is answered as unsupported here even if some deployment's Job-facing registry
offered such tools.

A clarification asks ONE question at a time (the first material need; a later
one is raised by the continuation if it is still needed). If the deployment
cannot park a Job on a question (no interaction store), the outcome's
`fallback_reply` - "send the request again with that included" - is used and
nothing is created.
"""

from __future__ import annotations

import logging
import zlib
from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.handle_this.clarification import QUESTIONS
from app.handle_this.base import (
    ClarificationNeed,
    FailureCode,
    GoalFamily,
    HandleThisRequest,
    HandleThisStatus,
)
from app.handle_this.intake import HandleThisIntake

logger = logging.getLogger("jarvis.handle_this")

#: The families a Job may be handed. COMPUTER_WORKFLOW is deliberately absent.
HANDOFF_FAMILIES = frozenset({GoalFamily.RESEARCH, GoalFamily.ANALYSIS, GoalFamily.ORGANIZATION})

_ADDRESSES = ("Sir", "Boss")


def address_for(key: str) -> str:
    """"Sir" or "Boss" - the only forms of address Jarvis uses - fixed for a
    given request. (Same rule as app/notifications/text.py; a test pins them.)"""
    return _ADDRESSES[zlib.crc32(key.encode("utf-8")) % len(_ADDRESSES)]


class OutcomeKind(Enum):
    START_JOB = "start_job"
    NEEDS_CLARIFICATION = "needs_clarification"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


@dataclass(frozen=True)
class HandleThisOutcome:
    """See the module docstring. `job_params` is set for START_JOB and for
    NEEDS_CLARIFICATION (the latter includes a `clarification`), never
    otherwise. `fallback_reply` is what to say if a clarification Job cannot be
    created. `request` is the understood request (None when understanding
    failed) - kept for callers that want to look, never needed to act."""

    kind: OutcomeKind
    reply: str
    job_params: dict[str, Any] | None = None
    already_running_reply: str | None = None
    #: What to say if the same request is sent again after its question was
    #: ANSWERED (the Job is underway, no longer waiting): only for a
    #: clarification outcome, whose `already_running_reply` is about waiting.
    underway_reply: str | None = None
    fallback_reply: str | None = None
    request: HandleThisRequest | None = None


# --- what the user is told ------------------------------------------------------------------------------------------------
# Plain, warm, one assistant: no internal terms (no "plan", "job engine",
# "capability", "tool"), no capability lists, no stack traces.

_STARTED = {
    GoalFamily.RESEARCH: "I'll do the research and come back with what I find.",
    GoalFamily.ANALYSIS: "I'll work through that and come back with my analysis.",
    GoalFamily.ORGANIZATION: "I'll get that organised and let you know when it's done.",
    GoalFamily.MIXED: "I'll work through all of that and come back to you.",
}

_UNSUPPORTED = (
    "I understand what you're asking, but I don't currently have the capability "
    "to handle that autonomously."
)

_FAILURES = {
    FailureCode.UNCLEAR_OBJECTIVE: (
        "I couldn't quite work out what you'd like me to do. Could you tell me a bit "
        "more about the outcome you want?"
    ),
    FailureCode.INVALID_REQUEST: "I couldn't read that request. Could you say it again?",
}
_FAILURE_DEFAULT = "I couldn't get started on that just now. Please try again in a moment."


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _started_reply(request: HandleThisRequest) -> str:
    return f"{address_for(request.id)}, I've got it. {_STARTED[request.goal_family]}"


def _already_running_reply(request: HandleThisRequest) -> str:
    return f"{address_for(request.id)}, I'm already working on that."


def _clarification_reply(request: HandleThisRequest) -> str:
    needs: tuple[ClarificationNeed, ...] = request.clarification.needs
    wanted = [_lower_first(n.missing) for n in needs]
    listed = wanted[0] if len(wanted) == 1 else ", ".join(wanted[:-1]) + " and " + wanted[-1]
    return (
        f"{address_for(request.id)}, before I start I need to know {listed}. "
        "Send me the request again with that included and I'll take it from there."
    )


class HandleThisService:
    """See the module docstring."""

    def __init__(self, intake: HandleThisIntake):
        self.intake = intake

    def handle(self, text: str, conversation_id: str | None) -> HandleThisOutcome:
        """Never raises: any unexpected fault is a FAILED outcome (no Job)."""
        try:
            return self._handle(text, conversation_id)
        except Exception:  # noqa: BLE001 - fail closed, without leaking the fault
            logger.warning("handle_this service_failed")
            return HandleThisOutcome(OutcomeKind.FAILED, f"Sir, {_FAILURE_DEFAULT}")

    def _handle(self, text: str, conversation_id: str | None) -> HandleThisOutcome:
        result = self.intake.understand(text)
        if not result.ok:
            reply = _FAILURES.get(result.failure.code, _FAILURE_DEFAULT)
            return HandleThisOutcome(OutcomeKind.FAILED, f"Sir, {reply}")

        request = result.request
        who = address_for(request.id)

        if request.status is HandleThisStatus.UNSUPPORTED:
            return self._unsupported(request)
        if request.status is HandleThisStatus.NEEDS_CLARIFICATION:
            return self._needs_clarification(request, conversation_id)
        if request.status is not HandleThisStatus.READY:  # nothing else may fall through to a Job
            return HandleThisOutcome(OutcomeKind.FAILED, f"{who}, {_FAILURE_DEFAULT}")
        if not set(request.required_families) <= HANDOFF_FAMILIES:
            return self._unsupported(request)  # e.g. a computer workflow: never handed to a Job

        logger.info("handle_this start_job request=%s family=%s", request.id, request.goal_family.value)
        return HandleThisOutcome(
            OutcomeKind.START_JOB,
            _started_reply(request),
            job_params=self._job_params(request, conversation_id),
            already_running_reply=_already_running_reply(request),
            request=request,
        )

    def _needs_clarification(
        self, request: HandleThisRequest, conversation_id: str | None
    ) -> HandleThisOutcome:
        """Ask the FIRST material need, as its fixed question. The Job is
        created waiting on it; the question is also what the user is told."""
        need = request.clarification.needs[0]
        question = QUESTIONS[need.topic]
        params = self._job_params(request, conversation_id)
        params["clarification"] = {"question": question, "topic": need.topic.value}
        who = address_for(request.id)
        logger.info("handle_this needs_clarification request=%s topic=%s", request.id, need.topic.value)
        return HandleThisOutcome(
            OutcomeKind.NEEDS_CLARIFICATION,
            f"{who}, {_lower_first(question)}",
            job_params=params,
            already_running_reply=f"{who}, I'm still waiting for your answer: {question}",
            underway_reply=_already_running_reply(request),
            fallback_reply=_clarification_reply(request),
            request=request,
        )

    @staticmethod
    def _unsupported(request: HandleThisRequest) -> HandleThisOutcome:
        return HandleThisOutcome(
            OutcomeKind.UNSUPPORTED, f"{address_for(request.id)}, {_UNSUPPORTED}", request=request
        )

    @staticmethod
    def _job_params(request: HandleThisRequest, conversation_id: str | None) -> dict[str, Any]:
        """The ONLY arguments the existing `create_job` tool is called with:
        the user's words as the goal, the conversation, the relevant context
        (in the form the chat path already gives a Job's planner) and the
        understood objective. Nothing that selects a tool, grants a
        permission or changes autonomy - by construction, not by filtering."""
        params: dict[str, Any] = {
            "goal": request.original_request,
            "conversation_id": conversation_id,
            "objective": {
                "original_request": request.original_request,
                "interpreted_goal": request.interpreted_goal,
                "goal_family": request.goal_family.value,
                "required_families": [f.value for f in request.required_families],
                "constraints": [
                    {"topic": c.topic.value, "text": c.text, "source": c.source.value}
                    for c in request.constraints
                ],
                "success_criteria": list(request.success_criteria),
                "assumptions": [{"topic": a.topic.value, "text": a.text} for a in request.assumptions],
            },
        }
        context = request.context
        if context is not None and context.categories:
            params["request_context"] = {label: list(items) for label, items in context.categories.items()}
        return params
