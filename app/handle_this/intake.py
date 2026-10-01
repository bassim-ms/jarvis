"""
Handle This intake: from the user's words to an UNDERSTOOD request, and no
further.

    request
      -> interpret            (deterministic first; the AI only if that finds no signal)
      -> capability boundary  (is this kind of objective supported - and available now?)
      -> relevant context     (the existing Context Engine, selectively; never guessed at)
      -> clarification policy (ask only what is materially missing)
      -> HandleThisRequest    (READY / NEEDS_CLARIFICATION / UNSUPPORTED) or a structured failure

THE SEPARATION THAT MATTERS. This pipeline decides WHAT the user wants, which
broad family of work it is, which constraints and context bear on it, whether
something material is missing, and whether it is supported. It does NOT decide
steps, tools, dependencies, ordering or how anything is executed - that is the
Planner's, later. `HandleThisIntake` therefore has no ToolRegistry, JobStore,
subprocess, file or network access to misuse; it only reads the request, calls
an injected interpreter and an injected read-only context source, and returns
data. It creates no Jobs, no interactions and no notifications, changes no
permission or autonomy, and writes nothing to memory or the World Model.

FAILURE IS EXPLICIT. If the request cannot be read, the objective cannot be
made out, the interpreter is down or answers with something invalid, or the
Context Engine (when there is one) cannot be queried, the result is a
structured UnderstandingFailure - never a guess presented as an understanding,
and never a "fully understood" request built on context that was not retrieved.

LOGGING is structured and free of the user's words: only an id, the family,
the status, the method and counts.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable

from app.handle_this.base import (
    MAX_REQUEST_CHARS,
    Clarification,
    ContextHandoff,
    ContextStatus,
    FailureCode,
    GoalFamily,
    HandleThisRequest,
    HandleThisStatus,
    UnderstandingFailure,
    UnderstandingMeta,
    UnderstandingResult,
    UnsupportedCategory,
    UnsupportedReason,
    new_request_id,
)
from app.handle_this.boundary import DEFAULT_BOUNDARY, CapabilityBoundary
from app.handle_this.clarification import apply_policy
from app.handle_this.context import ContextSource, ContextUnavailableError, categories_for
from app.handle_this.signals import detect_computer_actions
from app.handle_this.interpretation import (
    DeterministicGoalInterpreter,
    GoalInterpreter,
    Interpretation,
    InterpreterUnavailableError,
    InvalidInterpretationError,
)

logger = logging.getLogger("jarvis.handle_this")


def _failure(code: FailureCode) -> UnderstandingResult:
    logger.warning("handle_this understanding_failed code=%s", code.value)
    return UnderstandingResult(failure=UnderstandingFailure(code))


class HandleThisIntake:
    """See the module docstring.

    `llm_interpreter` is consulted only when the deterministic interpreter
    finds no signal. `context_source` is the read-only Context Engine adapter
    (None = no Context Engine in this deployment). `available_tools` returns
    the names of the tools the eventual Planner would be offered; None skips
    the availability check and applies only the static family support."""

    def __init__(
        self,
        *,
        llm_interpreter: GoalInterpreter | None = None,
        deterministic: DeterministicGoalInterpreter | None = None,
        context_source: ContextSource | None = None,
        boundary: CapabilityBoundary = DEFAULT_BOUNDARY,
        available_tools: Callable[[], Iterable[str]] | None = None,
        id_factory: Callable[[], str] = new_request_id,
    ):
        self.llm_interpreter = llm_interpreter
        self.deterministic = deterministic or DeterministicGoalInterpreter()
        self.context_source = context_source
        self.boundary = boundary
        self.available_tools = available_tools
        self._new_id = id_factory

    def understand(self, message: str) -> UnderstandingResult:
        """Understand `message`. Never raises: any unexpected fault becomes an
        INTERNAL_ERROR failure."""
        try:
            return self._understand(message)
        except Exception:  # noqa: BLE001 - fail closed, without leaking the fault
            return _failure(FailureCode.INTERNAL_ERROR)

    # -- the pipeline ---------------------------------------------------------------------------

    def _understand(self, message: str) -> UnderstandingResult:
        if not isinstance(message, str) or not message.strip() or len(message) > MAX_REQUEST_CHARS:
            return _failure(FailureCode.INVALID_REQUEST)

        interpretation = self.deterministic.interpret(message)
        if interpretation is None:
            if self.llm_interpreter is None:
                return _failure(FailureCode.UNCLEAR_OBJECTIVE)
            try:
                interpretation = self.llm_interpreter.interpret(message)
            except InterpreterUnavailableError:
                return _failure(FailureCode.INTERPRETER_UNAVAILABLE)
            except InvalidInterpretationError:
                return _failure(FailureCode.INTERPRETER_INVALID_OUTPUT)

        meta = UnderstandingMeta(interpretation.method, interpretation.confidence)

        if interpretation.goal_family is GoalFamily.UNSUPPORTED:
            return self._unsupported(
                message, interpretation, meta, UnsupportedReason(interpretation.unsupported_category)
            )

        unavailable = self.boundary.assess(
            interpretation.required_families,
            self.available_tools() if self.available_tools is not None else None,
        ).unavailable
        if unavailable:
            return self._unsupported(
                message,
                interpretation,
                meta,
                UnsupportedReason(UnsupportedCategory.UNAVAILABLE_CAPABILITY, tuple(unavailable)),
            )

        if GoalFamily.COMPUTER_WORKFLOW in interpretation.required_families:
            # A recognised computer workflow does NOT grant every computer capability: each action the wording asks
            # for needs its own tools actually on offer to the Planner. One that is not (a click, typing, filling a
            # form, launching an application) makes the whole objective unsupported - never a partial plan that
            # quietly does less than was asked.
            missing = self.boundary.assess_actions(
                detect_computer_actions(message),
                self.available_tools() if self.available_tools is not None else None,
            )
            if missing:
                return self._unsupported(
                    message,
                    interpretation,
                    meta,
                    UnsupportedReason(UnsupportedCategory.UNAVAILABLE_CAPABILITY, (GoalFamily.COMPUTER_WORKFLOW,)),
                )

        context = self._context(message, interpretation)
        if context is None:
            return _failure(FailureCode.CONTEXT_UNAVAILABLE)

        outcome = apply_policy(message, interpretation, context)
        clarification = Clarification(outcome.needs) if outcome.needs else None
        request = HandleThisRequest(
            id=self._new_id(),
            original_request=message,  # exactly as received
            interpreted_goal=interpretation.interpreted_goal,
            goal_family=interpretation.goal_family,
            required_families=interpretation.required_families,
            status=HandleThisStatus.NEEDS_CLARIFICATION if clarification else HandleThisStatus.READY,
            understanding=meta,
            constraints=outcome.constraints,
            assumptions=outcome.assumptions,
            success_criteria=interpretation.success_criteria,
            context=context,
            clarification=clarification,
        )
        logger.info(
            "handle_this understood request=%s family=%s status=%s method=%s constraints=%d needs=%d",
            request.id,
            request.goal_family.value,
            request.status.value,
            meta.method,
            len(request.constraints),
            len(outcome.needs),
        )
        return UnderstandingResult(request=request)

    def _context(self, message: str, interpretation: Interpretation) -> ContextHandoff | None:
        """The relevant context, a NOT_CONFIGURED handoff when this deployment
        has no Context Engine, or None when the engine could not be queried."""
        if self.context_source is None:
            return ContextHandoff(status=ContextStatus.NOT_CONFIGURED)
        logger.debug(
            "handle_this context_requested categories=%d",
            len(categories_for(interpretation.required_families)),
        )
        try:
            return self.context_source.retrieve(message, interpretation.required_families)
        except ContextUnavailableError:
            return None

    def _unsupported(
        self,
        message: str,
        interpretation: Interpretation,
        meta: UnderstandingMeta,
        reason: UnsupportedReason,
    ) -> UnderstandingResult:
        request = HandleThisRequest(
            id=self._new_id(),
            original_request=message,
            interpreted_goal=interpretation.interpreted_goal,
            goal_family=GoalFamily.UNSUPPORTED,
            required_families=(),
            status=HandleThisStatus.UNSUPPORTED,
            understanding=meta,
            unsupported=reason,
        )
        logger.info(
            "handle_this unsupported request=%s category=%s method=%s",
            request.id,
            reason.category.value,
            meta.method,
        )
        return UnderstandingResult(request=request)
