"""
ContextEngine (M16.2): the read-only retrieval/interpretation layer
between a RequestContext (app/context_engine/base.py) and a
ContextSnapshot (app/world_model/base.py).

    RequestContext -> candidate retrieval -> hard filtering ->
    relevance scoring -> budget selection -> ContextSnapshot

STRICTLY READ-ONLY: every World Model access here goes through
WorldModelService's/WorldModelStore's own read methods (list_*/get_*) -
never save_*/create_*/transition_*/link_*/add_*_link. ContextEngine never
executes a Tool, never touches ToolRegistry, never mutates a Job, and
never approves a confirmation - M15 remains the sole execution authority,
ToolRegistry remains the sole action layer, and this module only ever
reads and reshapes state that already exists (M16.2's own explicit
non-goals).

Memory stays a separate, optional retrieval source (never redesigned
here): ContextEngine only ever calls MemoryStore.get_facts() - the same
existing method app/automation/base.py:ConditionContext and
app/tools/defaults.py's CreateJobTool context_provider already use - and
only when RequiredContextCategory.MEMORY is actually requested and a
MemoryStore was injected.

Task/Job linkage (TaskLink) is resolved by id/ref only - see this
module's own `task_label_provider` parameter docstring below for the one
place a caller can optionally supply human-readable Task/Job text
without this module ever importing app.tasks/app.orchestration itself.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from app.context_engine.base import (
    CategoryRetrieval,
    ContextBudget,
    ContextWindow,
    ItemProvenance,
    RequestContext,
    RequiredContextCategory,
)
from app.context_engine.scoring import (
    DEFAULT_WEIGHTS,
    RelevanceWeights,
    any_hint_matches,
    matches_hint,
    score_blocker,
    score_commitment,
    score_event,
    score_goal,
    score_project,
    score_resource,
    tie_break_key,
)
from app.memory.base import MemoryStore
from app.world_model.base import (
    Blocker,
    BlockerStatus,
    Commitment,
    CommitmentStatus,
    ContextSnapshot,
    Event,
    Goal,
    GoalStatus,
    LinkedEntityKind,
    Preference,
    Project,
    ProjectStatus,
    Resource,
    TaskLink,
    TERMINAL_COMMITMENT_STATUSES,
    TERMINAL_GOAL_STATUSES,
    TERMINAL_PROJECT_STATUSES,
)
from app.world_model.service import WorldModelService


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ContextEngine:
    """Builds a ContextSnapshot for one RequestContext against one
    WorldModelService - read-only, deterministic, no AI call.

    `task_label_provider`, if given, is called as
    `task_label_provider(link)` for each TaskLink candidate and should
    return a human-readable label for the linked Task/Job (e.g. its
    `text`/`goal`) or None if unavailable. This is the ONLY way
    human-readable text for a linked Task/Job ever reaches this module -
    ContextEngine itself never imports app.tasks or app.orchestration,
    keeping it decoupled from both execution systems (matches
    TaskLink's own "ref_id is stored opaquely" design, app/world_model/
    base.py). Without a provider, explicit "mentioned_tasks" matching
    falls back to comparing hints against the TaskLink's raw `ref_id`,
    which will rarely match a human phrase - callers that want real
    explicit-task-match behavior should supply one (e.g. wiring it to
    `task_store.get(link.ref_id).text` / `job_store.get(link.ref_id).goal`
    as appropriate for `link.ref_type`).
    """

    def __init__(
        self,
        world_model: WorldModelService,
        memory: MemoryStore | None = None,
        *,
        weights: RelevanceWeights = DEFAULT_WEIGHTS,
        task_label_provider: Callable[[TaskLink], str | None] | None = None,
        candidate_source: Callable[[], list] | None = None,
    ):
        self.world_model = world_model
        self.memory = memory
        # Zero-arg callable returning the staged, unconfirmed candidates
        # (`UnconfirmedCandidate`s - see app/memory_policy/). `None` (the default,
        # every existing caller/test): no candidate ever reaches a snapshot. The
        # engine only READS them, filters by confidence, and hands them on labelled
        # unconfirmed in their own field; it never promotes one.
        self.candidate_source = candidate_source
        self.weights = weights
        self.task_label_provider = task_label_provider

    # -- public entry point ------------------------------------------------

    def build_snapshot(
        self, request_context: RequestContext, *, now: str | None = None, device_context=None
    ) -> ContextSnapshot:
        current_time = now or _now_iso()
        required = request_context.required_context
        budget = request_context.context_budget
        retrieval_metadata: dict[str, CategoryRetrieval] = {}

        # Projects are resolved once regardless of whether PROJECTS
        # itself is in `required` - other categories (goals, commitments,
        # events, blockers, resources) score a Project->X relationship
        # signal off of `explicit_project_ids`, so this one cheap,
        # unfiltered list_projects() call is shared rather than repeated.
        all_projects = self.world_model.store.list_projects()
        explicit_project_ids = {
            p.id
            for p in all_projects
            if any_hint_matches(request_context.mentioned_projects, p.name)
        }

        active_projects: list[Project] = []
        if RequiredContextCategory.PROJECTS in required:
            active_projects, retrieval_metadata["projects"] = self._select_projects(
                all_projects, request_context, explicit_project_ids, current_time, budget
            )

        relevant_goals: list[Goal] = []
        explicit_goal_ids: set[str] = set()
        all_goals = self.world_model.store.list_goals()
        explicit_goal_ids = {
            g.id
            for g in all_goals
            if any_hint_matches(request_context.mentioned_goals, g.title)
        }
        if RequiredContextCategory.GOALS in required:
            relevant_goals, retrieval_metadata["goals"] = self._select_goals(
                all_goals,
                request_context,
                explicit_project_ids,
                explicit_goal_ids,
                current_time,
                budget,
            )

        relevant_tasks: list[TaskLink] = []
        if RequiredContextCategory.TASKS in required:
            relevant_tasks, retrieval_metadata["tasks"] = self._select_task_links(
                request_context, explicit_project_ids, explicit_goal_ids, budget
            )

        upcoming_commitments: list[Commitment] = []
        if RequiredContextCategory.COMMITMENTS in required:
            upcoming_commitments, retrieval_metadata["commitments"] = self._select_commitments(
                request_context, explicit_project_ids, current_time, budget
            )

        upcoming_events: list[Event] = []
        if RequiredContextCategory.EVENTS in required:
            upcoming_events, retrieval_metadata["events"] = self._select_events(
                request_context, explicit_project_ids, current_time, budget
            )

        active_blockers: list[Blocker] = []
        if RequiredContextCategory.BLOCKERS in required:
            active_blockers, retrieval_metadata["blockers"] = self._select_blockers(
                request_context, explicit_project_ids, explicit_goal_ids, budget
            )

        relevant_resources: list[Resource] = []
        if RequiredContextCategory.RESOURCES in required:
            relevant_resources, retrieval_metadata["resources"] = self._select_resources(
                request_context, explicit_project_ids, budget
            )

        relevant_preferences: list[Preference] = []
        if RequiredContextCategory.PREFERENCES in required:
            relevant_preferences, retrieval_metadata["preferences"] = self._select_preferences(
                budget
            )

        relevant_memory: list[str] = []
        if RequiredContextCategory.MEMORY in required and self.memory is not None:
            relevant_memory = self._select_memory(budget)

        unconfirmed_candidates: list = []
        if RequiredContextCategory.MEMORY in required and self.candidate_source is not None:
            unconfirmed_candidates = self._select_candidates()

        return ContextSnapshot(
            current_time=current_time,
            active_projects=active_projects,
            relevant_goals=relevant_goals,
            relevant_tasks=relevant_tasks,
            upcoming_commitments=upcoming_commitments,
            upcoming_events=upcoming_events,
            active_blockers=active_blockers,
            relevant_resources=relevant_resources,
            relevant_preferences=relevant_preferences,
            relevant_memory=relevant_memory,
            unconfirmed_candidates=unconfirmed_candidates,
            # Who sent this request (from the Device Gateway), when known: the RequestContext's own, unless a
            # caller passes one explicitly. Carried through unchanged - never stored, never derived.
            device_context=device_context if device_context is not None else request_context.device_context,
            context_window=ContextWindow(
                start=request_context.time_scope.start, end=request_context.time_scope.end
            ),
            retrieval_metadata=retrieval_metadata,
        )

    # -- Projects -----------------------------------------------------------

    def _select_projects(
        self,
        all_projects: list[Project],
        request_context: RequestContext,
        explicit_project_ids: set[str],
        current_time: str,
        budget: ContextBudget,
    ) -> tuple[list[Project], CategoryRetrieval]:
        # Hard filter: archived/completed (terminal) projects are
        # normally excluded, UNLESS the user explicitly named them - an
        # explicitly-requested project is never silently dropped, and
        # still carries its real (e.g. ARCHIVED) status in the returned
        # Project object, so it stays "clearly marked as archived."
        candidates = [
            p
            for p in all_projects
            if p.status not in TERMINAL_PROJECT_STATUSES or p.id in explicit_project_ids
        ]

        scored: list[tuple[float, list[str], Project]] = []
        for project in candidates:
            has_blocker = any(
                b.status is BlockerStatus.ACTIVE
                for b in self.world_model.get_blockers_for_entity(
                    LinkedEntityKind.PROJECT, project.id
                )
            )
            score, reasons = score_project(
                project,
                explicit_hints=request_context.mentioned_projects,
                current_time=current_time,
                has_active_blocker=has_blocker,
                weights=self.weights,
            )
            scored.append((score, reasons, project))

        selected = self._budget_select(
            scored, budget.projects, key_fn=lambda p: (p.priority, None, p.updated_at, p.id)
        )
        return (
            [p for _, _, p in selected],
            # `considered` = candidates entering the pipeline BEFORE hard
            # filtering (len(all_projects)), never the post-filter count -
            # see CategoryRetrieval's own docstring.
            self._to_category_retrieval(len(all_projects), selected),
        )

    # -- Goals ----------------------------------------------------------

    def _select_goals(
        self,
        all_goals: list[Goal],
        request_context: RequestContext,
        explicit_project_ids: set[str],
        explicit_goal_ids: set[str],
        current_time: str,
        budget: ContextBudget,
    ) -> tuple[list[Goal], CategoryRetrieval]:
        candidates = [
            g
            for g in all_goals
            if g.status not in TERMINAL_GOAL_STATUSES
            or g.id in explicit_goal_ids
            or g.project_id in explicit_project_ids
        ]

        scored = []
        for goal in candidates:
            has_blocker = any(
                b.status is BlockerStatus.ACTIVE
                for b in self.world_model.get_blockers_for_entity(
                    LinkedEntityKind.GOAL, goal.id
                )
            )
            has_dependency = bool(self.world_model.get_goal_dependencies(goal.id))
            score, reasons = score_goal(
                goal,
                explicit_hints=request_context.mentioned_goals,
                project_explicitly_matched=goal.project_id in explicit_project_ids,
                current_time=current_time,
                has_active_blocker=has_blocker,
                has_dependency=has_dependency,
                weights=self.weights,
            )
            scored.append((score, reasons, goal))

        selected = self._budget_select(
            scored,
            budget.goals,
            key_fn=lambda g: (g.priority, g.target_date, g.updated_at, g.id),
        )
        return (
            [g for _, _, g in selected],
            self._to_category_retrieval(len(all_goals), selected),
        )

    # -- Task/Job linkage ---------------------------------------------------

    def _select_task_links(
        self,
        request_context: RequestContext,
        explicit_project_ids: set[str],
        explicit_goal_ids: set[str],
        budget: ContextBudget,
    ) -> tuple[list[TaskLink], CategoryRetrieval]:
        candidates = self.world_model.store.list_task_links()

        scored = []
        for link in candidates:
            label = self.task_label_provider(link) if self.task_label_provider else None
            explicit = any_hint_matches(
                request_context.mentioned_tasks, label or link.ref_id
            )
            relationship = (
                link.project_id in explicit_project_ids
                or link.goal_id in explicit_goal_ids
            )
            score = 0.0
            reasons: list[str] = []
            if explicit:
                score += self.weights.EXPLICIT_MATCH
                reasons.append("explicit_match")
            elif relationship:
                score += self.weights.RELATIONSHIP
                reasons.append("relationship_to_project")
            scored.append((score, reasons, link))

        selected = self._budget_select(
            scored, budget.tasks, key_fn=lambda link: (None, None, link.created_at, link.id)
        )
        return (
            [link for _, _, link in selected],
            self._to_category_retrieval(len(candidates), selected),
        )

    # -- Commitments ------------------------------------------------------

    def _select_commitments(
        self,
        request_context: RequestContext,
        explicit_project_ids: set[str],
        current_time: str,
        budget: ContextBudget,
    ) -> tuple[list[Commitment], CategoryRetrieval]:
        all_commitments = self.world_model.store.list_commitments()
        explicit_ids = {
            c.id
            for c in all_commitments
            if any_hint_matches(request_context.mentioned_commitments, c.title)
        }
        # Fulfilled/cancelled commitments are excluded from "upcoming"
        # context by default; MISSED stays IN by default (it's still
        # meaningful - an overdue obligation, not resolved) unless the
        # caller wants strict "still open" semantics via explicit filtering
        # downstream. Any terminal commitment named explicitly is kept.
        candidates = [
            c
            for c in all_commitments
            if c.status not in {CommitmentStatus.FULFILLED, CommitmentStatus.CANCELLED}
            or c.id in explicit_ids
            or c.project_id in explicit_project_ids
        ]

        scored = []
        for commitment in candidates:
            score, reasons = score_commitment(
                commitment,
                explicit_hints=request_context.mentioned_commitments,
                project_explicitly_matched=commitment.project_id in explicit_project_ids,
                current_time=current_time,
                weights=self.weights,
            )
            scored.append((score, reasons, commitment))

        selected = self._budget_select(
            scored,
            budget.commitments,
            key_fn=lambda c: (c.priority, c.due_at, c.updated_at, c.id),
        )
        return (
            [c for _, _, c in selected],
            self._to_category_retrieval(len(all_commitments), selected),
        )

    # -- Events -----------------------------------------------------------

    def _select_events(
        self,
        request_context: RequestContext,
        explicit_project_ids: set[str],
        current_time: str,
        budget: ContextBudget,
    ) -> tuple[list[Event], CategoryRetrieval]:
        all_events = self.world_model.store.list_events()
        explicit_ids = {
            e.id
            for e in all_events
            if any_hint_matches(request_context.mentioned_projects, e.title)
        }
        # Past events are excluded from "upcoming" by default unless
        # historical context was explicitly requested (TimeScopeKind.
        # OVERDUE, this module's closest equivalent to "explicitly asking
        # about the past") or the event itself was explicitly named.
        include_past = request_context.time_scope.kind.value == "overdue"
        candidates = [
            e
            for e in all_events
            if e.starts_at >= current_time
            or include_past
            or e.id in explicit_ids
            or e.project_id in explicit_project_ids
        ]

        scored = []
        for event in candidates:
            score, reasons = score_event(
                event,
                explicit_hints=request_context.mentioned_projects,
                project_explicitly_matched=event.project_id in explicit_project_ids,
                current_time=current_time,
                weights=self.weights,
            )
            scored.append((score, reasons, event))

        selected = self._budget_select(
            scored,
            budget.events,
            key_fn=lambda e: (None, e.starts_at, e.updated_at, e.id),
        )
        return (
            [e for _, _, e in selected],
            self._to_category_retrieval(len(all_events), selected),
        )

    # -- Blockers ---------------------------------------------------------

    def _select_blockers(
        self,
        request_context: RequestContext,
        explicit_project_ids: set[str],
        explicit_goal_ids: set[str],
        budget: ContextBudget,
    ) -> tuple[list[Blocker], CategoryRetrieval]:
        all_blockers = self.world_model.store.list_blockers()
        explicit_hints = (
            request_context.mentioned_projects
            + request_context.mentioned_goals
            + request_context.mentioned_tasks
        )
        explicit_ids = {
            b.id for b in all_blockers if any_hint_matches(explicit_hints, b.title)
        }

        def _linked_to_explicit(blocker: Blocker) -> bool:
            for link in self.world_model.store.list_blocker_links_for_blocker(blocker.id):
                if link.entity_kind is LinkedEntityKind.PROJECT and link.entity_id in explicit_project_ids:
                    return True
                if link.entity_kind is LinkedEntityKind.GOAL and link.entity_id in explicit_goal_ids:
                    return True
            return False

        candidates = [
            b
            for b in all_blockers
            if b.status is BlockerStatus.ACTIVE or b.id in explicit_ids
        ]

        scored = []
        for blocker in candidates:
            score, reasons = score_blocker(
                blocker,
                explicit_hints=explicit_hints,
                linked_to_explicit_entity=_linked_to_explicit(blocker),
                weights=self.weights,
            )
            scored.append((score, reasons, blocker))

        selected = self._budget_select(
            scored,
            budget.blockers,
            key_fn=lambda b: (None, None, b.updated_at, b.id),
        )
        return (
            [b for _, _, b in selected],
            self._to_category_retrieval(len(all_blockers), selected),
        )

    # -- Resources ----------------------------------------------------------

    def _select_resources(
        self,
        request_context: RequestContext,
        explicit_project_ids: set[str],
        budget: ContextBudget,
    ) -> tuple[list[Resource], CategoryRetrieval]:
        all_resources = self.world_model.store.list_resources()
        explicit_ids = {
            r.id
            for r in all_resources
            if any_hint_matches(request_context.mentioned_resources, r.name)
        }

        def _linked_to_explicit(resource: Resource) -> bool:
            for link in self.world_model.store.list_resource_links_for_resource(resource.id):
                if link.entity_kind is LinkedEntityKind.PROJECT and link.entity_id in explicit_project_ids:
                    return True
            return False

        # No hard-filter/terminal-state exclusion for resources - the
        # spec's own hard-filter examples don't name resources, and an
        # UNAVAILABLE resource is still often relevant context (e.g.
        # "the printer is broken" - the World Model represents that as a
        # Blocker, but a caller may still want to see the Resource too).
        scored = []
        for resource in all_resources:
            score, reasons = score_resource(
                resource,
                explicit_hints=request_context.mentioned_resources,
                linked_to_explicit_entity=resource.id in explicit_ids
                or _linked_to_explicit(resource),
                weights=self.weights,
            )
            scored.append((score, reasons, resource))

        selected = self._budget_select(
            scored,
            budget.resources,
            key_fn=lambda r: (None, None, r.updated_at, r.id),
        )
        return (
            [r for _, _, r in selected],
            self._to_category_retrieval(len(all_resources), selected),
        )

    # -- Preferences ------------------------------------------------------

    def _select_preferences(
        self, budget: ContextBudget
    ) -> tuple[list[Preference], CategoryRetrieval]:
        all_preferences = self.world_model.store.list_preferences()
        scored = []
        for preference in all_preferences:
            score = preference.confidence * self.weights.CURRENT_STATE_ACTIVE
            reasons = ["confidence_weighted"]
            scored.append((score, reasons, preference))

        selected = self._budget_select(
            scored,
            budget.preferences,
            key_fn=lambda p: (None, None, p.updated_at, p.id),
        )
        return (
            [p for _, _, p in selected],
            self._to_category_retrieval(len(all_preferences), selected),
        )

    # -- Memory (separate retrieval source, never redesigned) ---------------

    def _select_memory(self, budget: ContextBudget) -> list[str]:
        facts = self.memory.get_facts() if self.memory else {}
        lines = [f"{key}: {value}" for key, value in sorted(facts.items())]
        return lines[: budget.memory]

    # -- Unconfirmed candidates (a hypothesis source, never a fact category) ------

    def _select_candidates(self) -> list:
        """The staged candidates confident enough to be worth showing, strongest
        first, bounded. They are returned exactly as staged (status
        "unconfirmed", their confidence and origin intact): nothing here writes,
        promotes or relabels one. A fault in the source means "none"."""
        from app.context_engine.base import CANDIDATE_CONTEXT_MIN_CONFIDENCE, MAX_CONTEXT_CANDIDATES

        try:
            found = list(self.candidate_source())
        except Exception:  # noqa: BLE001 - a missing hint is never worth failing a request
            return []
        eligible = [
            c
            for c in found
            if getattr(c, "status", None) == "unconfirmed" and c.confidence >= CANDIDATE_CONTEXT_MIN_CONFIDENCE
        ]
        eligible.sort(key=lambda c: (-c.confidence, c.updated_at, c.id))
        return eligible[:MAX_CONTEXT_CANDIDATES]

    # -- shared selection/scoring helpers -----------------------------------

    @staticmethod
    def _budget_select(
        scored: list[tuple[float, list[str], object]],
        limit: int,
        *,
        key_fn: Callable[[object], tuple],
    ) -> list[tuple[float, list[str], object]]:
        """Sort `scored` deterministically (highest score first, then the
        entity-specific tie-break key) and keep the top `limit`."""

        def sort_key(entry: tuple[float, list[str], object]) -> tuple:
            score, _reasons, item = entry
            priority, proximity_timestamp, updated_at, entity_id = key_fn(item)
            return (-score,) + tie_break_key(
                priority=priority,
                proximity_timestamp=proximity_timestamp,
                updated_at=updated_at,
                entity_id=entity_id,
            )

        ordered = sorted(scored, key=sort_key)
        return ordered[:limit]

    @staticmethod
    def _to_category_retrieval(
        considered: int, selected: list[tuple[float, list[str], object]]
    ) -> CategoryRetrieval:
        items = {
            getattr(item, "id"): ItemProvenance(relevance_score=score, reasons=tuple(reasons))
            for score, reasons, item in selected
        }
        return CategoryRetrieval(considered=considered, selected=len(selected), items=items)
