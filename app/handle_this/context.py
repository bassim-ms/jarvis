"""
The context handoff: asking the EXISTING Context Engine
(app/context_engine/) for what is relevant to an objective, in the shape a
Planner will later want - without becoming a second memory or context system.

- Selective. The categories requested depend on the objective's families
  (`FAMILY_CONTEXT_CATEGORIES`): research and analysis want projects, goals,
  resources and preferences; organization wants projects, goals, tasks,
  commitments and events; and so on. Never the whole World Model, and never
  the opt-in MEMORY category. Whatever an engine returns for a category that
  was not requested is dropped here, so a misbehaving engine cannot widen the
  handoff.
- Budgeted. The Context Engine's own ContextBudget (per-category item limits)
  applies, and this adapter re-applies it to what it serializes.
- Read-only. Nothing is written to the World Model or to memory: the
  interpreted goal is not stored as a fact, and nothing is learned from the
  request.
- Honest about failure. If the engine cannot be queried this raises
  ContextUnavailableError - the caller must NOT carry on as if the context had
  been consulted. (A deployment with no Context Engine at all is a different,
  legitimate situation: understanding then rests on the request alone and the
  handoff says so with ContextStatus.NOT_CONFIGURED.)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.context_engine.base import (
    DEFAULT_CONTEXT_BUDGET,
    ContextBudget,
    RequestContext,
    RequiredContextCategory as Category,
)
from app.context_engine.engine import ContextEngine
from app.context_engine.request_builder import build_request_context
from app.context_engine.scoring import REASON_EXPLICIT_MATCH
from app.handle_this.base import (
    SUPPORTED_FAMILIES,
    ContextHandoff,
    ContextStatus,
    Constraint,
    ConstraintSource,
    GoalFamily,
    Topic,
    clean_line,
)
from app.world_model.base import ContextSnapshot

_ITEM_CHARS = 120
_MAX_KNOWN_CONSTRAINTS = 4


class ContextUnavailableError(RuntimeError):
    """The Context Engine could not be queried."""


#: Which World Model categories each family of objective can use. Fixed,
#: application-controlled, and deliberately short of "everything".
FAMILY_CONTEXT_CATEGORIES: dict[GoalFamily, frozenset[Category]] = {
    GoalFamily.RESEARCH: frozenset(
        {Category.PROJECTS, Category.GOALS, Category.RESOURCES, Category.PREFERENCES}
    ),
    GoalFamily.ANALYSIS: frozenset(
        {Category.PROJECTS, Category.GOALS, Category.RESOURCES, Category.PREFERENCES, Category.BLOCKERS}
    ),
    GoalFamily.ORGANIZATION: frozenset(
        {Category.PROJECTS, Category.GOALS, Category.TASKS, Category.COMMITMENTS, Category.EVENTS}
    ),
    GoalFamily.COMPUTER_WORKFLOW: frozenset(
        {Category.PROJECTS, Category.RESOURCES, Category.PREFERENCES}
    ),
}


def categories_for(families: Sequence[GoalFamily]) -> frozenset[Category]:
    """The union of what the given (supported) families can use. Anything that
    is not a supported family contributes nothing."""
    needed: set[Category] = set()
    for family in families:
        if family in SUPPORTED_FAMILIES:
            needed |= FAMILY_CONTEXT_CATEGORIES[family]
    return frozenset(needed)


class ContextSource(ABC):
    """Supplies the ContextHandoff for an objective."""

    @abstractmethod
    def retrieve(self, message: str, families: Sequence[GoalFamily]) -> ContextHandoff:
        """The relevant context, or ContextUnavailableError."""


#: (snapshot attribute, category, display label, item -> short text). Mirrors
#: the table JarvisCore uses to hand a snapshot to a Job's planner (a test
#: pins the two together), so the same category reads the same way everywhere.
_SNAPSHOT_FIELDS = (
    ("active_projects", Category.PROJECTS, "Active projects", lambda p: p.name),
    ("relevant_goals", Category.GOALS, "Relevant goals", lambda g: g.title),
    ("relevant_tasks", Category.TASKS, "Relevant tasks", lambda t: f"{t.ref_type.value}:{t.ref_id}"),
    ("upcoming_commitments", Category.COMMITMENTS, "Upcoming commitments", lambda c: c.title),
    ("upcoming_events", Category.EVENTS, "Upcoming events", lambda e: e.title),
    ("active_blockers", Category.BLOCKERS, "Active blockers", lambda b: b.title),
    ("relevant_resources", Category.RESOURCES, "Relevant resources", lambda r: r.name),
    ("relevant_preferences", Category.PREFERENCES, "Relevant preferences", lambda p: f"{p.key}={p.value}"),
    ("relevant_memory", Category.MEMORY, "Relevant memory", lambda m: m),
)


def _known_constraints(snapshot: ContextSnapshot, requested: frozenset[Category]) -> tuple[Constraint, ...]:
    """What the (already selective) snapshot settles: a stored budget, and the
    project the request explicitly names as what it is for. Read from the
    snapshot only."""
    known: list[Constraint] = []

    def add(topic: Topic, text: str) -> None:
        text = clean_line(text, _ITEM_CHARS)
        if text and len(known) < _MAX_KNOWN_CONSTRAINTS:
            known.append(Constraint(topic, text, ConstraintSource.CONTEXT))

    if Category.PREFERENCES in requested:
        for preference in snapshot.relevant_preferences:
            if "budget" in preference.key.lower():
                add(Topic.BUDGET, f"{preference.key}: {preference.value}")
    if Category.GOALS in requested:
        for goal in snapshot.relevant_goals:
            if "budget" in goal.title.lower():
                add(Topic.BUDGET, goal.title)
    if Category.RESOURCES in requested:
        for resource in snapshot.relevant_resources:
            if "budget" in resource.name.lower():
                add(Topic.BUDGET, resource.name)
    if Category.PROJECTS in requested:
        provenance = snapshot.retrieval_metadata.get("projects")
        for project in snapshot.active_projects:
            item = provenance.items.get(project.id) if provenance is not None else None
            if item is not None and REASON_EXPLICIT_MATCH in item.reasons:
                add(Topic.USE_CASE, project.name)
    return tuple(known)


class ContextEngineSource(ContextSource):
    """ContextSource over the existing ContextEngine. Builds the same
    deterministic RequestContext the chat path builds (mentions, time scope),
    narrows `required_context` to what the objective's families can use, and
    applies `budget`."""

    def __init__(self, engine: ContextEngine, budget: ContextBudget = DEFAULT_CONTEXT_BUDGET):
        self.engine = engine
        self.budget = budget

    def retrieve(self, message: str, families: Sequence[GoalFamily]) -> ContextHandoff:
        requested = categories_for(families)
        base = build_request_context(message)
        request_context = RequestContext(
            raw_request=message,
            context_intent=base.context_intent,
            mentioned_projects=list(base.mentioned_projects),
            mentioned_goals=list(base.mentioned_goals),
            mentioned_tasks=list(base.mentioned_tasks),
            mentioned_commitments=list(base.mentioned_commitments),
            mentioned_resources=list(base.mentioned_resources),
            time_scope=base.time_scope,
            required_context=requested,
            context_budget=self.budget,
        )
        try:
            snapshot = self.engine.build_snapshot(request_context)
        except Exception as exc:  # noqa: BLE001 - any failure means "not consulted"
            raise ContextUnavailableError("the Context Engine could not be queried") from exc

        categories: dict[str, tuple[str, ...]] = {}
        for attribute, category, label, describe in _SNAPSHOT_FIELDS:
            if category not in requested:
                continue  # never serialize what was not asked for
            items = list(getattr(snapshot, attribute))[: self.budget.for_category(category)]
            described = tuple(clean_line(describe(item), _ITEM_CHARS) for item in items)
            if described:
                categories[label] = described
        return ContextHandoff(
            status=ContextStatus.RETRIEVED,
            requested_categories=tuple(sorted(c.value for c in requested)),
            categories=categories,
            known_constraints=_known_constraints(snapshot, requested),
        )
