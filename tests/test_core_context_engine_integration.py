"""
Tests for JarvisCore's M16.2 integration point (app/core.py):
`context_engine` is an optional, additive constructor parameter, and
`handle_chat_with_context` is the one new method that exposes the
ContextSnapshot `_dispatch` builds - `handle_chat`/`handle_chat_with_job_id`
must remain byte-for-byte unaffected, whether or not a context_engine is
configured (see tests/test_core.py/tests/test_chat.py's own full suites,
which already prove this indirectly by continuing to pass unchanged).
"""

from app.core import JarvisCore
from app.context_engine.engine import ContextEngine
from app.providers.base import AIProvider
from app.world_model.base import (
    Blocker,
    BlockerSeverity,
    BlockerStatus,
    Commitment,
    CommitmentStatus,
    ContextSnapshot,
    Event,
    EntitySource,
    Goal,
    GoalStatus,
    Preference,
    Priority,
    Project,
    ProjectStatus,
    Resource,
    ResourceStatus,
    ResourceType,
    TaskLink,
    TaskRefType,
)
from app.world_model.service import WorldModelService
from app.world_model.sqlite_store import SQLiteWorldModelStore


class _FakeProvider(AIProvider):
    def __init__(self, reply: str = "fake reply"):
        self.reply = reply
        self.calls = 0

    def generate_reply(self, prompt: str) -> str:
        self.calls += 1
        return self.reply


def make_core(context_engine=None) -> JarvisCore:
    return JarvisCore(provider=_FakeProvider(), context_engine=context_engine)


def make_context_engine(service: WorldModelService | None = None) -> ContextEngine:
    service = service or WorldModelService(SQLiteWorldModelStore(":memory:"))
    return ContextEngine(service)


def test_context_engine_defaults_to_none():
    core = make_core()
    assert core.context_engine is None


def test_handle_chat_with_context_returns_none_snapshot_when_no_context_engine_configured():
    core = make_core()
    reply, conversation_id, sources, job_id, snapshot = core.handle_chat_with_context("hello")
    assert snapshot is None
    assert reply == "fake reply"


def test_handle_chat_with_context_returns_a_real_snapshot_when_configured():
    service = WorldModelService(SQLiteWorldModelStore(":memory:"))
    project = service.create_project("Robotics arm")
    core = make_core(context_engine=make_context_engine(service))

    reply, conversation_id, sources, job_id, snapshot = core.handle_chat_with_context(
        "hello there"
    )

    assert snapshot is not None
    assert project.id in {p.id for p in snapshot.active_projects}


def test_handle_chat_is_unaffected_by_a_configured_context_engine():
    """The existing 3-tuple contract - reply/conversation_id/sources -
    must be identical whether or not a context_engine is configured."""
    core_without = make_core()
    core_with = make_core(context_engine=make_context_engine())

    reply_a, conv_a, sources_a = core_without.handle_chat("hello")
    reply_b, conv_b, sources_b = core_with.handle_chat("hello")

    assert reply_a == reply_b == "fake reply"
    assert sources_a == sources_b == []


def test_handle_chat_with_job_id_is_unaffected_by_a_configured_context_engine():
    core_without = make_core()
    core_with = make_core(context_engine=make_context_engine())

    result_without = core_without.handle_chat_with_job_id("hello")
    result_with = core_with.handle_chat_with_job_id("hello")

    assert len(result_without) == len(result_with) == 4
    assert result_without[0] == result_with[0] == "fake reply"
    assert result_without[3] is None
    assert result_with[3] is None


def test_context_engine_never_mutates_the_world_model_during_a_chat_turn():
    service = WorldModelService(SQLiteWorldModelStore(":memory:"))
    project = service.create_project("Robotics arm")
    before = service.get_project(project.id)
    core = make_core(context_engine=make_context_engine(service))

    core.handle_chat_with_context("tell me about my robotics arm project")

    after = service.get_project(project.id)
    assert after == before


def test_context_engine_reuses_the_same_capability_decision_no_second_classifier(monkeypatch):
    """build_request_context must be called with the SAME Capability
    JarvisCore's own capability_router already decided - never a second,
    independent classification pass."""
    import app.core as core_module
    from app.capability import Capability

    captured = {}
    original = core_module.build_request_context

    def spy(message, *, capability=None, now=None):
        captured["capability"] = capability
        return original(message, capability=capability, now=now)

    monkeypatch.setattr(core_module, "build_request_context", spy)

    core = make_core(context_engine=make_context_engine())
    core.handle_chat_with_context("hello there")

    assert captured["capability"] is Capability.DIRECT


# ---------------------------------------------------------------------------
# _format_world_model_context - pure serializer, all 9 categories
# ---------------------------------------------------------------------------
#
# These test JarvisCore._format_world_model_context directly against a
# manually-constructed ContextSnapshot - the right level for proving pure
# serialization behavior (no re-ranking/filtering/truncation of its own),
# as opposed to tests/test_chat_context_engine_integration.py, which
# proves the same formatter is actually reached through the real /chat
# path. Category-level budget-respecting behavior (C) is proven at BOTH
# levels: directly here (with a snapshot whose list already exceeds 5
# items, the old hardcoded cap), and end-to-end there (blockers, budget=8).

_NOW = "2026-01-01T00:00:00+00:00"


def _project(name: str) -> Project:
    return Project(
        id=f"project-{name}",
        name=name,
        description="",
        status=ProjectStatus.ACTIVE,
        priority=Priority.MEDIUM,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _goal(title: str) -> Goal:
    return Goal(
        id=f"goal-{title}",
        project_id="project-x",
        title=title,
        description="",
        status=GoalStatus.ACTIVE,
        priority=Priority.MEDIUM,
        target_date=None,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _task_link(ref_id: str) -> TaskLink:
    return TaskLink(
        id=f"link-{ref_id}",
        ref_type=TaskRefType.JOB,
        ref_id=ref_id,
        created_at=_NOW,
        project_id="project-x",
    )


def _commitment(title: str) -> Commitment:
    return Commitment(
        id=f"commitment-{title}",
        title=title,
        description="",
        status=CommitmentStatus.OPEN,
        priority=Priority.MEDIUM,
        due_at=None,
        source=EntitySource.USER,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _event(title: str) -> Event:
    return Event(
        id=f"event-{title}",
        title=title,
        description="",
        starts_at=_NOW,
        ends_at=_NOW,
        location="",
        source=EntitySource.USER,
        external_id=None,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _blocker(title: str) -> Blocker:
    return Blocker(
        id=f"blocker-{title}",
        title=title,
        description="",
        status=BlockerStatus.ACTIVE,
        severity=BlockerSeverity.MEDIUM,
        detected_at=_NOW,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _resource(name: str) -> Resource:
    return Resource(
        id=f"resource-{name}",
        name=name,
        type=ResourceType.DEVICE,
        status=ResourceStatus.AVAILABLE,
        description="",
        created_at=_NOW,
        updated_at=_NOW,
    )


def _preference(key: str, value: str) -> Preference:
    return Preference(
        id=f"preference-{key}",
        key=key,
        value=value,
        category="general",
        source=EntitySource.USER,
        confidence=1.0,
        updated_at=_NOW,
    )


def _full_snapshot() -> ContextSnapshot:
    """A ContextSnapshot with all nine categories populated - the
    manually-built equivalent of ContextEngine deciding every category
    was relevant."""
    return ContextSnapshot(
        current_time=_NOW,
        active_projects=[_project("Robotics Arm")],
        relevant_goals=[_goal("Print gripper")],
        relevant_tasks=[_task_link("job-123")],
        upcoming_commitments=[_commitment("Submit assignment")],
        upcoming_events=[_event("Robotics meeting")],
        active_blockers=[_blocker("Printer nozzle isnt heating")],
        relevant_resources=[_resource("Soldering iron")],
        relevant_preferences=[_preference("response_style", "concise")],
        relevant_memory=["favorite_color: blue"],
    )


# -- A: each of the 9 categories reaches the prompt when populated --------


def test_format_world_model_context_serializes_all_nine_categories_when_populated():
    text = JarvisCore._format_world_model_context(_full_snapshot())

    assert "Robotics Arm" in text  # active_projects
    assert "Print gripper" in text  # relevant_goals
    assert "job:job-123" in text  # relevant_tasks (opaque ref_type:ref_id)
    assert "Submit assignment" in text  # upcoming_commitments
    assert "Robotics meeting" in text  # upcoming_events
    assert "Printer nozzle isnt heating" in text  # active_blockers
    assert "Soldering iron" in text  # relevant_resources
    assert "response_style=concise" in text  # relevant_preferences
    assert "favorite_color: blue" in text  # relevant_memory


def test_format_world_model_context_serializes_resources_alone():
    snapshot = ContextSnapshot(
        current_time=_NOW, relevant_resources=[_resource("3D printer")]
    )
    text = JarvisCore._format_world_model_context(snapshot)
    assert "Relevant resources: 3D printer" in text


def test_format_world_model_context_serializes_preferences_alone():
    snapshot = ContextSnapshot(
        current_time=_NOW, relevant_preferences=[_preference("unit_system", "metric")]
    )
    text = JarvisCore._format_world_model_context(snapshot)
    assert "Relevant preferences: unit_system=metric" in text


def test_format_world_model_context_serializes_memory_alone():
    snapshot = ContextSnapshot(current_time=_NOW, relevant_memory=["timezone: UTC+4"])
    text = JarvisCore._format_world_model_context(snapshot)
    assert "Relevant memory: timezone: UTC+4" in text


def test_format_world_model_context_serializes_tasks_alone():
    snapshot = ContextSnapshot(current_time=_NOW, relevant_tasks=[_task_link("abc")])
    text = JarvisCore._format_world_model_context(snapshot)
    assert "Relevant tasks: job:abc" in text


# -- B: empty categories are omitted --------------------------------------


def test_format_world_model_context_omits_empty_categories():
    snapshot = ContextSnapshot(current_time=_NOW, active_projects=[_project("Only project")])
    text = JarvisCore._format_world_model_context(snapshot)

    assert "Active projects" in text
    for label in (
        "Relevant goals",
        "Relevant tasks",
        "Upcoming commitments",
        "Upcoming events",
        "Active blockers",
        "Relevant resources",
        "Relevant preferences",
        "Relevant memory",
    ):
        assert label not in text


def test_format_world_model_context_returns_empty_string_for_a_fully_empty_snapshot():
    snapshot = ContextSnapshot(current_time=_NOW)
    assert JarvisCore._format_world_model_context(snapshot) == ""


def test_format_world_model_context_returns_empty_string_for_none():
    assert JarvisCore._format_world_model_context(None) == ""


# -- C: ContextEngine's per-category budget is respected, not re-limited --


def test_format_world_model_context_serializes_every_item_the_snapshot_holds():
    """A category with MORE than the old hardcoded cap (5) worth of
    items - proving the formatter imposes no limit of its own and shows
    exactly what ContextEngine already decided to include."""
    titles = [f"goal {i}" for i in range(9)]
    snapshot = ContextSnapshot(current_time=_NOW, relevant_goals=[_goal(t) for t in titles])

    text = JarvisCore._format_world_model_context(snapshot)

    for title in titles:
        assert title in text


# -- D: the formatter never imposes its own additional item/category limit --


def test_format_world_model_context_preserves_the_snapshots_own_item_order():
    """No re-ranking: items must appear in exactly the order the snapshot
    already holds them in (ContextEngine's own relevance order), never
    re-sorted by the formatter."""
    snapshot = ContextSnapshot(
        current_time=_NOW,
        relevant_goals=[_goal("zzz last but most relevant"), _goal("aaa first alphabetically")],
    )
    text = JarvisCore._format_world_model_context(snapshot)

    goals_line = next(line for line in text.splitlines() if line.startswith("- Relevant goals"))
    assert goals_line.index("zzz last but most relevant") < goals_line.index(
        "aaa first alphabetically"
    )


def test_format_world_model_context_has_no_hardcoded_category_whitelist():
    """All nine ContextSnapshot categories are in the formatter's own
    category table - not a smaller, hardcoded subset."""
    field_names = {name for name, _label, _describe in JarvisCore._WORLD_MODEL_CATEGORIES}
    assert field_names == {
        "active_projects",
        "relevant_goals",
        "relevant_tasks",
        "upcoming_commitments",
        "upcoming_events",
        "active_blockers",
        "relevant_resources",
        "relevant_preferences",
        "relevant_memory",
    }


# -- E: the original user message remains the final user-message section --


def test_build_prompt_still_ends_with_the_unaltered_user_message():
    prompt = JarvisCore._build_prompt(
        "What is blocking my robotics arm?", [], {}, _full_snapshot()
    )
    assert prompt.endswith("User: What is blocking my robotics arm?")


def test_build_prompt_world_model_section_precedes_the_user_message():
    prompt = JarvisCore._build_prompt("hello there", [], {}, _full_snapshot())
    world_model_index = prompt.index("What Jarvis currently knows")
    user_index = prompt.index("User: hello there")
    assert world_model_index < user_index
