"""
The focus door: where it sits in the real JarvisCore, what may reach it, what it must never take from the other flows, and the
import / read-only boundaries of the package. Real JarvisCore, the production capability router and recording stand-ins for the model,
the tools and the other doors.
"""

import ast
import inspect
import pathlib
import sys
from types import SimpleNamespace

import pytest

from app.capability import Capability, DeterministicCapabilityRouter
from app.core import JarvisCore
from app.focus import FocusService
from app.handle_this.routing import ObjectiveAwareCapabilityRouter
from app.memory.sqlite_store import SQLiteMemoryStore
from app.providers.base import AIProvider
from app.search.router import is_explicit_search_request
from app.tasks.base import TaskPriority
from app.tools.base import PermissionLevel, Tool, ToolResult
from app.tools.registry import ToolRegistry
from test_morning_brief import TZ, BriefRig

ARDUINO = "I need to finish my Arduino project today."
ARDUINO_DIRECT = "I need to finish my Arduino project."
WHAT_NOW = "What should I work on right now?"


class ChatProvider(AIProvider):
    """The model: every call is recorded, so a message a door answers is shown never to have reached it."""

    def __init__(self):
        self.calls = []

    def generate_reply(self, message):
        self.calls.append(message)
        return "ordinary chat reply"


class RecordingRegistry(ToolRegistry):
    def __init__(self):
        super().__init__()
        self.executed = []

    def execute(self, name, params=None, confirmed=None):
        self.executed.append(name)
        return super().execute(name, params, confirmed)


class Trap(Tool):
    description = "must not run for a focus request"
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def execute(self, **kwargs):
        return ToolResult.success(message="ran")


class Door:
    """Stands where the focus service does (or where another door does): records what reaches it."""

    def __init__(self, reply=None):
        self.reply = reply
        self.messages = []

    def command_reply(self, message, conversation_id=None, device_context=None):
        self.messages.append(message)
        return self.reply

    def recall_reply(self, message, conversation_id=None):
        self.messages.append(message)
        return self.reply

    def observe(self, message, conversation_id=None):
        self.messages.append(("observe", message))
        return None

    def redact(self, message):
        return message


class Writer(Door):
    """A memory writer that claims nothing but records every message it is shown (command, recall and observe)."""

    def __init__(self):
        super().__init__(None)
        self.observed = []

    def observe(self, message, conversation_id=None):
        self.observed.append(message)
        return None


def build_core(**doors):
    registry = RecordingRegistry()
    for name in ("create_job", "list_tasks"):
        registry.register(Trap(name))
    provider = ChatProvider()
    core = JarvisCore(
        provider=provider,
        memory=SQLiteMemoryStore(":memory:"),
        tools=registry,
        capability_router=ObjectiveAwareCapabilityRouter(DeterministicCapabilityRouter()),
        **doors,
    )
    return core, provider, registry


def say(core, message):
    try:
        return core.handle_chat(message)[0]
    except Exception as exc:  # noqa: BLE001 - e.g. an explicit search with no search tool: the door must simply not have been involved
        return exc


def outcome(core, provider, registry, message):
    """Everything observable about how a message was handled: the reply (or the error), the model calls and the tools run."""
    result = say(core, message)
    shown = (type(result).__name__, str(result)) if isinstance(result, Exception) else result
    return shown, list(provider.calls), list(registry.executed)


# ================================================ what reaches the door ===============================================================


@pytest.mark.parametrize("message", [ARDUINO, ARDUINO_DIRECT, WHAT_NOW, "What should I work on?"])
def test_an_answered_focus_request_is_never_sent_to_the_model(message):
    focus = Door("FOCUS ANSWER")
    core, provider, registry = build_core(focus=focus)

    assert say(core, message) == "FOCUS ANSWER" and provider.calls == [] and focus.messages == [message]
    assert registry.executed == []


def test_a_message_the_door_does_not_claim_carries_on_to_the_model_untouched():
    focus = Door(None)
    core, provider, _ = build_core(focus=focus)

    assert say(core, "I need to finish my coffee.") == "ordinary chat reply"
    assert provider.calls == ["I need to finish my coffee."] and focus.messages == ["I need to finish my coffee."]


@pytest.mark.parametrize("message", ["I need to finish my coffee today.", ARDUINO, WHAT_NOW, ARDUINO_DIRECT])
def test_a_door_that_declines_or_is_absent_changes_nothing_for_any_message(message):
    """A SEARCH-routed message (one with "today" in it) goes to the existing search path with or without focus - that path is not
    this feature's business, and it must be exactly as it was."""
    assert outcome(*build_core(focus=Door(None)), message) == outcome(*build_core(), message)


@pytest.mark.parametrize("message", [ARDUINO, WHAT_NOW, ARDUINO_DIRECT])
def test_a_faulty_focus_service_is_not_a_focus_request(message):
    class Exploding:
        def command_reply(self, message, conversation_id=None):
            raise RuntimeError("boom")

    assert outcome(*build_core(focus=Exploding()), message) == outcome(*build_core(), message)


def test_the_exchange_is_stored_like_any_other_reply():
    core, _, _ = build_core(focus=Door("FOCUS ANSWER"))

    reply, conversation_id, *_ = core.handle_chat(ARDUINO)
    stored = core.memory.get_messages(conversation_id)

    assert reply == "FOCUS ANSWER" and [m["content"] if isinstance(m, dict) else m.content for m in stored][-2:] == [ARDUINO, "FOCUS ANSWER"]


# ============================================ what must never reach it ================================================================


NEVER_REACHES = [
    ("use tool list_tasks", Capability.TOOL),
    ("Create a task: finish Arduino project", Capability.TASK),
    ("what is the status of my task?", Capability.TASK_STATUS),
    ("yes", Capability.CONFIRMATION),
    ("approve the plan", Capability.SELF_DEV_APPROVAL),
    ("self-develop: add a dark mode toggle", Capability.SELF_DEVELOPMENT),
    ("Search for Arduino project ideas today", Capability.SEARCH),  # an EXPLICIT search
    ("Research three laptops under $900 for my robotics work, compare them, and save the findings", Capability.TASK),
]


@pytest.mark.parametrize("message,capability", NEVER_REACHES)
def test_tool_task_status_confirmation_self_development_and_explicit_search_never_reach_the_door(message, capability):
    focus = Door("FOCUS ANSWER")
    core, _, _ = build_core(focus=focus)
    assert core.capability_router.route(message).capability is capability  # the premise: this is how the real router classifies it

    result = say(core, message)

    assert focus.messages == [] and result != "FOCUS ANSWER"


def test_an_explicit_search_that_contains_every_trigger_word_is_still_a_search():
    message = "Search for Arduino project ideas to finish today"
    assert is_explicit_search_request(message)
    focus = Door("FOCUS ANSWER")
    core, _, _ = build_core(focus=focus)

    say(core, message)

    assert focus.messages == []


def test_a_tool_command_runs_its_tool_and_a_task_request_still_creates_its_job_not_a_focus_answer():
    focus = Door("FOCUS ANSWER")
    core, _, registry = build_core(focus=focus)

    say(core, "use tool list_tasks")
    say(core, "Create a task: finish Arduino project")

    assert registry.executed[0] == "list_tasks" and "create_job" in registry.executed
    assert focus.messages == []


def test_a_reply_to_a_waiting_interaction_is_answered_by_the_interaction_and_never_by_focus():
    class Resolver:
        def handle(self, message, conversation_id, **kwargs):
            return SimpleNamespace(handled=True, reply="INTERACTION ANSWER")

        def surface(self, conversation_id):
            return None

        def awaiting_reply(self, conversation_id):
            return False

    focus = Door("FOCUS ANSWER")
    core, provider, _ = build_core(focus=focus, interaction_resolver=Resolver())

    assert say(core, ARDUINO) == "INTERACTION ANSWER" and focus.messages == [] and provider.calls == []


def test_an_unhandled_interaction_resolver_leaves_the_door_available():
    class Resolver:
        def handle(self, message, conversation_id, **kwargs):
            return SimpleNamespace(handled=False, reply=None)

        def surface(self, conversation_id):
            return None

        def awaiting_reply(self, conversation_id):
            return False

    core, _, _ = build_core(focus=Door("FOCUS ANSWER"), interaction_resolver=Resolver())

    assert say(core, ARDUINO) == "FOCUS ANSWER"


# ============================================ every other door claims its own phrases first ===========================================


OTHER_DOORS = ["activity_context", "morning_brief", "device_operations", "reminder_operations", "scheduled_job_operations", "memory_writer"]


#: every door takes the plain (DIRECT) phrasing; all but the memory-command door also take the SEARCH-routed one (the recall test below covers that)
DOOR_CASES = [(name, ARDUINO_DIRECT) for name in OTHER_DOORS] + [(name, ARDUINO) for name in OTHER_DOORS if name != "memory_writer"]


@pytest.mark.parametrize("name,message", DOOR_CASES)
def test_a_message_another_door_claims_never_reaches_focus(name, message):
    other, focus = Door(f"CLAIMED BY {name}"), Door("FOCUS ANSWER")
    core, provider, _ = build_core(focus=focus, **{name: other})

    assert say(core, message) == f"CLAIMED BY {name}" and focus.messages == [] and provider.calls == []


def test_a_memory_recall_answer_also_comes_before_focus():
    class Recall(Door):
        def command_reply(self, message, conversation_id=None):
            return None

        def recall_reply(self, message, conversation_id=None):
            return "RECALL ANSWER"

    focus = Door("FOCUS ANSWER")
    core, _, _ = build_core(focus=focus, memory_writer=Recall(None))

    assert say(core, ARDUINO) == "RECALL ANSWER" and focus.messages == []


def test_focus_is_the_last_door_in_the_dispatch():
    source = inspect.getsource(JarvisCore._dispatch)
    focus_at = source.index("self.focus.command_reply")
    for earlier in (
        "self.activity_context.command_reply", "self.morning_brief.command_reply", "self.device_operations.command_reply",
        "self.reminder_operations.command_reply", "self.scheduled_job_operations.command_reply", "self.memory_writer.command_reply",
        "self.memory_writer.recall_reply",
    ):
        assert source.index(earlier) < focus_at, earlier
    assert source.index("self.focus.command_reply") < source.index("self.context_engine.build_snapshot")


def test_the_door_uses_exactly_the_conversation_path_gate_of_its_siblings():
    source = inspect.getsource(JarvisCore._dispatch)
    block = source[source.index("self.focus is not None") - 140 : source.index("self.focus.command_reply")]

    assert "memory_reply is None" in block and "interaction_reply is None" in block
    assert "decision.capability is Capability.DIRECT" in block
    assert "Capability.SEARCH and not is_explicit_search_request(message)" in block
    for forbidden in ("Capability.TASK", "Capability.TOOL", "Capability.CONFIRMATION", "SELF_DEV"):
        assert forbidden not in block


# ============================================ a claimed message does not feed memory; a declined one still does =========================


def test_a_message_focus_answers_is_not_also_observed_by_the_memory_writer():
    writer = Writer()
    core, _, _ = build_core(focus=Door("FOCUS ANSWER"), memory_writer=writer)

    say(core, ARDUINO_DIRECT)

    assert writer.observed == []  # nothing is staged from a message that was answered as a focus request


def test_a_message_focus_declines_is_still_observed_exactly_as_before():
    writer = Writer()
    core, _, _ = build_core(focus=Door(None), memory_writer=writer)

    say(core, "I need to finish my coffee.")

    assert writer.observed == ["I need to finish my coffee."]


# ============================================ real stores, real service, through the real core ========================================


def real_focus_core(tmp_path, **doors):
    rig = BriefRig(tmp_path)
    project = rig.wm.create_project("Arduino project")
    rig.add_task("Arduino: test the servo", TaskPriority.HIGH, "2026-09-29")
    rig.add_task("Buy milk", TaskPriority.HIGH, "2026-09-28")
    focus = FocusService(tasks=rig.tasks, calendar=rig.calendar, world_model=rig.wm, activity=rig.context, clock=lambda: rig.now.astimezone(TZ))
    core, provider, registry = build_core(focus=focus, **doors)
    return rig, core, provider, registry, project


def test_the_arduino_request_through_the_real_core_answers_from_the_stores_without_a_model_a_tool_or_a_job(tmp_path):
    rig, core, provider, registry, _ = real_focus_core(tmp_path)
    before = rig.dbs()

    reply = say(core, ARDUINO)

    assert reply.startswith('This sounds like your project "Arduino project". Start with: "Arduino: test the servo".')
    assert 'Not part of this, but overdue, high priority: "Buy milk".' in reply
    assert provider.calls == [] and registry.executed == [] and rig.jobs.list() == [] and rig.calendar.mutations == []
    assert rig.dbs() == before


@pytest.mark.parametrize("message", ["Let's do it.", "Okay, continue.", "yes", "go ahead", "do it"])
def test_a_follow_up_is_never_read_as_accepting_a_focus_suggestion(tmp_path, message):
    """Slice 1 is stateless: after a focus answer, "Let's do it." is just a message - it starts nothing and approves nothing, and is handled
    exactly as it would be had the focus answer never been given."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    rig, core, provider, registry, _ = real_focus_core(tmp_path / "a")
    say(core, ARDUINO)
    before, calendar_reads = rig.dbs(), len(rig.calendar.calls)
    after_focus = outcome(core, provider, registry, message)
    after_focus = (after_focus[0], after_focus[1][-1:], after_focus[2])  # the model calls / tools this message caused

    _, fresh_core, fresh_provider, fresh_registry, _ = real_focus_core(tmp_path / "b")
    fresh = outcome(fresh_core, fresh_provider, fresh_registry, message)

    assert after_focus == fresh
    assert "create_job" not in after_focus[2] and rig.jobs.list() == []
    assert rig.dbs() == before and len(rig.calendar.calls) == calendar_reads  # nothing written, nothing read


@pytest.mark.parametrize(
    "message",
    [
        "My Arduino project is annoying.", "My Arduino project is annoying today.", "Can you explain what a project deadline means?",
        "I finished my Arduino project today.", "How do I finish a project faster?", "The project deadline is today.",
        "I need to finish my coffee today.", "What is a project?", "I need to finish my Arduino project today and then email Sam.",
    ],
)
def test_ordinary_sentences_with_the_trigger_words_are_handled_exactly_as_without_focus(tmp_path, message):
    rig, core, provider, registry, _ = real_focus_core(tmp_path)

    assert outcome(core, provider, registry, message) == outcome(*build_core(), message)  # not claimed: what happened before still happens
    assert rig.calendar.calls == [] and rig.jobs.list() == []  # and focus read nothing for it


def test_a_focus_question_with_an_empty_world_asks_instead_of_inventing(tmp_path):
    rig = BriefRig(tmp_path)
    focus = FocusService(tasks=rig.tasks, calendar=rig.calendar, world_model=rig.wm, activity=rig.context, clock=lambda: rig.now.astimezone(TZ))
    core, provider, _ = build_core(focus=focus)

    reply = say(core, WHAT_NOW)

    assert reply.startswith("I don't have any open tasks to rank.") and provider.calls == []


# ============================================ the package boundary ====================================================================

FOCUS_DIR = pathlib.Path(__file__).resolve().parent.parent / "app" / "focus"
ALLOWED_APP_IMPORTS = {
    "app.focus", "app.focus.base", "app.focus.recognition", "app.focus.render", "app.focus.service",
    "app.context_engine.scoring", "app.morning_brief.gather", "app.tasks.base",
}


def _modules(path):
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            found.append(node.module or "")
    return found


def test_the_package_imports_only_the_agreed_existing_helpers_and_the_standard_library():
    files = sorted(FOCUS_DIR.glob("*.py"))
    assert {f.name for f in files} == {"__init__.py", "base.py", "recognition.py", "render.py", "service.py"}
    for path in files:
        for module in _modules(path):
            top = module.split(".")[0]
            if top == "app":
                assert module in ALLOWED_APP_IMPORTS, f"{path.name} imports {module}"
            else:
                assert top in sys.stdlib_module_names or top == "__future__", f"{path.name} imports {module}"


def test_the_package_cannot_start_work_reach_the_network_or_touch_the_disk():
    banned = {"threading", "asyncio", "subprocess", "socket", "sqlite3", "os", "pathlib", "shutil", "urllib", "http", "requests", "time", "random", "sched"}
    for path in FOCUS_DIR.glob("*.py"):
        assert not banned & {m.split(".")[0] for m in _modules(path)}, path.name
        calls = {n.func.id for n in ast.walk(ast.parse(path.read_text(encoding="utf-8"))) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert not calls & {"open", "exec", "eval", "compile", "__import__", "input", "print"}, path.name


def test_the_package_only_reads_the_sources_it_is_given():
    """Every method it calls on an injected source is a read; no write verb appears anywhere in the package."""
    writes = {
        "create", "complete", "cancel", "update", "delete", "save", "set", "submit", "execute", "run", "send", "notify", "write", "add",
        "remove", "insert", "commit", "transition", "link", "create_event", "update_event", "delete_event", "put", "post", "start", "launch",
        "create_project", "create_goal", "mark_surfaced", "clear", "release", "register",
    }
    called = set()
    for path in FOCUS_DIR.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
    assert not called & writes, called & writes
    assert {"list", "get_active_projects", "get_active_goals", "current"} <= called  # ...and these are the reads


def test_the_package_never_reads_a_calendar_description():
    assert "description" not in (FOCUS_DIR / "service.py").read_text(encoding="utf-8").replace("`description`", "")


def test_the_package_has_no_model_or_provider_in_it():
    for path in FOCUS_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for word in ("generate_reply", "provider.", "llm", "gemini", "openai", "anthropic"):
            assert word not in text.replace("calendar provider", "").replace("calendar_provider", ""), (path.name, word)


# ============================================ the real application ====================================================================


def test_the_real_application_wires_focus_read_only_and_adds_no_tool():
    import app.main as live

    focus = live.core.focus
    assert isinstance(focus, FocusService)
    assert focus.tasks is live._task_store and focus.calendar is live._calendar_provider
    assert focus.world_model is live._world_model_service and focus.activity is live._activity_context
    assert not [name for name in getattr(live.core.tools, "_tools", {}) if "focus" in name]
    assert focus.command_reply("what is the capital of France?") is None
    assert focus.command_reply("My Arduino project is annoying.") is None
