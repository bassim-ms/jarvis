"""
Tests for deterministic step references (app/orchestration/references.py):
one PlanStep argument taking its value from an earlier step's stored result.

Three layers, each tested on its own:
  1. the reference module (syntax, path walking, bounding),
  2. plan validation (`validate_plan`: dependency rule, deferred type check),
  3. execution in JobEngine (resolution at run time, post-resolution
     validation, failure semantics, persistence across a restart).

Nothing here involves an AI: a reference that can't be resolved fails its
step, it is never guessed.
"""

import json

import pytest

from app.orchestration.base import (
    Job,
    JobStatus,
    Plan,
    PlanStep,
    RetryPolicy,
    StepStatus,
)
from app.orchestration.engine import JobEngine
from app.orchestration.planner import Planner, PlanningError
from app.orchestration.references import (
    MAX_REFERENCE_CHARS,
    ReferenceError,
    bound_value,
    looks_like_reference,
    resolve_path,
    resolve_reference,
    validate_reference,
)
from app.orchestration.sqlite_store import SQLiteJobStore
from app.orchestration.tools import ConfirmJobTool
from app.orchestration.validator import PlanValidationError, validate_plan
from app.tools.base import PermissionLevel, Tool, ToolArgumentError, ToolParameter, ToolResult
from app.tools.registry import ToolRegistry

RESULT = {
    "status": "success",
    "message": "found 2",
    "data": {
        "query": "acme",
        "results": [
            {"title": "Acme Inc", "url": "https://acme.example", "score": 7},
            {"title": "Acme Wiki", "url": "https://wiki.example", "score": 3},
        ],
        "empty": None,
    },
}


def ref(from_step="s1", path="data.results.0.title"):
    return {"from_step": from_step, "path": path}


# --- 1. the reference module ------------------------------------------------------------


def test_validate_reference_accepts_a_well_formed_reference():
    assert validate_reference(ref()) == ("s1", ("data", "results", "0", "title"))
    assert validate_reference(ref(path="message")) == ("s1", ("message",))
    assert validate_reference(ref("step-2_b", "status")) == ("step-2_b", ("status",))


@pytest.mark.parametrize(
    "bad",
    [
        {"from_step": "s1"},  # missing path
        {"path": "data"},  # missing from_step
        {"from_step": "s1", "path": "data", "extra": 1},  # padded
        {"from_step": 1, "path": "data"},
        {"from_step": "s 1", "path": "data"},
        {"from_step": "", "path": "data"},
        {"from_step": "s1", "path": ""},
        {"from_step": "s1", "path": None},
        {"from_step": "s1", "path": "data." + "a" * 300},
        {"from_step": "s1", "path": ".".join(["data"] + ["a"] * 8)},  # 9 segments
        {"from_step": "s1", "path": "data..title"},
        {"from_step": "s1", "path": "data.results[0]"},
        {"from_step": "s1", "path": "data.results.*"},
        {"from_step": "s1", "path": "data.a b"},
        {"from_step": "s1", "path": "data.__class__.x/y"},
        {"from_step": "s1", "path": "results.0.title"},  # not a result field
        {"from_step": "s1", "path": "__class__"},
        "s1.data",
        ["s1", "data"],
    ],
)
def test_validate_reference_rejects_anything_else(bad):
    with pytest.raises(ReferenceError):
        validate_reference(bad)


def test_looks_like_reference_flags_anything_using_from_step():
    assert looks_like_reference({"from_step": "s1"})  # even a malformed one
    assert not looks_like_reference({"url": "x"})
    assert not looks_like_reference("from_step")
    assert not looks_like_reference(None)


def test_resolve_walks_dicts_lists_and_scalars_exactly():
    assert resolve_reference(ref(), RESULT) == "Acme Inc"
    assert resolve_reference(ref(path="data.results.1.score"), RESULT) == 3
    assert resolve_reference(ref(path="message"), RESULT) == "found 2"
    assert resolve_reference(ref(path="status"), RESULT) == "success"
    assert resolve_reference(ref(path="data.results.0"), RESULT) == RESULT["data"]["results"][0]
    assert resolve_reference(ref(path="data.empty"), RESULT) is None  # a real null


@pytest.mark.parametrize(
    "path,fragment",
    [
        ("data.missing", "no key 'missing'"),
        ("data.results.2.title", "out of range"),
        ("data.results.first", "not a list index"),
        ("data.results.-1", "not a list index"),
        ("data.query.length", "cannot step into a str"),
        ("data.empty.x", "cannot step into a NoneType"),
        ("data.results.0.title.x", "cannot step into a str"),
        ("message.x", "cannot step into a str"),
    ],
)
def test_an_unresolvable_path_is_an_error_never_a_guess(path, fragment):
    with pytest.raises(ReferenceError, match=fragment):
        resolve_reference(ref(path=path), RESULT)


@pytest.mark.parametrize("missing", [None, {}])
def test_a_missing_source_result_is_an_error(missing):
    with pytest.raises(ReferenceError, match="no stored result"):
        resolve_reference(ref(), missing)


def test_resolve_path_never_reaches_object_internals():
    """A path is looked up as a dict KEY or list INDEX only - never as an
    attribute - so it can't walk into Python internals."""

    class Holder:
        secret = "x"

    with pytest.raises(ReferenceError):
        resolve_path({"data": Holder()}, ("data", "secret"), "data.secret")


def test_bound_value_returns_a_value_that_fits_completely_unchanged():
    for value in ("short", 42, 3.5, True, None, [1, 2, 3], {"a": "b"}):
        assert bound_value(value) == value
    original = {"results": [{"title": "t", "snippet": "s" * 100}]}
    assert bound_value(original) is original  # the very same object: nothing rebuilt


def test_a_value_exactly_at_the_limit_is_accepted():
    assert bound_value("x" * MAX_REFERENCE_CHARS) == "x" * MAX_REFERENCE_CHARS
    just_fits = {"k": "v" * (MAX_REFERENCE_CHARS - len(json.dumps({"k": ""})))}
    assert len(json.dumps(just_fits)) == MAX_REFERENCE_CHARS
    assert bound_value(just_fits) == just_fits


def test_an_oversized_string_fails_with_a_clear_size_error_and_is_not_truncated():
    with pytest.raises(ReferenceError) as exc_info:
        bound_value("A" * (MAX_REFERENCE_CHARS + 1))

    message = str(exc_info.value)
    assert "too large" in message
    assert f"{MAX_REFERENCE_CHARS + 1} characters" in message and f"limit is {MAX_REFERENCE_CHARS}" in message
    assert "not truncated" in message


def test_an_oversized_container_fails_and_is_not_pruned():
    big = {"results": [{"title": f"t{i}", "snippet": "S" * 400} for i in range(60)]}
    assert len(json.dumps(big)) > MAX_REFERENCE_CHARS

    with pytest.raises(ReferenceError, match="too large"):
        bound_value(big)
    with pytest.raises(ReferenceError, match="too large"):
        bound_value(big["results"])  # a list too


def test_resolve_reference_applies_the_same_limit():
    result = {"status": "success", "message": "M" * (MAX_REFERENCE_CHARS + 1), "data": None}

    with pytest.raises(ReferenceError, match="too large"):
        resolve_reference(ref(path="message"), result)


def test_the_limit_is_one_module_constant():
    import app.orchestration.references as references

    assert references.MAX_REFERENCE_CHARS == 12000
    # nothing else in the module hardcodes a competing size
    source = open(references.__file__, encoding="utf-8").read()
    assert source.count("12000") == 1


# --- 2. plan validation --------------------------------------------------------------------


class ProducerTool(Tool):
    name = "produce"
    description = "produces data"
    permission = PermissionLevel.READ

    def execute(self) -> ToolResult:
        return ToolResult.success(message="produced", data=RESULT["data"])


class ConsumerTool(Tool):
    """Takes a string and an optional number; records exactly what it got."""

    name = "consume"
    description = "consumes"
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self):
        self.calls: list[dict] = []

    @property
    def parameters(self):
        return [
            ToolParameter(name="text", type="string"),
            ToolParameter(name="count", type="number", required=False, default=0),
        ]

    def validate_arguments(self, text: str, count: float = 0) -> None:
        if text == "forbidden":
            raise ToolArgumentError("that text is not allowed")

    def execute(self, text: str, count: float = 0) -> ToolResult:
        self.calls.append({"text": text, "count": count})
        return ToolResult.success(message=f"consumed {text}")


class ObjectConsumerTool(Tool):
    name = "consume_object"
    description = "consumes an object"
    permission = PermissionLevel.SAFE_ACTION

    def __init__(self):
        self.calls: list[dict] = []

    @property
    def parameters(self):
        return [ToolParameter(name="payload", type="object")]

    def execute(self, payload: dict) -> ToolResult:
        self.calls.append(payload)
        return ToolResult.success(message="ok")


def registry_with(*tools):
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def raw(step_id, tool, arguments=None, depends_on=None, **extra):
    return {
        "id": step_id,
        "tool": tool,
        "arguments": arguments or {},
        "depends_on": depends_on or [],
        **extra,
    }


def validate(steps, registry, **kwargs):
    return validate_plan(
        steps, registry, max_steps=10, default_retry=RetryPolicy(max_attempts=1), **kwargs
    )


def test_a_reference_to_a_declared_dependency_validates():
    registry = registry_with(ProducerTool(), ConsumerTool())

    steps = validate(
        [
            raw("s1", "produce"),
            raw("s2", "consume", {"text": ref()}, depends_on=["s1"]),
        ],
        registry,
    )

    assert steps[1].arguments == {"text": ref()}  # stored as written, resolved later


def test_a_reference_can_supply_a_required_parameter_and_defers_its_type_check():
    """`count` is a number, the reference points at a string - that can
    only be judged once resolved, so plan validation must not reject it
    (execution time does; see the engine tests below)."""
    registry = registry_with(ProducerTool(), ConsumerTool())

    validate(
        [
            raw("s1", "produce"),
            raw("s2", "consume", {"text": ref(), "count": ref(path="data.query")}, depends_on=["s1"]),
        ],
        registry,
    )


def test_a_literal_of_the_wrong_type_is_still_rejected_at_plan_time():
    registry = registry_with(ProducerTool(), ConsumerTool())

    with pytest.raises(PlanValidationError, match="must be of type 'number'"):
        validate(
            [
                raw("s1", "produce"),
                raw("s2", "consume", {"text": ref(), "count": "many"}, depends_on=["s1"]),
            ],
            registry,
        )


def test_a_reference_to_a_step_that_is_not_a_declared_dependency_is_rejected():
    registry = registry_with(ProducerTool(), ConsumerTool())

    with pytest.raises(PlanValidationError, match="must be listed in its depends_on"):
        validate(
            [raw("s1", "produce"), raw("s2", "consume", {"text": ref()})],  # no depends_on
            registry,
        )


def test_depending_on_a_different_step_does_not_license_the_reference():
    registry = registry_with(ProducerTool(), ConsumerTool())

    with pytest.raises(PlanValidationError, match="must be listed in its depends_on"):
        validate(
            [
                raw("s1", "produce"),
                raw("s0", "produce"),
                raw("s2", "consume", {"text": ref("s1")}, depends_on=["s0"]),
            ],
            registry,
        )


def test_a_reference_to_an_unknown_step_is_rejected():
    registry = registry_with(ProducerTool(), ConsumerTool())

    with pytest.raises(PlanValidationError, match="unknown step"):
        validate(
            [raw("s2", "consume", {"text": ref("ghost")}, depends_on=["ghost"])], registry
        )


@pytest.mark.parametrize(
    "bad_reference",
    [
        {"from_step": "s1", "path": "results.0.title"},  # bad root
        {"from_step": "s1", "path": "data.results[0]"},
        {"from_step": "s1", "path": "data", "default": "x"},  # padded: half a reference
        {"from_step": "s1"},
    ],
)
def test_a_malformed_reference_is_rejected_not_passed_through_as_a_literal(bad_reference):
    registry = registry_with(ProducerTool(), ConsumerTool())

    with pytest.raises(PlanValidationError, match="argument 'text'"):
        validate(
            [raw("s1", "produce"), raw("s2", "consume", {"text": bad_reference}, depends_on=["s1"])],
            registry,
        )


def test_a_replan_step_may_reference_an_already_completed_step():
    registry = registry_with(ConsumerTool())

    steps = validate(
        [raw("s3", "consume", {"text": ref("s1")}, depends_on=["s1"])],
        registry,
        known_step_ids=frozenset({"s1"}),
    )

    assert steps[0].depends_on == ["s1"]


def test_an_object_argument_with_other_keys_is_an_ordinary_literal():
    registry = registry_with(ObjectConsumerTool())

    steps = validate([raw("s1", "consume_object", {"payload": {"a": 1}})], registry)

    assert steps[0].arguments == {"payload": {"a": 1}}


# --- 3. execution in JobEngine ---------------------------------------------------------------


class ScriptedPlanner(Planner):
    def __init__(self, *plans):
        self.plans = list(plans)
        self.calls = []

    def plan(self, goal, *, available_tools, context, autonomy, replan_context=None):
        self.calls.append(replan_context)
        if not self.plans:
            raise PlanningError("no more plans")
        return self.plans.pop(0)


def make_engine(store, registry, planner, **overrides):
    defaults = dict(
        max_steps=20,
        max_total_steps=40,
        max_retries_per_step=2,
        max_replans=2,
        retry_delay_seconds=0,
        poll_interval_seconds=1000,
    )
    defaults.update(overrides)
    return JobEngine(store=store, tools=registry, planner=planner, **defaults)


def run(plans, *tools, **engine_kwargs):
    store = SQLiteJobStore(":memory:")
    registry = registry_with(*tools)
    planner = ScriptedPlanner(*plans)
    engine = make_engine(store, registry, planner, **engine_kwargs)
    job = store.create("goal")
    engine.tick()
    return store.get(job.id), store, planner


def test_a_valid_reference_is_resolved_at_execution_time():
    consumer = ConsumerTool()

    job, store, _ = run(
        [
            [
                raw("s1", "produce"),
                raw("s2", "consume", {"text": ref(), "count": ref(path="data.results.1.score")}, ["s1"]),
            ]
        ],
        ProducerTool(),
        consumer,
    )

    assert job.status == JobStatus.COMPLETED
    assert consumer.calls == [{"text": "Acme Inc", "count": 3}]
    # The plan keeps the reference as written; the resolved value is not
    # written back over it.
    assert job.plan.get("s2").arguments["text"] == ref()


def test_an_object_can_be_passed_by_reference():
    consumer = ObjectConsumerTool()

    job, _, _ = run(
        [[raw("s1", "produce"), raw("s2", "consume_object", {"payload": ref(path="data")}, ["s1"])]],
        ProducerTool(),
        consumer,
    )

    assert job.status == JobStatus.COMPLETED
    assert consumer.calls == [RESULT["data"]]


def test_an_invalid_path_fails_the_step_and_the_tool_is_never_called():
    consumer = ConsumerTool()

    job, _, _ = run(
        [[raw("s1", "produce"), raw("s2", "consume", {"text": ref(path="data.nope")}, ["s1"])]],
        ProducerTool(),
        consumer,
        max_replans=0,
    )

    assert job.status == JobStatus.FAILED
    step = job.plan.get("s2")
    assert step.status == StepStatus.FAILED
    assert "Could not resolve a step reference" in step.error and "nope" in step.error
    assert consumer.calls == []
    assert step.attempts == 0  # it never even started


def test_a_reference_failure_is_never_retried():
    """max_retries_per_step=5 would re-run a flaky tool, but a reference
    can't start resolving differently - one failure, no retries."""
    consumer = ConsumerTool()

    job, _, _ = run(
        [[raw("s1", "produce"), raw("s2", "consume", {"text": ref(path="data.nope")}, ["s1"])]],
        ProducerTool(),
        consumer,
        max_replans=0,
        max_retries_per_step=5,
    )

    assert [e.kind for e in job.events].count("step_reference_failed") == 1
    assert [e.kind for e in job.events].count("step_started") == 1  # only s1's


def test_a_reference_failure_goes_through_the_existing_bounded_replan():
    consumer = ConsumerTool()

    job, _, planner = run(
        [
            [raw("s1", "produce"), raw("s2", "consume", {"text": ref(path="data.nope")}, ["s1"])],
            [raw("s3", "consume", {"text": "written out literally"}, ["s1"])],
        ],
        ProducerTool(),
        consumer,
        max_replans=1,
    )

    assert job.status == JobStatus.COMPLETED
    assert job.replan_count == 1
    assert consumer.calls == [{"text": "written out literally", "count": 0}]
    replan_context = planner.calls[1]
    assert replan_context.failed_step.id == "s2"
    assert "nope" in replan_context.failure_reason


def test_repeated_reference_failures_stop_at_the_replan_limit():
    bad = [raw("s1", "produce"), raw("s2", "consume", {"text": ref(path="data.nope")}, ["s1"])]
    bad_replan = [raw("s3", "consume", {"text": ref("s1", "data.nope")}, ["s1"])]
    bad_replan_2 = [raw("s4", "consume", {"text": ref("s1", "data.nope")}, ["s1"])]

    job, _, planner = run([bad, bad_replan, bad_replan_2], ProducerTool(), ConsumerTool(), max_replans=2)

    assert job.status == JobStatus.FAILED
    assert job.replan_count == 2
    assert len(planner.calls) == 3  # initial plan + exactly two replans


def test_a_source_with_no_useful_value_fails_the_referencing_step():
    class NothingTool(Tool):
        name = "nothing"
        description = "returns success with no data"
        permission = PermissionLevel.READ

        def execute(self) -> ToolResult:
            return ToolResult.success()

    consumer = ConsumerTool()
    job, _, _ = run(
        [[raw("s1", "nothing"), raw("s2", "consume", {"text": ref(path="data.title")}, ["s1"])]],
        NothingTool(),
        consumer,
        max_replans=0,
    )

    assert job.plan.get("s2").status == StepStatus.FAILED
    assert "cannot step into a NoneType" in job.plan.get("s2").error
    assert consumer.calls == []


def test_a_resolved_value_of_the_wrong_type_fails_the_step():
    consumer = ConsumerTool()

    job, _, _ = run(
        [
            [
                raw("s1", "produce"),
                # count is a number; data.query is the string "acme"
                raw("s2", "consume", {"text": ref(), "count": ref(path="data.query")}, ["s1"]),
            ]
        ],
        ProducerTool(),
        consumer,
        max_replans=0,
    )

    step = job.plan.get("s2")
    assert step.status == StepStatus.FAILED
    assert "resolved arguments are not acceptable" in step.error
    assert "must be of type 'number'" in step.error
    assert consumer.calls == []


def test_a_resolved_argument_set_missing_a_required_parameter_fails():
    class EmptyResultTool(Tool):
        name = "keys"
        description = "x"
        permission = PermissionLevel.READ

        def execute(self) -> ToolResult:
            return ToolResult.success(data={"text": None})

    consumer = ConsumerTool()
    job, _, _ = run(
        [[raw("s1", "keys"), raw("s2", "consume", {"text": ref(path="data.text")}, ["s1"])]],
        EmptyResultTool(),
        consumer,
        max_replans=0,
    )

    # A null is a real value of the wrong type for a string parameter.
    assert job.plan.get("s2").status == StepStatus.FAILED
    assert consumer.calls == []


def test_the_tools_own_value_check_runs_on_the_resolved_value():
    class ForbiddenTool(Tool):
        name = "forbidden_source"
        description = "x"
        permission = PermissionLevel.READ

        def execute(self) -> ToolResult:
            return ToolResult.success(message="forbidden")

    consumer = ConsumerTool()
    job, _, _ = run(
        [[raw("s1", "forbidden_source"), raw("s2", "consume", {"text": ref(path="message")}, ["s1"])]],
        ForbiddenTool(),
        consumer,
        max_replans=0,
    )

    assert "that text is not allowed" in job.plan.get("s2").error
    assert consumer.calls == []


def test_an_oversized_referenced_string_fails_the_step_and_the_tool_is_never_called():
    """Fail rather than truncate: cutting research data could silently
    change its meaning."""

    class LongTool(Tool):
        name = "long"
        description = "x"
        permission = PermissionLevel.READ

        def execute(self) -> ToolResult:
            return ToolResult.success(message="L" * (MAX_REFERENCE_CHARS + 1))

    consumer = ConsumerTool()
    job, _, _ = run(
        [[raw("s1", "long"), raw("s2", "consume", {"text": ref(path="message")}, ["s1"])]],
        LongTool(),
        consumer,
        max_replans=0,
    )

    step = job.plan.get("s2")
    assert job.status == JobStatus.FAILED
    assert step.status == StepStatus.FAILED
    assert "Could not resolve a step reference" in step.error
    assert "argument 'text'" in step.error and "too large" in step.error
    assert consumer.calls == []
    assert "reference_truncated" not in [e.kind for e in job.events]  # nothing is ever truncated


def test_an_oversized_reference_goes_through_the_existing_bounded_replan():
    class LongTool(Tool):
        name = "long"
        description = "x"
        permission = PermissionLevel.READ

        def execute(self) -> ToolResult:
            return ToolResult.success(message="L" * (MAX_REFERENCE_CHARS + 1))

    consumer = ConsumerTool()
    job, _, planner = run(
        [
            [raw("s1", "long"), raw("s2", "consume", {"text": ref(path="message")}, ["s1"])],
            [raw("s3", "consume", {"text": "a short literal instead"}, ["s1"])],
        ],
        LongTool(),
        consumer,
        max_replans=1,
    )

    assert job.status == JobStatus.COMPLETED
    assert job.replan_count == 1
    assert consumer.calls == [{"text": "a short literal instead", "count": 0}]
    assert "too large" in planner.calls[1].failure_reason


def test_an_oversized_container_fails_the_step():
    class SprawlTool(Tool):
        name = "sprawl"
        description = "x"
        permission = PermissionLevel.READ

        def execute(self) -> ToolResult:
            return ToolResult.success(data={f"k{i}": "v" for i in range(5000)})

    consumer = ObjectConsumerTool()
    job, _, _ = run(
        [[raw("s1", "sprawl"), raw("s2", "consume_object", {"payload": ref(path="data")}, ["s1"])]],
        SprawlTool(),
        consumer,
        max_replans=0,
    )

    assert "too large" in job.plan.get("s2").error
    assert consumer.calls == []


def test_a_resolved_value_that_looks_like_a_reference_is_not_resolved_again():
    """Tool output is untrusted data. Even if it contains something shaped
    like a reference, it is passed on verbatim - resolution is one level,
    driven only by the validated plan."""

    class SneakyTool(Tool):
        name = "sneaky"
        description = "x"
        permission = PermissionLevel.READ

        def execute(self) -> ToolResult:
            return ToolResult.success(data={"payload": {"from_step": "s1", "path": "data"}})

    consumer = ObjectConsumerTool()
    job, _, _ = run(
        [[raw("s1", "sneaky"), raw("s2", "consume_object", {"payload": ref(path="data.payload")}, ["s1"])]],
        SneakyTool(),
        consumer,
    )

    assert job.status == JobStatus.COMPLETED
    assert consumer.calls == [{"from_step": "s1", "path": "data"}]  # inert data


def test_execution_time_resolution_rechecks_the_dependency_rule_on_a_stored_plan():
    """Defence in depth: a plan that somehow bypassed validation (a
    hand-edited row) still can't reference an undeclared step."""
    registry = registry_with(ProducerTool(), ConsumerTool())
    engine = make_engine(SQLiteJobStore(":memory:"), registry, ScriptedPlanner())
    producer = PlanStep(
        id="s1", tool_name="produce", status=StepStatus.SUCCEEDED, result=RESULT
    )
    consumer_step = PlanStep(id="s2", tool_name="consume", arguments={"text": ref()}, depends_on=[])
    job = Job(
        id="j", goal="g", status=JobStatus.RUNNING, priority=None, autonomy=None,
        created_at="t", updated_at="t", plan=Plan(steps=[producer, consumer_step]),
    )

    with pytest.raises(ReferenceError, match="not one of this step's declared dependencies"):
        engine._resolve_step_arguments(job, consumer_step)


def test_execution_time_resolution_requires_the_source_to_have_succeeded():
    registry = registry_with(ProducerTool(), ConsumerTool())
    engine = make_engine(SQLiteJobStore(":memory:"), registry, ScriptedPlanner())
    for status in (StepStatus.PENDING, StepStatus.FAILED, StepStatus.SKIPPED):
        producer = PlanStep(id="s1", tool_name="produce", status=status, result=RESULT)
        consumer_step = PlanStep(
            id="s2", tool_name="consume", arguments={"text": ref()}, depends_on=["s1"]
        )
        job = Job(
            id="j", goal="g", status=JobStatus.RUNNING, priority=None, autonomy=None,
            created_at="t", updated_at="t", plan=Plan(steps=[producer, consumer_step]),
        )

        with pytest.raises(ReferenceError, match="has not succeeded"):
            engine._resolve_step_arguments(job, consumer_step)


def test_a_missing_stored_result_on_a_succeeded_source_is_an_error():
    registry = registry_with(ProducerTool(), ConsumerTool())
    engine = make_engine(SQLiteJobStore(":memory:"), registry, ScriptedPlanner())
    producer = PlanStep(id="s1", tool_name="produce", status=StepStatus.SUCCEEDED, result=None)
    consumer_step = PlanStep(id="s2", tool_name="consume", arguments={"text": ref()}, depends_on=["s1"])
    job = Job(
        id="j", goal="g", status=JobStatus.RUNNING, priority=None, autonomy=None,
        created_at="t", updated_at="t", plan=Plan(steps=[producer, consumer_step]),
    )

    with pytest.raises(ReferenceError, match="no stored result"):
        engine._resolve_step_arguments(job, consumer_step)


def test_a_step_without_references_is_executed_exactly_as_before():
    consumer = ConsumerTool()

    job, _, _ = run([[raw("s1", "consume", {"text": "plain", "count": 2})]], consumer)

    assert job.status == JobStatus.COMPLETED
    assert consumer.calls == [{"text": "plain", "count": 2}]


# --- a gated step shows what it will actually receive ---------------------------------------------


class GatedConsumerTool(ConsumerTool):
    name = "gated_consume"
    permission = PermissionLevel.CONFIRM


def test_a_confirmation_message_shows_resolved_values_not_opaque_references():
    store = SQLiteJobStore(":memory:")
    registry = registry_with(ProducerTool(), GatedConsumerTool())
    planner = ScriptedPlanner(
        [raw("s1", "produce"), raw("s2", "gated_consume", {"text": ref()}, ["s1"])]
    )
    engine = make_engine(store, registry, planner)
    job = store.create("goal")

    engine.tick()

    waiting = store.get(job.id)
    assert waiting.status == JobStatus.AWAITING_CONFIRMATION
    assert "text='Acme Inc'" in waiting.confirmation_message
    assert "from_step" not in waiting.confirmation_message


def test_a_long_value_in_a_confirmation_message_is_shortened():
    class LongTool(Tool):
        name = "long"
        description = "x"
        permission = PermissionLevel.READ

        def execute(self) -> ToolResult:
            return ToolResult.success(message="Z" * 3000)

    store = SQLiteJobStore(":memory:")
    registry = registry_with(LongTool(), GatedConsumerTool())
    engine = make_engine(
        store,
        registry,
        ScriptedPlanner([raw("s1", "long"), raw("s2", "gated_consume", {"text": ref(path="message")}, ["s1"])]),
    )
    job = store.create("goal")

    engine.tick()

    message = store.get(job.id).confirmation_message
    assert len(message) < 600 and "...[truncated]" in message


# --- persistence / restart ---------------------------------------------------------------------------


def test_references_resolve_from_persisted_results_after_a_restart(tmp_path):
    """Step 1 runs and is persisted; the process 'dies' while step 2 (a
    gated step referencing it) waits for confirmation. A brand-new store
    and engine on the same file confirm and finish it - the reference is
    resolved from the persisted result, not from anything in memory."""
    db = str(tmp_path / "jobs.db")

    store_1 = SQLiteJobStore(db)
    engine_1 = make_engine(
        store_1,
        registry_with(ProducerTool(), GatedConsumerTool()),
        ScriptedPlanner([raw("s1", "produce"), raw("s2", "gated_consume", {"text": ref()}, ["s1"])]),
    )
    job = store_1.create("goal")
    engine_1.tick()
    assert store_1.get(job.id).status == JobStatus.AWAITING_CONFIRMATION
    del engine_1, store_1  # "restart"

    store_2 = SQLiteJobStore(db)
    consumer = GatedConsumerTool()
    registry_2 = registry_with(ProducerTool(), consumer, ConfirmJobTool(store=store_2))
    engine_2 = make_engine(store_2, registry_2, ScriptedPlanner())  # would fail if it replanned
    assert registry_2.execute("confirm_job", {"job_id": job.id, "approve": True}).status.value == "success"

    engine_2.tick()

    finished = store_2.get(job.id)
    assert finished.status == JobStatus.COMPLETED
    assert consumer.calls == [{"text": "Acme Inc", "count": 0}]


# --- the two limits are distinct: references 12000, analyze material 8000 ---------------------------------


class _BigTool(Tool):
    name = "big"
    description = "returns a string of a chosen size"
    permission = PermissionLevel.READ

    def __init__(self, size):
        self.size = size

    def execute(self) -> ToolResult:
        return ToolResult.success(data={"blob": "B" * self.size})


def test_a_reference_up_to_the_reference_limit_resolves():
    consumer = ConsumerTool()

    job, _, _ = run(
        [[raw("s1", "big"), raw("s2", "consume", {"text": ref(path="data.blob")}, ["s1"])]],
        _BigTool(MAX_REFERENCE_CHARS),
        consumer,
    )

    assert job.status == JobStatus.COMPLETED
    assert consumer.calls[0]["text"] == "B" * MAX_REFERENCE_CHARS  # whole, untouched


def test_a_reference_one_character_over_the_limit_fails_clearly():
    consumer = ConsumerTool()

    job, _, _ = run(
        [[raw("s1", "big"), raw("s2", "consume", {"text": ref(path="data.blob")}, ["s1"])]],
        _BigTool(MAX_REFERENCE_CHARS + 1),
        consumer,
        max_replans=0,
    )

    error = job.plan.get("s2").error
    assert job.plan.get("s2").status == StepStatus.FAILED
    assert f"{MAX_REFERENCE_CHARS + 1} characters" in error and f"the limit is {MAX_REFERENCE_CHARS}" in error
    assert consumer.calls == []


def test_material_between_the_analyze_limit_and_the_reference_limit_resolves_but_analyze_refuses_it():
    """Reference resolution is bounded at 12000; analyze's own input at
    8000. A 9000-char value resolves fine, then analyze rejects it with its
    own clear size error - the model is never called, nothing is cut."""
    from app.tools.analyze import MAX_MATERIAL_CHARS, AnalyzeTool

    class NeverCalled:
        def generate_reply(self, message):
            raise AssertionError("the model must not be called for oversized material")

    assert MAX_MATERIAL_CHARS == 8000 < MAX_REFERENCE_CHARS
    job, _, _ = run(
        [
            [
                raw("s1", "big"),
                raw("s2", "analyze", {"instruction": "summarise", "material": ref(path="data")}, ["s1"]),
            ]
        ],
        _BigTool(9000),
        AnalyzeTool(provider=NeverCalled()),
        max_replans=0,
    )

    error = job.plan.get("s2").error
    assert job.plan.get("s2").status == StepStatus.FAILED
    assert "too large" in error and f"limit {MAX_MATERIAL_CHARS}" in error
