"""
Tests for app/self_development/base.py: DevelopmentPlan/PlannedFileChange
validation (including that path safety/secret exclusion is enforced at
CONSTRUCTION time, not just by the planner that builds one from AI
output), and DevelopmentSession/SessionStage bookkeeping.
"""

import pytest

from app.self_development.base import (
    CheckRun,
    CommandResult,
    DevelopmentCapability,
    DevelopmentPlan,
    DevelopmentPlanError,
    DevelopmentSession,
    PlannedFileChange,
    ROLLBACK_AVAILABLE_STAGES,
    SessionStage,
    permission_for,
)
from app.tools.base import PermissionLevel


# -- DevelopmentCapability -> PermissionLevel mapping ------------------------


def test_every_capability_has_a_permission():
    for capability in DevelopmentCapability:
        assert isinstance(permission_for(capability), PermissionLevel)


def test_automatic_capabilities_map_to_read_or_safe_action():
    from app.tools.base import AUTOMATIC_PERMISSIONS

    for capability in (
        DevelopmentCapability.CODE_READ,
        DevelopmentCapability.GIT_DIFF,
        DevelopmentCapability.TEST_RUN,
        DevelopmentCapability.BRANCH_CREATE,
    ):
        assert permission_for(capability) in AUTOMATIC_PERMISSIONS


def test_confirm_gated_capabilities():
    assert permission_for(DevelopmentCapability.CODE_EDIT) is PermissionLevel.CONFIRM
    assert permission_for(DevelopmentCapability.COMMAND_RUN) is PermissionLevel.CONFIRM
    assert permission_for(DevelopmentCapability.ROLLBACK) is PermissionLevel.CONFIRM


def test_commit_requires_the_strongest_permission_tier():
    assert permission_for(DevelopmentCapability.COMMIT) is PermissionLevel.DANGEROUS


# -- PlannedFileChange --------------------------------------------------


def test_planned_file_change_accepts_a_safe_path():
    change = PlannedFileChange(path="app/foo.py", description="add a function")
    assert change.path == "app/foo.py"


def test_planned_file_change_rejects_traversal():
    with pytest.raises(DevelopmentPlanError):
        PlannedFileChange(path="../etc/passwd", description="x")


def test_planned_file_change_rejects_absolute_path():
    with pytest.raises(DevelopmentPlanError):
        PlannedFileChange(path="/etc/passwd", description="x")


def test_planned_file_change_rejects_a_secret_file():
    with pytest.raises(DevelopmentPlanError):
        PlannedFileChange(path=".env", description="add a key")


def test_planned_file_change_rejects_empty_description():
    with pytest.raises(DevelopmentPlanError):
        PlannedFileChange(path="app/foo.py", description="")


def test_planned_file_change_round_trips_through_dict():
    change = PlannedFileChange(path="app/foo.py", description="x")
    assert PlannedFileChange.from_dict(change.to_dict()) == change


# -- DevelopmentPlan ------------------------------------------------------


def test_minimal_development_plan_constructs():
    plan = DevelopmentPlan(objective="Investigate a failing test.")
    assert plan.files_affected == ()
    assert plan.file_paths() == frozenset()


def test_development_plan_with_a_secret_file_change_is_rejected_at_construction():
    """A DevelopmentPlan can never even be built with a secret file in
    its own file list - defense in depth beyond whatever validation
    the planner that builds one from AI output also does."""
    with pytest.raises(DevelopmentPlanError):
        DevelopmentPlan(
            objective="x",
            files_affected=(PlannedFileChange(path=".env", description="y"),),
        )


def test_development_plan_rejects_non_capability_required_permissions():
    with pytest.raises(DevelopmentPlanError):
        DevelopmentPlan(objective="x", required_permissions=("code_edit",))  # a raw string


def test_development_plan_empty_objective_rejected():
    with pytest.raises(DevelopmentPlanError):
        DevelopmentPlan(objective="")


def test_development_plan_file_paths_helper():
    plan = DevelopmentPlan(
        objective="x",
        files_affected=(
            PlannedFileChange(path="a.py", description="d1"),
            PlannedFileChange(path="b.py", description="d2"),
        ),
        required_permissions=(DevelopmentCapability.CODE_EDIT,),
    )
    assert plan.file_paths() == {"a.py", "b.py"}


def test_development_plan_round_trips_through_dict():
    plan = DevelopmentPlan(
        objective="Add a feature",
        files_affected=(PlannedFileChange(path="a.py", description="d1"),),
        dependencies=("app.tools.base",),
        tests_to_run=("pytest -q",),
        risks=("might break X",),
        required_permissions=(DevelopmentCapability.CODE_EDIT, DevelopmentCapability.TEST_RUN),
    )
    assert DevelopmentPlan.from_dict(plan.to_dict()) == plan


# -- DevelopmentSession / SessionStage ------------------------------------


def test_new_session_has_no_rollback_available():
    session = DevelopmentSession(
        id="s1", request="do X", stage=SessionStage.STARTED, original_branch="master"
    )
    assert session.rollback_available() is False


def test_rollback_available_only_for_the_expected_stages_and_only_with_a_dev_branch():
    for stage in SessionStage:
        session = DevelopmentSession(
            id="s1", request="do X", stage=stage, original_branch="master", dev_branch="dev/s1"
        )
        expected = stage in ROLLBACK_AVAILABLE_STAGES
        assert session.rollback_available() is expected, stage


def test_rollback_never_available_without_a_dev_branch_even_in_an_eligible_stage():
    session = DevelopmentSession(
        id="s1",
        request="do X",
        stage=SessionStage.BRANCH_CREATED,
        original_branch="master",
        dev_branch=None,
    )
    assert session.rollback_available() is False


def test_session_log_appends_without_mutating_the_previous_tuple():
    session = DevelopmentSession(
        id="s1", request="do X", stage=SessionStage.STARTED, original_branch="master"
    )
    before = session.messages
    session.log("first")
    session.log("second")
    assert before == ()
    assert session.messages == ("first", "second")


# -- CommandResult ---------------------------------------------------------


def test_command_result_succeeded_requires_exit_code_zero_and_no_timeout():
    ok = CommandResult(argv=("pytest",), exit_code=0, stdout="", stderr="", timed_out=False)
    failed = CommandResult(argv=("pytest",), exit_code=1, stdout="", stderr="", timed_out=False)
    timed_out = CommandResult(argv=("pytest",), exit_code=None, stdout="", stderr="", timed_out=True)
    assert ok.succeeded is True
    assert failed.succeeded is False
    assert timed_out.succeeded is False


def test_check_run_holds_its_command_and_result():
    result = CommandResult(argv=("pytest",), exit_code=0, stdout="ok", stderr="", timed_out=False)
    check = CheckRun(command="pytest -q", result=result)
    assert check.command == "pytest -q"
    assert check.result.succeeded is True
