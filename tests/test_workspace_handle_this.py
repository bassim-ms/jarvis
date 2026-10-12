"""
Automation Slice A1 - Workspace Files and Handle This: a request to make or use a text file must be UNDERSTOOD as supported work (so it reaches the
Planner, which is offered the workspace tools), while changing, replacing or deleting existing files stays unsupported - the create-only promise holds
at the front door as well as in the tool.

Real: the capability boundary, the deterministic interpreter, the intake and the LLM interpreter's prompt. The model's answer is scripted.
"""

import json

from app.handle_this.base import FailureCode, GoalFamily, HandleThisStatus, UnsupportedCategory
from app.handle_this.boundary import DEFAULT_BOUNDARY
from app.handle_this.intake import HandleThisIntake
from app.handle_this.interpretation import _INSTRUCTIONS
from test_handle_this_intake import llm_intake, llm_json, request_of

WORKSPACE_TOOLS = ("list_workspace_files", "read_workspace_file", "write_workspace_file")
OFFERED = ["web_search", "analyze", "save_note", "create_task", *WORKSPACE_TOOLS]


def test_the_boundary_declares_the_three_workspace_tools_under_organization():
    declared = DEFAULT_BOUNDARY.declared_tools(GoalFamily.ORGANIZATION)

    assert set(WORKSPACE_TOOLS) <= declared
    assert not set(WORKSPACE_TOOLS) & DEFAULT_BOUNDARY.declared_tools(GoalFamily.COMPUTER_WORKFLOW)  # files are not a computer-control workflow
    assert DEFAULT_BOUNDARY.assess([GoalFamily.ORGANIZATION], ["write_workspace_file"]).supported  # one tool of the family is enough


def test_the_workspace_tools_are_listed_for_organization_and_nothing_else_widened():
    assert DEFAULT_BOUNDARY.declared_tools(GoalFamily.RESEARCH) == {"web_search"}
    assert DEFAULT_BOUNDARY.declared_tools(GoalFamily.ANALYSIS) == {"analyze"}
    everything = set().union(*(DEFAULT_BOUNDARY.declared_tools(f) for f in (GoalFamily.RESEARCH, GoalFamily.ANALYSIS, GoalFamily.ORGANIZATION)))
    assert not {"delete_workspace_file", "overwrite_workspace_file", "rename_workspace_file", "append_workspace_file"} & everything


def test_a_request_to_make_a_study_guide_and_save_it_as_a_file_is_understood_as_organization():
    intake = HandleThisIntake(available_tools=lambda: OFFERED)

    request = request_of(intake, "Prepare a study guide for my physics exam and save it as a file")

    assert request.status is HandleThisStatus.READY and request.goal_family is GoalFamily.ORGANIZATION


def test_a_request_the_deterministic_reader_cannot_place_goes_to_the_model_and_is_accepted_when_it_says_organization():
    provider, intake = llm_intake(llm_json(interpreted_goal="Make a study checklist from physics-notes.md"), available_tools=lambda: OFFERED)

    request = request_of(intake, "Prepare a study checklist from physics-notes.md")

    assert request.status is HandleThisStatus.READY and request.goal_family is GoalFamily.ORGANIZATION
    assert len(provider.prompts) == 1  # the model was asked exactly once, and only because no pattern recognised the wording


def test_without_the_model_that_same_request_is_not_guessed_at():
    result = HandleThisIntake(available_tools=lambda: OFFERED).understand("Prepare a study checklist from physics-notes.md")

    assert not result.ok and result.failure.code is FailureCode.UNCLEAR_OBJECTIVE  # no signal: never guessed into a family


def test_the_interpreters_instructions_now_count_reading_and_creating_new_files_as_organization_and_changing_existing_ones_as_unsupported():
    assert "reading or creating NEW text files in the user's Jarvis workspace folder" in _INSTRUCTIONS
    assert "changing, replacing or deleting existing files" in _INSTRUCTIONS
    assert "organization = tasks, reminders, notes, calendar, and" in _INSTRUCTIONS  # the family definition still leads with the old ones


def test_the_prompt_the_model_actually_receives_carries_those_instructions():
    provider, intake = llm_intake(llm_json(), available_tools=lambda: OFFERED)

    request_of(intake, "Prepare a study checklist from physics-notes.md")

    assert "creating NEW text files" in provider.prompts[0] and "deleting existing files" in provider.prompts[0]


def test_the_model_cannot_widen_the_boundary_by_calling_a_file_request_a_family_with_no_tools_on_offer():
    provider, intake = llm_intake(llm_json(), available_tools=lambda: ["web_search", "analyze"])  # no organization tool at all is on offer

    result = intake.understand("Prepare a study checklist from physics-notes.md")

    assert result.ok and result.request.status is HandleThisStatus.UNSUPPORTED
    assert result.request.unsupported.category is UnsupportedCategory.UNAVAILABLE_CAPABILITY  # the boundary, not the model, decides availability


def test_destructive_file_requests_stay_unsupported_before_any_model_or_tool_is_involved():
    intake = HandleThisIntake(available_tools=lambda: OFFERED)

    for message in (
        "Delete all the files in my workspace",
        "Rename all my files",
        "Overwrite the file physics-notes.md with this summary",
        "Remove my files and start again",
    ):
        request = request_of(intake, message)
        assert request.status is HandleThisStatus.UNSUPPORTED, message
        assert request.unsupported.category is UnsupportedCategory.FILE_OR_SYSTEM_CHANGE, message


def test_a_model_that_calls_an_overwrite_unsupported_is_believed():
    unsupported = llm_json(goal_family="unsupported", required_families=[], unsupported_category="file_or_system_change")
    provider, intake = llm_intake(unsupported, available_tools=lambda: OFFERED)

    request = request_of(intake, "Replace the contents of physics-notes.md with a summary")

    assert request.status is HandleThisStatus.UNSUPPORTED and request.unsupported.category is UnsupportedCategory.FILE_OR_SYSTEM_CHANGE
    assert json.loads(unsupported)["unsupported_category"] == "file_or_system_change"
