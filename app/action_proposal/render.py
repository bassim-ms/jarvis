"""
Wording a proposal and the replies around it - fixed templates, bounded, no model.

The proposal is shown EXACTLY as it is stored: `render_proposal` is a pure function of the steps, so what the user reads is what a later
approval covers. Stored text appears only in quoted slots (the step text was built that way), and every step says whose it is: "Your part:"
or "My part:". Nothing here says Jarvis will do something it has no tool for.
"""

from __future__ import annotations

from app.action_proposal.base import AMBIGUOUS, COMPLETION_TEXT_LIMIT, NO_MATCH, UNSHOWABLE, ActionProposal
from app.morning_brief.gather import safe_text

MAX_CHARS = 2000  # room for the user's two steps, a reminder, and a note step shown in full - the user must see exactly what a yes covers


def _positions(numbers: tuple[int, ...]) -> str:
    if len(numbers) == 1:
        return f"step {numbers[0]}"
    return "steps " + ", ".join(str(n) for n in numbers[:-1]) + f" and {numbers[-1]}"


def approval_line(proposal: ActionProposal) -> str:
    mine = proposal.my_steps
    return (
        f"Say yes and I'll do my part ({_positions(mine)}) - I won't start anything else on my own. "
        "Say no to drop it."
        if mine
        else ""
    )


def header(proposal: ActionProposal) -> str:
    if proposal.subject:
        return f"Here's what I'd do for \"{proposal.subject}\":"
    return "Here's what I'd do:"


def numbered(proposal: ActionProposal) -> list[str]:
    return [f"{i}. {step.text}" for i, step in enumerate(proposal.steps, start=1)]


def render_proposal(proposal: ActionProposal, *, approvable: bool = True) -> str:
    lines = [header(proposal), *numbered(proposal)]
    if approvable and proposal.my_steps:
        lines.append(approval_line(proposal))
    elif not proposal.my_steps:
        lines.append("These steps are all yours - there's nothing I can do on my own for this yet, so there's nothing to approve.")
    text = "\n".join(lines)
    return text if len(text) <= MAX_CHARS else text[: MAX_CHARS - 3].rstrip() + "..."


def with_approval_line(body: str, proposal: ActionProposal) -> str:
    """A reply about a waiting proposal always ends by restating what a yes would do, so a later yes has one clear referent."""
    line = approval_line(proposal)
    return f"{body}\n{line}" if line else body


def explain_step(proposal: ActionProposal, number: int) -> str:
    step = proposal.steps[number - 1]
    owner = "mine" if step.mine else "yours"
    body = f"Step {number} is {owner}: {step.text.split(': ', 1)[-1].removeprefix('then ')}"
    if step.why:
        body += f" Why: {step.why}."
    return with_approval_line(body, proposal)


def explain_after(proposal: ActionProposal, number: int) -> str:
    if number >= len(proposal.steps):
        body = f"Step {number} is the last step - nothing comes after it."
    else:
        body = f"After step {number} comes step {number + 1}: {proposal.steps[number].text}"
    return with_approval_line(body, proposal)


def explain_why_plan(proposal: ActionProposal) -> str:
    first = proposal.steps[0]
    if first.why:
        body = f"I put step 1 first because {first.why}."
    else:
        body = "I went by what's already on your list - nothing in it said otherwise."
    return with_approval_line(body, proposal)


def explain_why_step(proposal: ActionProposal, number: int) -> str:
    step = proposal.steps[number - 1]
    body = f"Step {number}: {step.why}." if step.why else f"Step {number} comes straight from your own list."
    return with_approval_line(body, proposal)


def no_such_step(proposal: ActionProposal) -> str:
    count = len(proposal.steps)
    return with_approval_line(f"There {'is only 1 step' if count == 1 else f'are only {count} steps'}.", proposal)


def nudge(proposal: ActionProposal) -> str:
    return with_approval_line('I need a clear yes before I do anything - say "yes" or "go ahead".', proposal)


def cannot_change(proposal: ActionProposal) -> str:
    return with_approval_line(
        'I can only skip a step for now - say "skip step 2" - or ask for a fresh plan once your tasks have changed.', proposal
    )


NOTE_UNAVAILABLE = "I can't add the note step to this plan right now, so a yes won't save a note."
TASK_UNAVAILABLE = "I can't add the task step to this plan right now, so a yes won't add a task."
DROPPED = 'Okay - I\'ve dropped that plan. Say "make a plan" any time for a fresh one.'
NOTHING_TO_DO = "Okay - without that there's nothing left for me to do, so I've dropped the plan."
UNWIRED = "Understood - but I can't carry out my part yet, so nothing has been started."
FAILED = "Something went wrong, so nothing was started. Say \"make a plan\" to try again."


PLAN_AGAIN = 'Say "make a plan" for a fresh one.'
COMPLETE_AGAIN = 'Say it again with the task\'s exact words in quotes - for example: Mark "<the task>" as done.'


def changed(problems: list[str], again: str = PLAN_AGAIN) -> str:
    listed = "\n".join(f"- {problem}" for problem in problems[:4])
    return f"Things changed since I showed you the plan, so I haven't done anything:\n{listed}\n{again}"


# -- marking one named task done: why there is no proposal (stateless: nothing is held, nothing was changed) ------------------------------------


def _called(words: str) -> str:
    """How to name the task the user asked for: `called "<their words>"` - only if their words come back unchanged - otherwise `with those words`."""
    shown = safe_text(words, COMPLETION_TEXT_LIMIT)
    return f'called "{shown}"' if shown and shown == words else "with those words"


def completion_none(reason: str, words: str, matches: int = 0) -> str:
    called = _called(words)
    if reason == NO_MATCH:
        return (
            f"I don't see an open task {called}. I only mark a task done when its exact words are in quotes and exactly one open task "
            f"has them - nothing was changed. {COMPLETE_AGAIN}"
        )
    if reason == AMBIGUOUS:
        return f"I found {matches} open tasks {called}, so I can't tell which one you mean - nothing was changed."
    if reason == UNSHOWABLE:
        return "I found that task, but I can't show its words back to you exactly, so I haven't marked anything - nothing was changed."
    return "I can't mark tasks done right now, so nothing was changed."
