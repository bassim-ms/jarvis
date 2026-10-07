"""
Wording a proposal and the replies around it - fixed templates, bounded, no model.

The proposal is shown EXACTLY as it is stored: `render_proposal` is a pure function of the steps, so what the user reads is what a later
approval covers. Stored text appears only in quoted slots (the step text was built that way), and every step says whose it is: "Your part:"
or "My part:". Nothing here says Jarvis will do something it has no tool for.
"""

from __future__ import annotations

from app.action_proposal.base import ActionProposal

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
DROPPED = 'Okay - I\'ve dropped that plan. Say "make a plan" any time for a fresh one.'
NOTHING_TO_DO = "Okay - without that there's nothing left for me to do, so I've dropped the plan."
UNWIRED = "Understood - but I can't carry out my part yet, so nothing has been started."
FAILED = "Something went wrong, so nothing was started. Say \"make a plan\" to try again."


def changed(problems: list[str]) -> str:
    listed = "\n".join(f"- {problem}" for problem in problems[:4])
    return f'Things changed since I showed you the plan, so I haven\'t done anything:\n{listed}\nSay "make a plan" for a fresh one.'
