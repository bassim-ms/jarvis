"""
Write policy for persistent memory: the types.

The question this package answers, deterministically, for every proposed
write to the World Model or Memory: SHOULD it be persisted, WHERE, with what
provenance, and does it need the user first?

    OBSERVED (a proposal) -> CANDIDATE (staged, not a fact) -> VALIDATED (the user
    said it plainly) -> PERSISTED (World Model / Memory, through their own services)

Nothing here touches a store. `WritePolicy` (policy.py) decides; `MemoryWriteService`
(service.py) applies the decision through the EXISTING WorldModelService and
MemoryStore; a model can only ever PROPOSE (through a registered tool), and a
model's words never choose the outcome.

The distinctions this keeps:

  World Model   "what is currently true / happening in my world"  (preferences,
                projects, goals)                                    -> WORLD_MODEL
  Memory        "what Jarvis deliberately retains about the user"  -> MEMORY
  Candidate     "something Jarvis believes or was told uncertainly" -> CANDIDATE (staged in
                this package's own small table; never read as a fact, never shown to
                the planner or the context prompts as one)
  Temporary     "useful right now, not worth remembering"          -> TEMPORARY (nothing written)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.world_model.base import EntitySource

#: Longest value / key the policy will persist (one short line).
MAX_VALUE_CHARS = 200
MAX_KEY_CHARS = 60

#: Ceiling on a candidate's confidence: however often an inference is repeated,
#: it never reaches a persisted fact's confidence by itself.
CANDIDATE_CONFIDENCE_CEILING = 0.75
DEFAULT_CANDIDATE_CONFIDENCE = 0.5
#: What one more independent observation of the same candidate adds.
CORROBORATION_STEP = 0.1

#: Confidence a persisted USER-sourced write carries: an explicit command is
#: taken at face value; a plain first-person statement slightly less.
COMMAND_CONFIDENCE = 1.0
STATEMENT_CONFIDENCE = 0.9


class Origin(Enum):
    """Who authored the proposal - decided by TRUSTED code at the call site,
    never by a model's argument."""

    USER_COMMAND = "user_command"  # "remember that ...", "set my goal to ..."
    USER_STATEMENT = "user_statement"  # "I prefer dark mode."
    INFERENCE = "inference"  # something Jarvis concluded
    EXTERNAL = "external"  # a web page, email, document, search result, tool output


class Kind(Enum):
    FACT = "fact"  # free-form: Memory
    PREFERENCE = "preference"  # World Model Preference
    PROJECT = "project"  # World Model Project
    GOAL = "goal"  # World Model Goal (belongs to a Project)


class Destination(Enum):
    WORLD_MODEL = "world_model"
    MEMORY = "memory"
    CANDIDATE = "candidate"
    TEMPORARY = "temporary"
    ASK = "ask"  # not persisted: the user must say more first
    REJECT = "reject"  # never persisted


class Sensitivity(Enum):
    NONE = "none"
    SENSITIVE = "sensitive"  # personal / financial / health...: only with the user's say-so
    SECRET = "secret"  # credentials, keys, card numbers: NEVER stored, by anything


@dataclass(frozen=True)
class WriteProposal:
    """One proposed write.

    `key` is the stable name a later statement can contradict ("budget");
    `value` what is to be recorded; `evidence` the user's own words it came
    from. `hedged` - the user said it uncertainly ("maybe", "I think");
    `temporary` - it is about now, not a lasting truth; `supersede` - the user
    signalled a change ("now", "actually", "no longer");
    `acknowledged_sensitive` - the user explicitly said to keep something
    sensitive anyway (never lifts the ban on secrets); `project` - for a
    GOAL, the project's name (empty: the one active project)."""

    kind: Kind
    key: str
    value: str
    origin: Origin
    category: str = "general"
    evidence: str = ""
    confidence: float | None = None
    hedged: bool = False
    temporary: bool = False
    supersede: bool = False
    acknowledged_sensitive: bool = False
    project: str = ""


@dataclass(frozen=True)
class WriteDecision:
    """What the policy decided. `source`/`confidence` are the provenance a
    persisted write (or a candidate) carries. `existing_value` is what was
    already known for the key, if anything."""

    destination: Destination
    reason: str
    sensitivity: Sensitivity = Sensitivity.NONE
    source: EntitySource | None = None
    confidence: float = 0.0
    existing_value: str | None = None
    updates_existing: bool = False

    @property
    def persists(self) -> bool:
        return self.destination in (Destination.WORLD_MODEL, Destination.MEMORY)


@dataclass(frozen=True)
class WriteOutcome:
    """What actually happened, and what to tell the user (`message`, plain
    words: no internals)."""

    decision: WriteDecision
    persisted: bool
    message: str
    ref: str | None = None  # the id of the entity / candidate written, if any
