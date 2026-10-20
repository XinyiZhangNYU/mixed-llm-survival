"""Game state and events.

Every piece of information the engine produces is an Event with a visibility list.
`visible_to == "all"` is public; a list names the only agents who see it; an empty list
is ground truth kept for analysis and never shown to any agent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Union

from .config import Config

Visibility = Union[str, list]  # "all" or list of agent ids

# Phases that wait for agent actions.
MESSAGE, PROPOSE, VOTE, FALLBACK, TRANSFER, MEAL = (
    "MESSAGE", "PROPOSE", "VOTE", "FALLBACK", "TRANSFER", "MEAL")
GAME_OVER = "GAME_OVER"
ACTION_PHASES = (MESSAGE, PROPOSE, VOTE, FALLBACK, TRANSFER, MEAL)

# Phase -> key used for schemas and invalid-output defaults.
PHASE_KEY = {MESSAGE: "message", PROPOSE: "propose", VOTE: "vote",
             FALLBACK: "request", TRANSFER: "transfer", MEAL: "meal"}


@dataclass
class Event:
    seq: int
    day: int
    phase: str
    round: int | None
    kind: str
    data: dict
    visible_to: Visibility

    def visible(self, agent_id: str) -> bool:
        return self.visible_to == "all" or agent_id in self.visible_to


@dataclass
class AgentState:
    id: str
    stock: int = 0
    h: int = 0
    alive: bool = True
    died_day: int | None = None


@dataclass
class GameState:
    cfg: Config
    seed: int
    supply: list[int]
    agents: dict[str, AgentState]
    day: int = 0
    phase: str = MESSAGE
    round: int = 1
    store: int = 0
    proposer_counter: int = 0          # advances once per negotiation round
    proposal: dict | None = None       # allocation under vote
    message_queue: list[str] = field(default_factory=list)  # sequential message order
    message_pass: int = 1              # current message pass within the round
    events: list[Event] = field(default_factory=list)
    alive_end_of_day: list[int] = field(default_factory=list)
    step: int = 0                      # number of applied action batches

    @property
    def done(self) -> bool:
        return self.phase == GAME_OVER

    def living(self) -> list[str]:
        return [a for a in self.cfg.agent_ids if self.agents[a].alive]

    def proposer(self) -> str:
        living = self.living()
        return living[self.proposer_counter % len(living)]

    def emit(self, kind: str, data: dict, visible_to: Visibility, phase: str | None = None) -> Event:
        ev = Event(len(self.events), self.day, phase or self.phase,
                   self.round if (phase or self.phase) in (MESSAGE, PROPOSE, VOTE) else None,
                   kind, data, visible_to)
        self.events.append(ev)
        return ev

    # Serialization for checkpoints and replay comparison.
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cfg"] = self.cfg.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "GameState":
        d = dict(d)
        d["cfg"] = Config.from_dict(d["cfg"])
        d["agents"] = {k: AgentState(**v) for k, v in d["agents"].items()}
        d["events"] = [Event(**e) for e in d["events"]]
        return cls(**d)
