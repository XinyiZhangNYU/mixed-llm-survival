"""The single decision interface shared by model agents and scripted agents."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Config
from ..observe import Observation


@dataclass
class Decision:
    action: object                 # engine action dict (or None for "no proposal")
    error: str | None = None       # set when the reply could not be parsed
    raw: str | None = None         # raw model text, if any
    reasoning: str = ""
    meta: dict = field(default_factory=dict)   # tokens, latency, cost, model id


class Agent:
    """Subclasses implement `decide` (async) and/or `decide_sync`.

    `key` is one of message, propose, vote, request, transfer, meal.
    `feedback` is the validation error from the previous attempt, or None.
    """
    label: str = "agent"
    scripted: bool = False

    def start_game(self, agent_id: str, cfg: Config, seed: int) -> None:
        self.agent_id, self.cfg, self.seed = agent_id, cfg, seed

    async def decide(self, obs: Observation, key: str, feedback: str | None = None) -> Decision:
        return self.decide_sync(obs, key, feedback)

    def decide_sync(self, obs: Observation, key: str, feedback: str | None = None) -> Decision:
        raise NotImplementedError(f"{type(self).__name__} has no synchronous decide")
