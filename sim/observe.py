"""Per-agent observations: the only path from game state to an agent.

An observation contains the agent's own private status, public facts (day, store,
living agents, current proposer/proposal) and the events visible to that agent.
Ground-truth events (visible_to == []) and other agents' private events never pass.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from . import engine
from .state import MESSAGE, PROPOSE, VOTE, GameState


@dataclass
class Observation:
    agent_id: str
    day: int
    phase: str
    round: int | None
    stock: int
    h: int
    store: int
    living: list[str]
    proposer: str | None
    proposal: dict | None
    new_events: list[dict] = field(default_factory=list)
    last_seq: int = -1
    message_pass: int | None = None   # set in MESSAGE when a round has several passes

    def to_dict(self) -> dict:
        return asdict(self)


def visible_events(s: GameState, agent_id: str, since: int = -1) -> list[dict]:
    return [{"seq": e.seq, "day": e.day, "phase": e.phase, "round": e.round,
             "kind": e.kind, "data": e.data}
            for e in s.events[since + 1:] if e.visible(agent_id)]


def observe(s: GameState, agent_id: str, since: int = -1) -> Observation:
    me = s.agents[agent_id]
    negotiating = s.phase in (MESSAGE, PROPOSE, VOTE)
    return Observation(
        agent_id=agent_id,
        day=s.day,
        phase=s.phase,
        round=s.round if negotiating else None,
        stock=me.stock,
        h=me.h,
        store=s.store,
        living=s.living(),
        proposer=engine.current_proposer(s) if negotiating else None,
        proposal=dict(s.proposal) if s.phase == VOTE and s.proposal else None,
        new_events=visible_events(s, agent_id, since),
        last_seq=len(s.events) - 1,
        message_pass=s.message_pass if s.phase == MESSAGE and s.cfg.message_passes > 1 else None,
    )
