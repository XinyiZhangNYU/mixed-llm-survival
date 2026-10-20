"""Scripted agents behind the same interface as model agents.

They read only their own Observation; none sees hidden state.

- LotteryAgent: never proposes, rejects every proposal, always requests 1 in fallback.
  A group of them reproduces the all-request lottery reference.
- EqualSplitAgent: proposes an equal split of the store and accepts every proposal.
- NeedFirstAgent: posts its true status in a fixed format, proposes one unit to each agent
  that reported h=1 and no food, and accepts every proposal. With all seats scripted and
  truthful this equals the full-information need-first reference.
- ClaimHungryAgent: like NeedFirstAgent, but always posts "STATUS h=1 stock=0" and always
  requests 1 in fallback, whatever its true status. It never steals.
- ThiefAgent: steals one unit every transfer stage by a fixed target rule; other stages
  follow LotteryAgent unless configured otherwise. Its exact policy is an open setting.

All of them eat only when h = 1 (the meal rule of the earlier reference simulation).
"""
from __future__ import annotations

import re

from ..observe import Observation
from ..rng import stream
from .base import Agent, Decision

STATUS_RE = re.compile(r"^STATUS h=(\d) stock=(\d+)$")


class ScriptedAgent(Agent):
    scripted = True

    def start_game(self, agent_id, cfg, seed):
        super().start_game(agent_id, cfg, seed)
        self.memory: list[dict] = []

    def decide_sync(self, obs: Observation, key: str, feedback=None) -> Decision:
        self.memory.extend(obs.new_events)
        return Decision(getattr(self, key)(obs))

    # defaults
    def message(self, obs):
        return {"text": ""}

    def propose(self, obs):
        return None

    def vote(self, obs):
        return {"vote": "accept"}

    def request(self, obs):
        return {"request": 1}

    def transfer(self, obs):
        return {"action": "none", "target": None}

    def meal(self, obs):
        return {"eat": obs.h >= 1}


class LotteryAgent(ScriptedAgent):
    label = "scripted:lottery"

    def vote(self, obs):
        return {"vote": "reject"}


class EqualSplitAgent(ScriptedAgent):
    """rounding: "floor" leaves the remainder in the store (the earlier reference);
    "floor_random_remainder" gives the remainder one unit each to random distinct agents."""
    label = "scripted:equal"

    def __init__(self, rounding: str = "floor"):
        assert rounding in ("floor", "floor_random_remainder")
        self.rounding = rounding

    def propose(self, obs):
        n = len(obs.living)
        q, r = divmod(obs.store, n)
        alloc = {a: q for a in obs.living}
        if self.rounding == "floor_random_remainder" and r:
            for a in stream(self.seed, "equal_remainder", obs.day, obs.round).sample(obs.living, r):
                alloc[a] += 1
        return {"allocation": alloc}


class NeedFirstAgent(ScriptedAgent):
    label = "scripted:need_first"

    def message(self, obs):
        return {"text": f"STATUS h={obs.h} stock={obs.stock}"}

    def propose(self, obs):
        needy = []
        for e in self.memory:
            if e["kind"] == "message" and e["day"] == obs.day and e["round"] == obs.round:
                m = STATUS_RE.match(e["data"]["text"])
                if m and int(m.group(1)) == 1 and int(m.group(2)) == 0:
                    needy.append(e["data"]["from"])
        alloc = {a: 0 for a in obs.living}
        left = obs.store
        for a in obs.living:              # seat order
            if a in needy and left > 0:
                alloc[a] = 1
                left -= 1
        return {"allocation": alloc}

    def request(self, obs):
        return {"request": int(obs.h == 1 and obs.stock == 0)}


class ClaimHungryAgent(NeedFirstAgent):
    label = "scripted:claim_hungry"

    def message(self, obs):
        return {"text": "STATUS h=1 stock=0"}

    def request(self, obs):
        return {"request": 1}


class ThiefAgent(LotteryAgent):
    """target_rule: "random" (uniform over other living agents) or "recent_receiver"
    (a uniformly chosen agent publicly seen receiving food today, else random)."""
    label = "scripted:thief"

    def __init__(self, target_rule: str = "random", vote: str = "reject"):
        assert target_rule in ("random", "recent_receiver")
        self.target_rule, self._vote = target_rule, vote

    def vote(self, obs):
        return {"vote": self._vote}

    def transfer(self, obs):
        others = [a for a in obs.living if a != self.agent_id]
        if not others:
            return {"action": "none", "target": None}
        pool = others
        if self.target_rule == "recent_receiver":
            got = {a for e in self.memory if e["kind"] == "release" and e["day"] == obs.day
                   for a, u in e["data"]["received"].items() if u > 0}
            pool = [a for a in others if a in got] or others
        r = stream(self.seed, "thief_target", self.agent_id, obs.day)
        return {"action": "steal", "target": r.choice(pool)}
