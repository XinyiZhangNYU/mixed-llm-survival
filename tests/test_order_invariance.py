"""Results do not depend on the order in which agents answer or actions are listed."""
import asyncio
import copy
import random

from conftest import mixed_roster

from sim import engine
from sim.agents.base import Agent
from sim.config import Config
from sim.runner import run_game, run_game_sync


class SlowAgent(Agent):
    """Answers after a random delay, so completion order differs from call order."""

    def __init__(self, inner, rng):
        self.inner, self.rng, self.label = inner, rng, inner.label

    def start_game(self, agent_id, cfg, seed):
        super().start_game(agent_id, cfg, seed)
        self.inner.start_game(agent_id, cfg, seed)

    async def decide(self, obs, key, feedback=None):
        await asyncio.sleep(self.rng.random() * 0.002)
        return self.inner.decide_sync(obs, key, feedback)


def _events(s):
    return [(e.kind, e.data, e.visible_to) for e in s.events]


def test_apply_ignores_action_order():
    cfg = Config()
    for seed in range(10):
        s = engine.new_game(cfg, seed)
        roster = mixed_roster(cfg)
        for a, ag in roster.items():
            ag.start_game(a, cfg, seed)
        from sim.observe import observe
        while not s.done:
            reqs = engine.pending(s)
            acts = {r.agent: roster[r.agent].decide_sync(observe(s, r.agent), r.key).action for r in reqs}
            keys = list(acts)
            random.Random(seed + s.step).shuffle(keys)
            s2 = copy.deepcopy(s)
            engine.apply(s, acts)
            engine.apply(s2, {k: acts[k] for k in keys})
            assert s.to_dict() == s2.to_dict()


def test_async_completion_order_does_not_matter():
    cfg = Config()
    for seed in range(5):
        base = run_game_sync(cfg, mixed_roster(cfg), seed)["state"]
        rng = random.Random(seed)
        slow = {a: SlowAgent(ag, rng) for a, ag in mixed_roster(cfg).items()}
        got = asyncio.run(run_game(cfg, slow, seed))["state"]
        assert _events(got) == _events(base)
        assert got.to_dict() == base.to_dict()


def test_common_random_numbers_across_rosters():
    """Different behavior must not shift supply or initial hunger."""
    from sim.agents import LotteryAgent, NeedFirstAgent
    cfg = Config()
    for seed in range(5):
        a = run_game_sync(cfg, {x: LotteryAgent() for x in cfg.agent_ids}, seed)["state"]
        b = run_game_sync(cfg, {x: NeedFirstAgent() for x in cfg.agent_ids}, seed)["state"]
        ta = next(e for e in a.events if e.kind == "truth_game_start").data
        tb = next(e for e in b.events if e.kind == "truth_game_start").data
        assert ta == tb
