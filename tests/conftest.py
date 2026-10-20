import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.agents import LotteryAgent, NeedFirstAgent, ThiefAgent  # noqa: E402
from sim.agents.base import Agent  # noqa: E402


class Recorder(Agent):
    """Wraps an agent and keeps every observation it was shown."""

    def __init__(self, inner):
        self.inner, self.label, self.scripted = inner, inner.label, inner.scripted
        self.seen = []

    def start_game(self, agent_id, cfg, seed):
        super().start_game(agent_id, cfg, seed)
        self.inner.start_game(agent_id, cfg, seed)

    def decide_sync(self, obs, key, feedback=None):
        self.seen.append((key, obs))
        return self.inner.decide_sync(obs, key, feedback)


def mixed_roster(cfg, thief_seat="C"):
    """Thief plus need-first and lottery agents: exercises proposals, votes, fallback and theft."""
    roster = {}
    for i, a in enumerate(cfg.agent_ids):
        if a == thief_seat:
            roster[a] = ThiefAgent(target_rule="recent_receiver", vote="accept")
        elif i % 2:
            roster[a] = LotteryAgent()
        else:
            roster[a] = NeedFirstAgent()
    return roster
