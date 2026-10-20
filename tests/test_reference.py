"""Scripted reference groups through the engine reproduce the earlier standalone simulation
(N=5, T=15, supply {2,3,4} feasible-only, k=2): need-first 1.000, lottery 0.745, equal floor 0.600.
The full 20000-game check is tools/reproduce_references.py; this is a fast subset."""
from sim.agents import EqualSplitAgent, LotteryAgent, NeedFirstAgent
from sim.config import Config
from sim.runner import run_game_sync

GAMES = 400


def mean_survival(make):
    cfg = Config(supply_feasible_only=True)
    return sum(run_game_sync(cfg, {a: make() for a in cfg.agent_ids}, s)["survival"]
               for s in range(GAMES)) / GAMES


def test_need_first_keeps_everyone_alive():
    assert mean_survival(NeedFirstAgent) == 1.0


def test_lottery_close_to_earlier_figure():
    assert abs(mean_survival(LotteryAgent) - 0.745) < 0.02


def test_equal_floor_degenerates():
    assert abs(mean_survival(lambda: EqualSplitAgent("floor")) - 0.600) < 0.01
