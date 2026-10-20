"""Private information reaches only the agents allowed to see it."""
import json

import pytest
from conftest import Recorder, mixed_roster

from sim.config import Config
from sim.render import render_observation
from sim.runner import run_game_sync

PRIVATE_KINDS = {"initial_status", "transfer_result", "gift_received", "theft_loss"}


@pytest.mark.parametrize("q", [0.0, 1.0])
@pytest.mark.parametrize("seed", range(6))
def test_observations_are_redacted(q, seed):
    cfg = Config(q_reveal_thief=q)
    roster = {a: Recorder(ag) for a, ag in mixed_roster(cfg).items()}
    res = run_game_sync(cfg, roster, seed)
    s = res["state"]
    for me, rec in roster.items():
        for key, obs in rec.seen:
            assert s.agents[me].alive or obs.day <= (s.agents[me].died_day or 99)
            for ev in obs.new_events:
                orig = s.events[ev["seq"]]
                assert orig.visible(me), (me, orig)
                assert not orig.kind.startswith("truth_")
                if orig.kind in PRIVATE_KINDS:
                    assert orig.visible_to == [me]
                if orig.kind == "theft_loss" and q == 0.0:
                    assert ev["data"]["thief"] is None
            # nothing in the observation names another agent's stock or hunger
            blob = json.dumps(obs.to_dict())
            assert '"h": {' not in blob and '"stock": {' not in blob
            render_observation(obs)  # every visible event renders


def test_dead_agents_are_never_asked_and_stock_stays_sealed():
    cfg = Config()
    for seed in range(20):
        roster = {a: Recorder(ag) for a, ag in mixed_roster(cfg).items()}
        s = run_game_sync(cfg, roster, seed)["state"]
        for a, st in s.agents.items():
            if st.alive:
                continue
            assert all(obs.day <= st.died_day for _, obs in roster[a].seen)
        # the only agent-visible event carrying a stock figure is one's own initial status
        for ev in s.events:
            if ev.visible_to != [] and "stock" in ev.data:
                assert ev.kind == "initial_status" and len(ev.visible_to) == 1


def test_theft_victim_always_learns_and_q_controls_identity():
    revealed = total = 0
    for seed in range(40):
        cfg = Config(q_reveal_thief=0.5)
        s = run_game_sync(cfg, mixed_roster(cfg), seed)["state"]
        steals = [d for e in s.events if e.kind == "truth_transfers"
                  for d in e.data["successful"] if d[1] == "steal"]
        losses = [e for e in s.events if e.kind == "theft_loss"]
        assert len(steals) == len(losses)
        total += len(losses)
        revealed += sum(e.data["thief"] is not None for e in losses)
    assert total > 30 and 0.3 < revealed / total < 0.7
