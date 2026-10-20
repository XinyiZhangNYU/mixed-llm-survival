"""Game rules from the proposal, the model-agent retry path, and reply parsing."""
import asyncio
import json

import pytest

from sim import engine
from sim.agents import FakeProvider, LLMAgent, LotteryAgent, NeedFirstAgent
from sim.config import Config
from sim.render import render_observation, rules_text
from sim.rng import draw_supply, feasible, need_by
from sim.runner import run_game, run_game_sync
from sim.schemas import parse_reply
from sim.state import FALLBACK, MEAL, MESSAGE, TRANSFER


def fresh(cfg=None, seed=0):
    return engine.new_game(cfg or Config(), seed)


def all_act(s, action):
    return {r.agent: action for r in engine.pending(s)}


def negotiate(s, proposal, votes):
    engine.apply(s, all_act(s, {"text": ""}))
    engine.apply(s, {engine.current_proposer(s): proposal})
    if proposal is not None:
        engine.apply(s, {a: {"vote": votes(a)} for a in s.living()})


def test_need_and_feasibility_match_proposal_figures():
    assert need_by(15, 5, 2) == 37
    assert feasible([2] * 15, 5, 2) is False
    cfg = Config(supply_feasible_only=True)
    for seed in range(50):
        seq = draw_supply(cfg, seed)
        assert feasible(seq, 5, 2) and set(seq) <= {2, 3, 4}


def test_unanimity_release_and_carry_over():
    s = fresh()
    store0 = s.store
    alloc = {"allocation": {"A": 1}}
    negotiate(s, alloc, lambda a: "reject" if a == "B" else "accept")
    assert s.phase == MESSAGE and s.round == 2 and s.store == store0      # one reject blocks
    negotiate(s, alloc, lambda a: "accept")
    assert s.phase == TRANSFER and s.agents["A"].stock == 1
    assert s.store == store0 - 1                                           # rest stays in store
    engine.apply(s, all_act(s, {"action": "none", "target": None}))
    engine.apply(s, {"A": {"eat": False}})
    assert s.day == 2 and s.store == store0 - 1 + s.supply[1]             # carried over


def test_rounds_exhaust_to_fallback_and_lottery_rules():
    s = fresh(Config(rounds=2))
    for _ in range(2):
        negotiate(s, None, None)
    assert s.phase == FALLBACK
    store = s.store
    engine.apply(s, {a: {"request": 1} for a in s.living()})
    rel = next(e for e in reversed(s.events) if e.kind == "release")
    assert sum(rel.data["received"].values()) == min(store, 5)
    assert set(rel.data["received"].values()) <= {1}
    assert s.phase == TRANSFER


def test_fallback_without_excess_fills_every_request():
    s = fresh(Config(rounds=1))
    negotiate(s, None, None)
    engine.apply(s, {a: {"request": int(a in "AB")} for a in s.living()})
    rel = next(e for e in reversed(s.events) if e.kind == "release")
    assert rel.data["received"] == {"A": 1, "B": 1}


def test_hunger_and_death_boundaries():
    s = fresh(Config(rounds=1))
    hungry = [a for a in s.living() if s.agents[a].h == 1]
    assert len(hungry) == 2
    negotiate(s, {"allocation": {hungry[0]: 1}}, lambda a: "accept")
    engine.apply(s, all_act(s, {"action": "none", "target": None}))
    assert s.phase == MEAL and [r.agent for r in engine.pending(s)] == [hungry[0]]
    engine.apply(s, {hungry[0]: {"eat": True}})
    assert s.agents[hungry[0]].alive and s.agents[hungry[0]].h == 0
    assert not s.agents[hungry[1]].alive and s.agents[hungry[1]].died_day == 1
    deaths = next(e for e in reversed(s.events) if e.kind == "deaths")
    assert deaths.visible_to == "all" and deaths.data["died"] == [hungry[1]]
    assert all(s.agents[a].h == 1 for a in s.living() if a != hungry[0])


def test_proposer_rotates_and_skips_dead():
    s = fresh(Config(rounds=3))
    order = []
    for _ in range(3):
        order.append(engine.current_proposer(s))
        negotiate(s, None, None)
    ids = s.cfg.agent_ids
    i = ids.index(order[0])
    assert order == [ids[(i + k) % 5] for k in range(3)]
    s.agents[ids[(i + 3) % 5]].alive = False
    s.proposer_counter = (i + 3) % 5
    assert engine.current_proposer(s) == ids[(i + 4) % 5]


def test_rejected_proposal_passes_role_to_next_player():
    s = fresh()
    first = engine.current_proposer(s)
    negotiate(s, {"allocation": {"A": 1}}, lambda a: "reject")
    ids = s.cfg.agent_ids
    assert s.round == 2 and engine.current_proposer(s) == ids[(ids.index(first) + 1) % 5]


def test_illegal_actions_rejected():
    s = fresh()
    engine.apply(s, all_act(s, {"text": ""}))
    p = engine.current_proposer(s)
    assert engine.validate(s, p, {"allocation": {"A": s.store + 1}})
    assert engine.validate(s, p, {"allocation": {"A": -1}})
    assert engine.validate(s, p, {"allocation": {"Z": 1}})
    with pytest.raises(ValueError):
        engine.apply(s, {p: {"allocation": {"A": s.store + 1}}})


def test_parse_reply():
    act, why = parse_reply("transfer", 'ok ```json\n{"action": "take", "target": "B", "reasoning": "x"}\n```')
    assert act == {"action": "steal", "target": "B"} and why == "x"
    assert parse_reply("vote", '{"vote": "accept"}')[0] == {"vote": "accept"}
    with pytest.raises(ValueError):
        parse_reply("vote", '{"vote": "maybe"}')
    with pytest.raises(ValueError):
        parse_reply("meal", "I will eat")


def test_rules_and_observation_render():
    cfg = Config()
    text = rules_text(cfg, "C")
    assert "player C" in text and "{" not in text.split("Replies")[0]
    assert "agent" not in text.lower() and "allocated perfectly" not in text
    assert "2, 3 or 4" not in text and "announced when it arrives" in text
    shown = rules_text(Config(disclose_supply_values=True, initially_hungry=1), "C")
    assert "2, 3 or 4 units" in shown and "independently of other days" in shown
    assert "one randomly chosen player has h = 1" in shown
    with pytest.raises(ValueError):
        rules_text(Config(supply_feasible_only=True), "C")
    s = fresh(cfg)
    from sim.observe import observe
    out = render_observation(observe(s, "A"))
    assert "Your status: stock 0" in out and '"text"' in out


def test_llm_agent_retry_then_default(tmp_path):
    cfg = Config(days=2, rounds=1)
    # A answers garbage once, then validly; B is always invalid; others are scripted.
    good = {"message": '{"text": "hi"}', "propose": '{"allocation": {}}', "vote": '{"vote": "reject"}',
            "request": '{"request": 1}', "transfer": '{"action": "none"}', "meal": '{"eat": true}'}

    def a_reply(messages):
        # the very first request gets an unparseable reply; retries and later requests are valid
        return "not json" if len(messages) == 2 else _good_for(messages, good)

    roster = {x: LotteryAgent() for x in cfg.agent_ids}
    roster["A"] = LLMAgent(FakeProvider(a_reply, "fake-a"))
    roster["B"] = LLMAgent(FakeProvider(["nonsense"], "fake-b"))
    res = asyncio.run(run_game(cfg, roster, 0, log_dir=tmp_path))
    calls = [json.loads(x) for x in (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    a_calls = [c for c in calls if c.get("agent") == "A" and "attempt" in c]
    assert a_calls[0]["error"] and a_calls[1]["error"] is None          # retry recovered
    b_defaults = [c for c in calls if c.get("agent") == "B" and "default_used" in c]
    assert b_defaults and len([c for c in calls if c.get("agent") == "B" and "attempt" in c]) \
        == len(b_defaults) * (cfg.max_retries + 1)
    assert res["state"].done


def _good_for(messages, good):
    for m in reversed(messages):
        if m["role"] == "user" and "Reply:" in m["content"]:
            c = m["content"]
            for k, marker in [("message", '"text"'), ("propose", '"allocation"'), ("vote", '"vote"'),
                              ("request", '"request"'), ("transfer", '"action"'), ("meal", '"eat"')]:
                if marker in c.split("Reply:")[-1]:
                    return good[k]
    raise AssertionError("no request found")


def test_sync_and_reference_agents_smoke():
    cfg = Config(supply_feasible_only=True)
    r = run_game_sync(cfg, {a: NeedFirstAgent() for a in cfg.agent_ids}, 5)
    assert r["survival"] == 1.0 and r["all_alive"]


def test_cli_flatten_keeps_history_and_current_request():
    from sim.agents.cli import flatten
    msgs = [{"role": "system", "content": "RULES"},
            {"role": "user", "content": "req1"}, {"role": "assistant", "content": "rep1"},
            {"role": "user", "content": "req2"}]
    out = flatten(msgs)
    assert "RULES" not in out and out.endswith("--- CURRENT REQUEST ---\n\nreq2")
    assert out.index("req1") < out.index("rep1") < out.index("req2")
    assert flatten(msgs[:2]) == "req1"


def test_cost_cap_stops_between_steps_and_replays(tmp_path):
    from sim.runner import load_checkpoint, replay
    good = {"message": '{"text": "hi"}', "propose": '{"allocation": {}}', "vote": '{"vote": "reject"}',
            "request": '{"request": 1}', "transfer": '{"action": "none"}', "meal": '{"eat": true}'}

    class Priced(FakeProvider):
        async def complete(self, messages):
            text, _ = await super().complete(messages)
            return text, {"cost_usd": 0.1}

    cfg = Config(days=3, rounds=1)
    roster = {a: LLMAgent(Priced(lambda m: _good_for(m, good))) for a in cfg.agent_ids}
    res = asyncio.run(run_game(cfg, roster, 0, log_dir=tmp_path, cost_cap_usd=1.25))
    assert res["stopped"] == "cost_cap" and res["survival"] is None
    assert res["cost_usd"] <= 1.25 and not res["state"].done
    assert load_checkpoint(tmp_path).to_dict() == res["state"].to_dict() == replay(tmp_path).to_dict()
    saved = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert saved["stopped"] == "cost_cap" and saved["stopped_at"]["day"] == res["state"].day


def test_configs_render_for_model_agents():
    from pathlib import Path
    for p in sorted((Path(__file__).resolve().parents[1] / "configs").glob("*.yaml")):
        cfg = Config.from_yaml(p)
        assert not cfg.supply_feasible_only, p.name
        assert "allocated perfectly" not in rules_text(cfg, cfg.agent_ids[0])


def test_render_wording_for_own_proposal_units_and_sole_survivor():
    from sim.observe import Observation
    from sim.render import render_event, render_observation
    ev = {"kind": "day_start", "day": 2, "data": {"supply": 1, "store": 3, "living": ["A", "B"]}}
    assert "1 unit of supply" in render_event(ev, "A")
    ev = {"kind": "proposal", "day": 2, "round": 1, "data": {"proposer": "A", "allocation": {"A": 1}}}
    assert render_event(ev, "A").endswith("You proposed: A 1.")
    base = dict(agent_id="A", day=2, round=1, stock=0, h=0, store=1, new_events=[])
    from sim.state import TRANSFER, VOTE
    obs = Observation(**{**base, "phase": VOTE, "living": ["A", "B"], "proposer": "A",
                         "proposal": {"A": 1, "B": 0}})
    assert "Vote on your proposal" in render_observation(obs)
    obs = Observation(**{**base, "round": None, "phase": TRANSFER, "living": ["A"],
                         "proposer": None, "proposal": None})
    assert "only possible action is none" in render_observation(obs)


def _starve_day(s):
    negotiate(s, {"allocation": {}}, lambda a: "accept")
    engine.apply(s, all_act(s, {"action": "none", "target": None}))
    if s.phase == MEAL:
        engine.apply(s, all_act(s, {"eat": False}))


def test_game_stops_with_one_survivor_who_counts_as_alive():
    from sim.metrics import normalized_survival
    from sim.state import GAME_OVER
    s = fresh(Config(n_agents=2, initially_hungry=1, days=4, rounds=1, supply_values=(1, 2)))
    _starve_day(s)
    assert s.phase == GAME_OVER and s.day == 1 and len(s.living()) == 1
    end = s.events[-1]
    assert end.kind == "game_end"
    assert end.data == {"survivors": s.living(), "reason": "sole_survivor", "days_played": 1}
    assert s.alive_end_of_day == [1, 1, 1, 1] and normalized_survival(s) == 0.5
    assert engine.pending(s) == []


def test_game_continues_with_one_survivor_when_supply_can_be_zero():
    from sim.state import GAME_OVER
    s = fresh(Config(n_agents=2, initially_hungry=1, days=4, rounds=1, supply_values=(0, 1)))
    _starve_day(s)
    assert len(s.living()) == 1 and s.phase != GAME_OVER and s.day == 2


def test_priority_gap_skips_days_with_one_living_player():
    from types import SimpleNamespace as E

    from sim.metrics import priority_gap
    day1 = [E(kind="truth_day_start", day=1, data={"at_risk": ["A"], "store": 2, "alive": {"A": True, "B": True}}),
            E(kind="release", day=1, data={"received": {"A": 1}})]
    # Day 2 has a lone player (a game logged before the sole-survivor stop played such days).
    day2 = [E(kind="truth_day_start", day=2, data={"at_risk": [], "store": 1, "alive": {"A": True, "B": False}}),
            E(kind="release", day=2, data={"received": {"A": 1}})]
    # Day 1 only: 1 of 1 released units reached an at-risk player; the lottery sends 1 of 2.
    assert priority_gap(E(events=day1)) == pytest.approx(0.5)
    assert priority_gap(E(events=day1 + day2)) == pytest.approx(0.5)
