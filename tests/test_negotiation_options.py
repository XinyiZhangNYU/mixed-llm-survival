"""Negotiation options added after the first pilot games: vote rule, message passes,
proposal statements, the deliberation sentence and the public record."""
import json

import pytest
from conftest import Recorder, mixed_roster

from sim import engine
from sim.config import Config
from sim.render import DELIBERATION, render_observation, rules_text
from sim.runner import run_game_sync
from sim.schemas import parse_reply
from sim.state import MESSAGE, PROPOSE, TRANSFER, VOTE

V2 = dict(n_agents=4, days=6, rounds=2, message_order="sequential", message_passes=2,
          vote_rule="all_but_one", proposal_statement=True, deliberation_prompt=True,
          public_record=True)


@pytest.mark.parametrize("rule,need", [
    ("unanimous", {2: 2, 3: 3, 4: 4, 5: 5}),
    ("majority", {2: 2, 3: 2, 4: 3, 5: 3}),
    ("all_but_one", {2: 2, 3: 2, 4: 3, 5: 4}),
])
def test_votes_needed(rule, need):
    cfg = Config(vote_rule=rule)
    assert {n: engine.votes_needed(cfg, n) for n in need} == need


def _to_vote(s, proposal):
    while s.phase == MESSAGE:
        engine.apply(s, {r.agent: {"text": ""} for r in engine.pending(s)})
    engine.apply(s, {engine.current_proposer(s): proposal})
    assert s.phase == VOTE


def test_one_reject_passes_under_all_but_one_but_two_do_not():
    s = engine.new_game(Config(**V2), 0)
    _to_vote(s, {"allocation": {"A": 1}})
    engine.apply(s, {a: {"vote": "reject" if a == "B" else "accept"} for a in s.living()})
    assert s.phase == TRANSFER and s.agents["A"].stock == 1

    s = engine.new_game(Config(**V2), 0)
    _to_vote(s, {"allocation": {"A": 1}})
    engine.apply(s, {a: {"vote": "reject" if a in "BC" else "accept"} for a in s.living()})
    assert s.phase == MESSAGE and s.round == 2


def test_sequential_passes_start_with_proposer_and_repeat_order():
    s = engine.new_game(Config(**V2), 3)
    order = []
    while s.phase == MESSAGE:
        (r,) = engine.pending(s)
        order.append(r.agent)
        engine.apply(s, {r.agent: {"text": r.agent}})
    p = engine.current_proposer(s)
    ids = s.cfg.agent_ids
    first = [ids[(ids.index(p) + k) % 4] for k in range(4)]
    assert order == first + first and s.phase == PROPOSE
    passes = [e.data["pass"] for e in s.events if e.kind == "message"]
    assert passes == [1] * 4 + [2] * 4


def test_simultaneous_passes():
    s = engine.new_game(Config(message_passes=2), 0)
    for _ in range(2):
        assert s.phase == MESSAGE and len(engine.pending(s)) == s.cfg.n_agents
        engine.apply(s, {r.agent: {"text": ""} for r in engine.pending(s)})
    assert s.phase == PROPOSE


def test_statement_is_public_and_gated_by_config():
    s = engine.new_game(Config(**V2), 0)
    _to_vote(s, {"allocation": {"A": 1}, "statement": "A is hungry."})
    (ev,) = [e for e in s.events if e.kind == "proposal"]
    assert ev.visible_to == "all" and ev.data["statement"] == "A is hungry."

    s = engine.new_game(Config(), 0)
    engine.apply(s, {r.agent: {"text": ""} for r in engine.pending(s)})
    assert engine.validate(s, engine.current_proposer(s),
                           {"allocation": {"A": 1}, "statement": "x"}) is not None
    assert engine.validate(s, engine.current_proposer(s), {"allocation": {"A": 1}}) is None


def test_parse_proposal_statement():
    act, _ = parse_reply("propose", '{"allocation": {"A": 1}, "statement": "why"}')
    assert act == {"allocation": {"A": 1}, "statement": "why"}
    act, _ = parse_reply("propose", '{"allocation": {"A": 1}}')
    assert act == {"allocation": {"A": 1}}


@pytest.mark.parametrize("seed", range(4))
def test_public_record_matches_announced_events(seed):
    cfg = Config(**V2)
    roster = {a: Recorder(ag) for a, ag in mixed_roster(cfg).items()}
    s = run_game_sync(cfg, roster, seed)["state"]
    records = [e for e in s.events if e.kind == "public_record"]
    assert [e.day for e in records] == list(range(2, s.day + 1))
    last = records[-1].data["days"]
    assert [d["day"] for d in last] == list(range(1, records[-1].day))
    for d in last:
        rel = [e for e in s.events if e.kind == "release" and e.day == d["day"]][0]
        assert d["release"]["received"] == rel.data["received"]
        assert d["supply"] == s.supply[d["day"] - 1]
    assert "truth" not in json.dumps([e.data for e in records])
    # every seat sees the record and it renders
    for rec in roster.values():
        for key, obs in rec.seen:
            text = render_observation(obs, cfg)
            assert DELIBERATION in text
            if any(e["kind"] == "public_record" for e in obs.new_events):
                assert "Record of earlier days" in text


def test_rules_text_describes_options():
    t = rules_text(Config(**V2), "A")
    assert "goes around twice" in t and "except at most one vote accept" in t
    assert "public statement" in t and "record of earlier days" in t
    plain = rules_text(Config(), "A")
    assert "every living player posts one public message." in plain
    assert "If every vote is accept" in plain and "record of earlier days" not in plain
