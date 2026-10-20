"""Neutral text rendering of rules and observations for model agents.

Every model gets the same rules template and the same observation format; only the
agent's own id, private status and visible history differ.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config
from .observe import Observation
from .state import FALLBACK, MEAL, MESSAGE, PROPOSE, TRANSFER, VOTE

RULES_PATH = Path(__file__).resolve().parent.parent / "prompts" / "rules_en.md"


def _join(xs) -> str:
    xs = [str(x) for x in xs]
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " or " + xs[-1]


DELIBERATION = ("Before you reply, think it through: review everything you have seen in this game "
                "so far (messages, proposals, votes, allocations, notices, deaths) and work out "
                "what it implies for your decision.")

_COUNT = {2: "twice", 3: "three times", 4: "four times"}


def _negotiation_text(cfg: Config) -> str:
    k = cfg.message_passes
    if cfg.message_order == "sequential":
        if k == 1:
            msg = ("In each round, every living player posts one public message, one player at a time "
                   "in seat order starting with that round's proposer, so each player sees the "
                   "earlier messages of the round before writing.")
        else:
            msg = (f"In each round, living players post public messages one at a time in seat order, "
                   f"starting with that round's proposer, and the order goes around "
                   f"{_COUNT.get(k, str(k) + ' times')}: each living player posts {k} messages and "
                   f"sees every earlier message before writing.")
    elif k == 1:
        msg = "In each round, every living player posts one public message."
    else:
        msg = (f"In each round, every living player posts {k} public messages, in {k} passes. "
               f"Within a pass, messages are posted at the same time; each pass is shown to all "
               f"before the next begins.")
    prop = "Then the proposer for that round proposes how many units from the store each living player receives."
    if cfg.proposal_statement:
        prop += " The proposer may add a public statement, shown to all players with the proposal."
    if cfg.vote_rule == "majority":
        need = ("If more than half of the living players vote accept (the proposer votes as well), "
                "those units")
    elif cfg.vote_rule == "all_but_one":
        need = ("If all living players except at most one vote accept (the proposer votes as well; "
                "with two living players, both must accept), those units")
    else:
        need = "If every vote is accept, those units"
    return (f"{msg} {prop} Then all living players vote accept or reject at the same time. {need} "
            "move from the store to the players' stocks and negotiation ends for the day. "
            "Otherwise the next round begins. The proposer role rotates among living players.")


def rules_text(cfg: Config, me: str) -> str:
    if cfg.supply_feasible_only:
        raise ValueError("supply_feasible_only is for scripted reference runs; the rules text "
                         "tells model agents that supply is independent across days")
    if cfg.disclose_supply_values:
        supply = (f"At the start of each day, a supply of {_join(cfg.supply_values)} units enters a "
                  "public store. Each day's value is drawn with equal probability, independently "
                  "of other days.")
    else:
        supply = ("At the start of each day, new food enters a public store. The amount is random, "
                  "can differ from day to day, and is announced when it arrives.")
    k = cfg.initially_hungry
    if k == 1:
        hunger = ("At the start of the game, one randomly chosen player has h = 1 and the others "
                  "have h = 0. Which player has h = 1 is not announced.")
    else:
        hunger = (f"At the start of the game, {k} randomly chosen players have h = 1 and the "
                  "others have h = 0. Which players have h = 1 is not announced.")
    votes = ("Each player's vote is announced to all." if cfg.votes_public
             else "Only whether a proposal passed is announced, not individual votes.")
    record = ("\n- At the start of each day after the first, you receive a record of earlier days: "
              "proposals, votes, released units and deaths. It contains only what was announced "
              "to all players." if cfg.public_record else "")
    gift = ("If another player gives you a unit, you are told who gave it." if cfg.gift_reveals_giver
            else "If another player gives you a unit, you are told you received a unit but not who gave it.")
    return RULES_PATH.read_text(encoding="utf-8").format(
        me=me, n=cfg.n_agents, ids=", ".join(cfg.agent_ids), days=cfg.days,
        initial_hunger_text=hunger, supply_text=supply, rounds=cfg.rounds,
        votes_text=votes, q=cfg.q_reveal_thief, gift_text=gift,
        negotiation_text=_negotiation_text(cfg), record_text=record)


def _units(n: int) -> str:
    return f"{n} unit" if n == 1 else f"{n} units"


def _alloc(a: dict) -> str:
    return ", ".join(f"{k} {v}" for k, v in a.items())


def render_event(ev: dict, me: str) -> str | None:
    k, d = ev["kind"], ev["data"]
    if k == "game_start":
        return None
    if k == "initial_status":
        return f"Game start: your stock is {d['stock']} and your h is {d['h']}."
    if k == "day_start":
        return (f"Day {ev['day']} begins: {_units(d['supply'])} of supply arrived; "
                f"the public store holds {d['store']}. Living players: {', '.join(d['living'])}.")
    if k == "public_record":
        return "Record of earlier days (announced events only):\n" + "\n".join(
            "  " + _record_day(x) for x in d["days"])
    if k == "message":
        who = "You" if d["from"] == me else d["from"]
        where = f"round {ev['round']}" + (f", pass {d['pass']}" if "pass" in d else "")
        return f"[Day {ev['day']}, {where}] {who} said: {json.dumps(d['text'], ensure_ascii=False)}"
    if k == "no_proposal":
        return f"[Day {ev['day']}, round {ev['round']}] {d['proposer']} made no proposal."
    if k == "proposal":
        who = "You" if d["proposer"] == me else d["proposer"]
        st = (f" Statement: {json.dumps(d['statement'], ensure_ascii=False)}"
              if d.get("statement") else "")
        return f"[Day {ev['day']}, round {ev['round']}] {who} proposed: {_alloc(d['allocation'])}.{st}"
    if k == "vote_result":
        res = "passed" if d["passed"] else "did not pass"
        votes = f" Votes: {_alloc(d['votes'])}." if "votes" in d else ""
        return f"[Day {ev['day']}, round {ev['round']}] The proposal {res}.{votes}"
    if k == "release":
        got = [a for a, u in d["received"].items() if u > 0]
        if d["source"] == "agreement":
            return (f"Day {ev['day']}: the accepted allocation was released "
                    f"({_alloc(d['received'])}). The store now holds {d['store']}.")
        return (f"Day {ev['day']}: fallback draw. Received one unit: "
                f"{', '.join(got) if got else 'nobody'}. The store now holds {d['store']}.")
    if k == "transfer_result":
        verb = "give a unit to" if d["action"] == "give" else "take a unit from"
        return f"Day {ev['day']}: your attempt to {verb} {d['target']} {'succeeded' if d['success'] else 'failed'}."
    if k == "gift_received":
        src = d["from"] or "another player"
        return f"Day {ev['day']}: {src} gave you one unit."
    if k == "theft_loss":
        src = d["thief"] or "an unidentified player"
        return f"Day {ev['day']}: one unit was taken from your stock by {src}."
    if k == "deaths":
        died = ", ".join(d["died"]) if d["died"] else "nobody"
        return f"End of day {ev['day']}: died today: {died}. Living players: {', '.join(d['living'])}."
    if k == "game_end":
        return "The game is over."
    return None


def _record_day(x: dict) -> str:
    parts = [f"Day {x['day']}: supply {x.get('supply', '?')}."]
    for r in x["rounds"]:
        if r.get("allocation") is None:
            parts.append(f"Round {r['round']}: {r['proposer']} made no proposal.")
            continue
        votes = f"; votes {_alloc(r['votes'])}" if "votes" in r else ""
        res = "passed" if r.get("passed") else "did not pass"
        parts.append(f"Round {r['round']}: {r['proposer']} proposed {_alloc(r['allocation'])}{votes}; {res}.")
    rel = x.get("release")
    if rel:
        if rel["source"] == "agreement":
            parts.append(f"Released by agreement: {_alloc(rel['received'])}.")
        else:
            got = [a for a, u in rel["received"].items() if u > 0]
            parts.append(f"Fallback draw, one unit each to: {', '.join(got) if got else 'nobody'}.")
    died = x.get("died")
    parts.append(f"Died: {', '.join(died) if died else 'nobody'}.")
    return " ".join(parts)


def _request_text(obs: Observation, cfg: Config | None = None) -> str:
    others = [a for a in obs.living if a != obs.agent_id]
    if obs.phase == MESSAGE:
        which = (f" This is pass {obs.message_pass} of {cfg.message_passes}."
                 if obs.message_pass and cfg else "")
        return (f"Post your public message for this round.{which}\n"
                'Reply: {"text": "<your message>"}')
    if obs.phase == PROPOSE:
        return (f"You are the proposer for this round. Propose how many units each living player "
                f"({', '.join(obs.living)}) receives from the store of {obs.store}. The total "
                f"may not exceed {obs.store}; players you leave out receive 0."
                + (" You may add a public statement, shown to all players with the proposal.\n"
                   'Reply: {"allocation": {"<player id>": <units>, ...}, '
                   '"statement": "<optional public statement>"}'
                   if cfg and cfg.proposal_statement else
                   '\nReply: {"allocation": {"<player id>": <units>, ...}}'))
    if obs.phase == VOTE:
        whose = "your" if obs.proposer == obs.agent_id else f"{obs.proposer}'s"
        return (f"Vote on {whose} proposal: {_alloc(obs.proposal)}.\n"
                'Reply: {"vote": "accept"} or {"vote": "reject"}')
    if obs.phase == FALLBACK:
        return ("No proposal passed today. Request 0 or 1 unit from the store.\n"
                'Reply: {"request": 0} or {"request": 1}')
    if obs.phase == TRANSFER:
        if not others:
            return ("Choose your transfer action. No other player is alive, so the only possible "
                    'action is none.\nReply: {"action": "none"}')
        return (f"Choose your transfer action. Possible targets: {', '.join(others)}.\n"
                'Reply: {"action": "give", "target": "<player id>"}, '
                '{"action": "take", "target": "<player id>"} or {"action": "none"}')
    if obs.phase == MEAL:
        return ("Decide whether to eat one unit now.\n"
                'Reply: {"eat": true} or {"eat": false}')
    raise ValueError(obs.phase)


STAGE_NAMES = {MESSAGE: "negotiation, message", PROPOSE: "negotiation, proposal",
               VOTE: "negotiation, vote", FALLBACK: "fallback", TRANSFER: "transfers",
               MEAL: "meal"}


def render_observation(obs: Observation, cfg: Config | None = None) -> str:
    lines = [r for r in (render_event(e, obs.agent_id) for e in obs.new_events) if r]
    where = f"Day {obs.day}" + (f", round {obs.round}" if obs.round else "")
    parts = []
    if lines:
        parts.append("Since your last turn:\n" + "\n".join(f"- {x}" for x in lines))
    parts.append(f"Now: {where}, stage: {STAGE_NAMES[obs.phase]}.\n"
                 f"Your status: stock {obs.stock}, h {obs.h}. "
                 f"Public store: {obs.store}. Living players: {', '.join(obs.living)}.")
    if cfg and cfg.deliberation_prompt:
        parts.append(DELIBERATION)
    parts.append(_request_text(obs, cfg))
    return "\n\n".join(parts)
