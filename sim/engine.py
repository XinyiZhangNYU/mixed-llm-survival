"""Deterministic game engine: the single source of truth.

Daily flow:
  DAY_START (automatic: supply enters the store)
  NEGOTIATE, up to R rounds: MESSAGE (one or more passes) -> PROPOSE -> VOTE
    (enough accept votes, per cfg.vote_rule, release the allocation)
  FALLBACK (only if no proposal passed): every living agent requests 0 or 1
  TRANSFER: give / steal / none, settled on the pre-transfer stock snapshot
  MEAL: agents holding food choose whether to eat
  HUNGER / DAY_END (automatic): h update, deaths announced, dead stock sealed

Public API: new_game, pending, validate, apply. Actions are plain dicts; `apply` collects a
whole phase's actions at once, so the order in which agents answered cannot matter.
`apply` mutates the state in place and returns the events it produced.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .rng import draw_supply, stream
from .state import (FALLBACK, GAME_OVER, MEAL, MESSAGE, PHASE_KEY, PROPOSE, TRANSFER,
                    VOTE, AgentState, Event, GameState)


@dataclass(frozen=True)
class Request:
    agent: str
    phase: str

    @property
    def key(self) -> str:
        return PHASE_KEY[self.phase]


# ---------------------------------------------------------------- setup

def new_game(cfg: Config, seed: int) -> GameState:
    ids = cfg.agent_ids
    supply = draw_supply(cfg, seed)
    hungry = set(stream(seed, "initial_hunger").sample(ids, cfg.initially_hungry))
    agents = {a: AgentState(a, stock=0, h=1 if a in hungry else 0) for a in ids}
    s = GameState(cfg=cfg, seed=seed, supply=supply, agents=agents)
    s.proposer_counter = stream(seed, "proposer_start").randrange(cfg.n_agents)
    s.emit("game_start", {
        "agent_ids": ids, "days": cfg.days, "rounds": cfg.rounds,
        "supply_values": list(cfg.supply_values),
        "supply_feasible_only": cfg.supply_feasible_only,
        "initially_hungry": cfg.initially_hungry,
        "q_reveal_thief": cfg.q_reveal_thief,
    }, "all", phase="GAME_START")
    for a in ids:
        s.emit("initial_status", {"stock": 0, "h": agents[a].h}, [a], phase="GAME_START")
    s.emit("truth_game_start", {"h": {a: agents[a].h for a in ids}, "supply": supply},
           [], phase="GAME_START")
    _start_day(s, 1)
    return s


def _proposer_seat(s: GameState) -> int:
    ids = s.cfg.agent_ids
    n = len(ids)
    for k in range(n):
        idx = (s.proposer_counter + k) % n
        if s.agents[ids[idx]].alive:
            return idx
    raise RuntimeError("no living agent")


def votes_needed(cfg: Config, n: int) -> int:
    """Accept votes needed to pass a proposal when n players are alive."""
    if cfg.vote_rule == "majority":
        return n // 2 + 1
    if cfg.vote_rule == "all_but_one":
        return max(n - 1, n // 2 + 1)
    return n


def current_proposer(s: GameState) -> str:
    return s.cfg.agent_ids[_proposer_seat(s)]


def _start_day(s: GameState, day: int) -> None:
    s.day = day
    if s.cfg.public_record and day > 1:
        s.emit("public_record", {"days": _public_record(s)}, "all", phase="DAY_START")
    x = s.supply[day - 1]
    s.store += x
    s.emit("day_start", {"supply": x, "store": s.store, "living": s.living()}, "all",
           phase="DAY_START")
    s.emit("truth_day_start", _truth(s), [], phase="DAY_START")
    _start_round(s, 1)


def _start_round(s: GameState, r: int) -> None:
    s.round = r
    s.phase = MESSAGE
    s.proposal = None
    s.message_pass = 1
    if s.cfg.message_order == "sequential":
        s.message_queue = _message_order(s)


def _message_order(s: GameState) -> list[str]:
    """Living agents in seat order, starting with the round's proposer."""
    ids = s.cfg.agent_ids
    start = _proposer_seat(s)
    order = [ids[(start + k) % len(ids)] for k in range(len(ids))]
    return [a for a in order if s.agents[a].alive]


def _public_record(s: GameState) -> list[dict]:
    """Per earlier day, what was announced to all: supply, proposals and votes, the release
    and deaths. Messages and proposal statements are left out (they stay in the history)."""
    days: dict[int, dict] = {}
    for e in s.events:
        if e.visible_to != "all" or e.day >= s.day:
            continue
        d = days.setdefault(e.day, {"day": e.day, "rounds": []})
        if e.kind == "day_start":
            d["supply"] = e.data["supply"]
        elif e.kind == "proposal":
            d["rounds"].append({"round": e.round, "proposer": e.data["proposer"],
                                "allocation": e.data["allocation"]})
        elif e.kind == "no_proposal":
            d["rounds"].append({"round": e.round, "proposer": e.data["proposer"],
                                "allocation": None, "passed": False})
        elif e.kind == "vote_result":
            d["rounds"][-1].update({k: v for k, v in e.data.items() if k in ("passed", "votes")})
        elif e.kind == "release":
            d["release"] = {"source": e.data["source"], "received": e.data["received"]}
        elif e.kind == "deaths":
            d["died"] = e.data["died"]
    return [days[k] for k in sorted(days) if k >= 1]


def _truth(s: GameState) -> dict:
    return {
        "store": s.store,
        "stock": {a: st.stock for a, st in s.agents.items()},
        "h": {a: st.h for a, st in s.agents.items()},
        "alive": {a: st.alive for a, st in s.agents.items()},
        "at_risk": [a for a in s.living() if s.agents[a].h == 1 and s.agents[a].stock == 0],
    }


# ---------------------------------------------------------------- queries

def pending(s: GameState) -> list[Request]:
    """Agents who must act now. Never empty unless the game is over."""
    if s.done:
        return []
    if s.phase == MESSAGE and s.cfg.message_order == "sequential":
        return [Request(s.message_queue[0], MESSAGE)]
    if s.phase == PROPOSE:
        return [Request(current_proposer(s), PROPOSE)]
    if s.phase == MEAL:
        return [Request(a, MEAL) for a in s.living() if s.agents[a].stock >= 1]
    return [Request(a, s.phase) for a in s.living()]


def validate(s: GameState, agent: str, action) -> str | None:
    """Return an error message if `action` is not legal for `agent` now, else None."""
    living = s.living()
    if agent not in living:
        return f"{agent} is not a living player"
    if s.phase == PROPOSE:
        if action is None:
            return None
        alloc = action.get("allocation") if isinstance(action, dict) else None
        if not isinstance(alloc, dict):
            return "allocation must be an object mapping player ids to units"
        for k, v in alloc.items():
            if k not in living:
                return f"{k} is not a living player"
            if not isinstance(v, int) or isinstance(v, bool) or v < 0:
                return f"units for {k} must be a nonnegative integer"
        total = sum(alloc.values())
        if total > s.store:
            return f"allocation totals {total} but the store holds {s.store}"
        if s.cfg.allocation_must_exhaust_store and total != s.store:
            return f"allocation must total exactly {s.store}"
        st = action.get("statement", "")
        if st:
            if not s.cfg.proposal_statement:
                return "this game has no proposal statements; leave statement out"
            if not isinstance(st, str):
                return "statement must be a string"
            if len(st) > s.cfg.max_message_chars:
                return f"statement longer than {s.cfg.max_message_chars} characters"
        return None
    if not isinstance(action, dict):
        return "action must be an object"
    if s.phase == MESSAGE:
        t = action.get("text")
        if not isinstance(t, str):
            return "text must be a string"
        if len(t) > s.cfg.max_message_chars:
            return f"message longer than {s.cfg.max_message_chars} characters"
        return None
    if s.phase == VOTE:
        return None if action.get("vote") in ("accept", "reject") else "vote must be accept or reject"
    if s.phase == FALLBACK:
        r = action.get("request")
        return None if r in (0, 1) and not isinstance(r, bool) else "request must be 0 or 1"
    if s.phase == TRANSFER:
        kind, target = action.get("action"), action.get("target")
        if kind == "none":
            return None
        if kind not in ("give", "steal"):
            return "action must be give, steal or none"
        if target == agent:
            return "target must be another player"
        if target not in living:
            return "target must be a living player"
        return None
    if s.phase == MEAL:
        return None if isinstance(action.get("eat"), bool) else "eat must be true or false"
    return f"no actions accepted in phase {s.phase}"


# ---------------------------------------------------------------- transitions

def apply(s: GameState, actions: dict[str, object]) -> list[Event]:
    """Settle one phase given every pending agent's action."""
    reqs = pending(s)
    expected = {r.agent for r in reqs}
    if set(actions) != expected:
        raise ValueError(f"expected actions from {sorted(expected)}, got {sorted(actions)}")
    for a in sorted(actions):
        err = validate(s, a, actions[a])
        if err:
            raise ValueError(f"illegal action from {a}: {err}")
    first = len(s.events)
    s.step += 1
    {MESSAGE: _apply_message, PROPOSE: _apply_propose, VOTE: _apply_vote,
     FALLBACK: _apply_fallback, TRANSFER: _apply_transfer, MEAL: _apply_meal}[s.phase](s, actions)
    return s.events[first:]


def _apply_message(s: GameState, actions) -> None:
    for a in sorted(actions):
        data = {"from": a, "text": actions[a]["text"]}
        if s.cfg.message_passes > 1:
            data["pass"] = s.message_pass
        s.emit("message", data, "all")
    if s.cfg.message_order == "sequential":
        s.message_queue.pop(0)
        if s.message_queue:
            return
    if s.message_pass < s.cfg.message_passes:
        s.message_pass += 1
        if s.cfg.message_order == "sequential":
            s.message_queue = _message_order(s)
        return
    s.phase = PROPOSE


def _end_round_without_agreement(s: GameState) -> None:
    s.proposer_counter = _proposer_seat(s) + 1
    if s.round < s.cfg.rounds:
        _start_round(s, s.round + 1)
    else:
        s.phase = FALLBACK
        s.proposal = None


def _apply_propose(s: GameState, actions) -> None:
    (p, action), = actions.items()
    if action is None:
        s.emit("no_proposal", {"proposer": p}, "all")
        _end_round_without_agreement(s)
        return
    alloc = {a: action["allocation"].get(a, 0) for a in s.living()}
    s.proposal = alloc
    data = {"proposer": p, "allocation": alloc}
    if action.get("statement"):
        data["statement"] = action["statement"]
    s.emit("proposal", data, "all")
    s.phase = VOTE


def _apply_vote(s: GameState, actions) -> None:
    votes = {a: actions[a]["vote"] for a in sorted(actions)}
    accepts = sum(v == "accept" for v in votes.values())
    passed = accepts >= votes_needed(s.cfg, len(votes))
    data = {"passed": passed}
    if s.cfg.votes_public:
        data["votes"] = votes
    s.emit("vote_result", data, "all")
    if not passed:
        _end_round_without_agreement(s)
        return
    s.proposer_counter = _proposer_seat(s) + 1
    alloc = s.proposal
    for a, units in alloc.items():
        s.agents[a].stock += units
    s.store -= sum(alloc.values())
    s.emit("release", {"source": "agreement", "received": alloc, "store": s.store}, "all")
    s.proposal = None
    s.phase = TRANSFER


def _apply_fallback(s: GameState, actions) -> None:
    requesters = [a for a in sorted(actions) if actions[a]["request"] == 1]
    if len(requesters) <= s.store:
        winners = requesters
    else:
        winners = sorted(stream(s.seed, "fallback", s.day).sample(requesters, s.store))
    for a in winners:
        s.agents[a].stock += 1
    s.store -= len(winners)
    s.emit("truth_fallback_requests", {"requesters": requesters}, [])
    s.emit("release", {"source": "fallback", "received": {a: 1 for a in winners},
                       "store": s.store}, "all")
    s.phase = TRANSFER


def settle_transfers(stock: dict[str, int], acts: dict[str, dict], seed: int, day: int):
    """Settle simultaneous give/steal actions against the pre-transfer snapshot.

    All claims on one agent's stock (its own gift and thefts from it) have equal priority;
    if they exceed that stock, a uniform draw picks which succeed. Units received here
    cannot fund another transfer in the same stage because only the snapshot is used.
    Returns (final stock dict, list of successful (actor, kind, source, recipient)).
    """
    claims: dict[str, list[tuple[str, str, str, str]]] = {a: [] for a in stock}
    for actor in sorted(acts):
        act = acts[actor]
        kind, target = act.get("action"), act.get("target")
        if kind == "give":
            claims[actor].append((actor, "give", actor, target))
        elif kind == "steal":
            claims[target].append((actor, "steal", target, actor))
    final = dict(stock)
    done = []
    for source in sorted(claims):
        pool = claims[source]
        cap = stock[source]
        if len(pool) > cap:
            pool = stream(seed, "contention", day, source).sample(pool, cap)
            pool.sort()
        for actor, kind, src, recipient in pool:
            final[src] -= 1
            final[recipient] += 1
            done.append((actor, kind, src, recipient))
    done.sort()
    return final, done


def _apply_transfer(s: GameState, actions) -> None:
    before = {a: s.agents[a].stock for a in s.living()}
    final, done = settle_transfers(before, actions, s.seed, s.day)
    for a, v in final.items():
        s.agents[a].stock = v
    succeeded = {actor for actor, *_ in done}
    for a in sorted(actions):
        act = actions[a]
        if act["action"] != "none":
            s.emit("transfer_result", {"action": act["action"], "target": act["target"],
                                       "success": a in succeeded}, [a])
    for actor, kind, src, recipient in done:
        if kind == "give":
            giver = actor if s.cfg.gift_reveals_giver else None
            s.emit("gift_received", {"from": giver}, [recipient])
        else:
            reveal = stream(s.seed, "theft_reveal", s.day, actor, src).random() < s.cfg.q_reveal_thief
            s.emit("theft_loss", {"thief": actor if reveal else None}, [src])
    s.emit("truth_transfers", {"actions": {a: actions[a] for a in sorted(actions)},
                               "successful": [list(d) for d in done],
                               "stock_before": before, "stock_after": final}, [])
    s.phase = MEAL
    if not pending(s):
        _apply_meal(s, {})


def _apply_meal(s: GameState, actions) -> None:
    living = s.living()
    eaters = {a for a in actions if actions[a]["eat"]}
    for a in living:
        st = s.agents[a]
        if a in eaters:
            st.stock -= 1
            st.h = 0
        else:
            st.h += 1
    s.emit("truth_meals", {"ate": sorted(eaters)}, [])
    died = [a for a in living if s.agents[a].h >= 2]
    for a in died:
        s.agents[a].alive = False
        s.agents[a].died_day = s.day          # stock stays with the dead agent, sealed
    s.emit("deaths", {"died": died, "living": s.living()}, "all", phase="DAY_END")
    s.emit("truth_day_end", _truth(s), [], phase="DAY_END")
    living = s.living()
    s.alive_end_of_day.append(len(living))
    # A lone survivor has nobody to bargain with or take from, and with every supply value >= 1
    # can eat each remaining day, so the game stops and that player counts as alive to day T.
    sole = (s.cfg.n_agents > 1 and len(living) == 1 and min(s.cfg.supply_values) >= 1)
    if s.day >= s.cfg.days or not living or sole:
        reason = "last_day" if s.day >= s.cfg.days else "sole_survivor" if living else "all_dead"
        s.alive_end_of_day += [len(living)] * (s.cfg.days - s.day)
        s.phase = GAME_OVER
        s.emit("game_end", {"survivors": living, "reason": reason, "days_played": s.day}, "all",
               phase=GAME_OVER)
        return
    _start_day(s, s.day + 1)
