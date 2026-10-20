"""Survival-utility Baron-Ferejohn benchmark under the frozen rules (pilot_n5_v2).

The linear one-day benchmark (baselines/references_n5.py) values every unit equally and
gives food left in the store no value. Here each player's payoff is the rules' own objective:
the expected number of days, from today to day T, at whose end the player is alive.

Model of one day:
- Complete information about today's store, every h and every stock; no transfers; a player
  eats only when h = 1 and holds a unit (eating at h = 0 leaves h at 0, so holding is never
  worse).
- R rounds, the proposer rotates over living seats as in the engine, a proposal passes with
  votes_needed(n) accepts (the proposer accepts its own proposal), and a responder accepts
  when the proposal gives it at least the value of rejecting. After R rejections every living
  player requests 1 unit in the fallback lottery (a unit in one's own stock is never worth less
  to oneself than the same unit in the store, since nothing can be taken).
- The proposer picks, among the proposals that pass, one that maximises its own payoff, if
  that is at least what it gets when the round fails. A selfish proposer is often indifferent
  (for example on the last day, when it is safe), so two tie-breaks are reported:
    efficient: the largest sum of the other players' payoffs, then fewest units released;
    minimal:   fewest units released.
  Remaining ties are a uniform draw (responders' values are averaged over that draw).
- Days after today follow a fixed continuation, the same for every player:
    need-first: each day one unit to each living player at h = 1 with no stock, a uniform
                draw if the store is short, the rest stays in the store (this is the policy
                that reaches the ex-post planner bound);
    lottery:    each day every living player requests 1 unit in the fallback lottery.
  Supply is i.i.d. uniform on the configured values. Bargaining is strategic only today, so
  this is a one-day benchmark, not an equilibrium of the whole game.

Outputs: an illustrative table, predictions for every observed LLM proposal from that day's
true state, and a 200-seed run in which every day is played by this benchmark.
"""
import json
import random
import statistics as st
import sys
from functools import lru_cache
from itertools import combinations, product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim import engine  # noqa: E402
from sim.config import Config  # noqa: E402
from sim.metrics import summarize  # noqa: E402
from sim.state import FALLBACK, MEAL, MESSAGE, PROPOSE, TRANSFER, VOTE  # noqa: E402

CFG = Config.from_yaml(ROOT / "configs" / "pilot_n5_v2.yaml")
N, T, R = CFG.n_agents, CFG.days, CFG.rounds
SUPPLY = tuple(CFG.supply_values)
IDS = CFG.agent_ids
EPS = 1e-9
ALPHA = 0.0   # weight each player puts on every other player's expected days (sensitivity only)
RUNS = ["pilot_n5_v2_cocoo_seed1", "pilot_n5_v2_cocoo_seed2", "pilot_n5_v2_cocoo_seed3",
        "pilot_n5_v2_ddddd_seed1"]

# A profile is a tuple over seats of (alive, h, stock).


def meals_left(t: int, h: int) -> int:
    """Most meals a player at hunger h on the morning of day t can still need through day T."""
    return 1 + (T - t) // 2 if h == 1 else (T - t + 1) // 2


def end_of_day(t: int, prof: tuple) -> tuple[tuple, tuple]:
    """Meal and hunger update. Returns the next morning's profile and who is alive tonight.
    Stock beyond what a player can still eat is dropped: it has no value to anyone."""
    out, alive = [], []
    for a, h, s in prof:
        if not a:
            out.append((0, 0, 0)); alive.append(0); continue
        if h == 1 and s >= 1:
            h, s = 0, s - 1
        else:
            h += 1
        if h >= 2:
            out.append((0, 0, 0)); alive.append(0); continue
        out.append((1, h, min(s, meals_left(t + 1, h)) if t < T else s)); alive.append(1)
    return tuple(out), tuple(alive)


def draw(cands: list[int], S: int) -> list[tuple[float, tuple]]:
    """Distribution of who receives a unit when each candidate asks for one."""
    if len(cands) <= S:
        rec = tuple(int(i in cands) for i in range(N))
        return [(1.0, rec)]
    subs = list(combinations(cands, S))
    return [(1.0 / len(subs), tuple(int(i in c) for i in range(N))) for c in subs]


def release(policy: str, S: int, prof: tuple):
    if policy == "need-first":
        cands = [i for i, (a, h, s) in enumerate(prof) if a and h == 1 and s == 0]
    else:
        cands = [i for i, (a, h, s) in enumerate(prof) if a]
    return draw(cands, S)


def add(prof, rec):
    return tuple((a, h, s + r) for (a, h, s), r in zip(prof, rec))


@lru_cache(maxsize=None)
def future(t: int, store: int, prof: tuple, policy: str) -> tuple:
    """Expected days alive per seat from day t to T under the continuation policy."""
    if t > T or not any(a for a, _, _ in prof):
        return (0.0,) * N
    tot = [0.0] * N
    for x in SUPPLY:
        S = store + x
        for p, rec in release(policy, S, prof):
            nxt, alive = end_of_day(t, add(prof, rec))
            fut = future(t + 1, S - sum(rec), nxt, policy)
            w = p / len(SUPPLY)
            for i in range(N):
                tot[i] += w * (alive[i] + fut[i])
    return tuple(tot)


def outcome(t, S, prof, rec, policy):
    """Payoffs if `rec` is released today from a store of S."""
    nxt, alive = end_of_day(t, add(prof, rec))
    fut = future(t + 1, S - sum(rec), nxt, policy)
    return tuple(alive[i] + fut[i] for i in range(N))


def mean(vecs):
    return tuple(sum(v[i] for v in vecs) / len(vecs) for i in range(N))


@lru_cache(maxsize=None)
def allocations(t, S, prof, policy):
    living = [i for i, (a, _, _) in enumerate(prof) if a]
    caps = [min(S, max(0, meals_left(t, prof[i][1]) - prof[i][2])) for i in living]
    out = []
    for xs in product(*(range(c + 1) for c in caps)):
        if sum(xs) <= S:
            rec = [0] * N
            for i, x in zip(living, xs):
                rec[i] = x
            rec = tuple(rec)
            out.append((rec, outcome(t, S, prof, rec, policy)))
    return out


def next_living(prof, seat):
    for k in range(1, N + 1):
        j = (seat + k) % N
        if prof[j][0]:
            return j
    return seat


def val(U):
    """Payoff each player acts on: own expected days plus ALPHA times everyone else's."""
    tot = sum(U)
    return tuple(u + ALPHA * (tot - u) for u in U)


def solve(t: int, S: int, prof: tuple, proposer: int, rounds_left: int, policy: str,
          tie: str = "efficient"):
    """Rounds from now to the fallback. Returns one entry per round:
    (proposer seat, agreed, tied allocations, payoff vector at that round)."""
    living = [i for i, (a, _, _) in enumerate(prof) if a]
    seats = [proposer]
    for _ in range(rounds_left - 1):
        seats.append(next_living(prof, seats[-1]))
    allocs = allocations(t, S, prof, policy)
    W = mean([outcome(t, S, prof, rec, policy) for _, rec in release("lottery", S, prof)]) \
        if len(living) > S else outcome(t, S, prof, release("lottery", S, prof)[0][1], policy)
    k = engine.votes_needed(CFG, len(living)) - 1
    plan = []
    for p in reversed(seats):
        others = [j for j in living if j != p]
        Wv = val(W)
        ok = [(rec, U) for rec, U in allocs if sum(val(U)[j] >= Wv[j] - EPS for j in others) >= k]
        best = max((val(U)[p] for _, U in ok), default=None)
        if best is not None and best >= Wv[p] - EPS:
            ties = [(rec, U) for rec, U in ok if val(U)[p] >= best - EPS]
            if tie == "efficient":
                top = max(sum(U) - U[p] for _, U in ties)
                ties = [(rec, U) for rec, U in ties if sum(U) - U[p] >= top - EPS]
            fewest = min(sum(rec) for rec, _ in ties)
            ties = [(rec, U) for rec, U in ties if sum(rec) == fewest]
            W = mean([U for _, U in ties])
            plan.append((p, True, ties, W))
        else:
            plan.append((p, False, [], W))
    return plan[::-1]


# ---------------------------------------------------------------- reports

def profile_from_truth(d):
    return tuple((int(d["alive"][a]), d["h"][a] if d["alive"][a] else 0,
                  d["stock"][a] if d["alive"][a] else 0) for a in IDS)


def illustrative(policy, tie):
    """Day 4 of 8, all five alive, no stocks, m players at h = 1: proposer's units and store
    left in round 1 for store sizes 1..8 (the proposer is an h = 0 player when m < 5)."""
    print(f"\n[{policy}, {tie}] day 4, five alive, no stocks; cell = proposer units / units released / store left"
          f" ('fb' = no round-1 agreement)")
    print("  m\\S  " + " ".join(f"{S:>9}" for S in range(1, 9)))
    for m in range(0, 6):
        prof = tuple((1, int(i >= N - m), 0) for i in range(N))      # hungry seats at the end
        cells = []
        for S in range(1, 9):
            p0 = solve(4, S, prof, 0, R, policy, tie)[0]
            if not p0[1]:
                cells.append("fb"); continue
            rec = p0[2][0][0]
            own = sorted({r[0] for r, _ in p0[2]})
            cells.append(f"{own[0] if len(own) == 1 else f'{own[0]}-{own[-1]}'}/{sum(rec)}/{S - sum(rec)}")
        print(f"  {m:>3}  " + " ".join(f"{c:>9}" for c in cells))


def observed(policy, tie):
    print(f"\n[{policy}, {tie}] observed proposals vs benchmark from the same day's true state")
    print("  run day rnd S n | observed: own released h1-unfed | benchmark: agree own released h1-unfed")
    tally = {"n": 0, "own<": 0, "own=": 0, "own>": 0, "rel<": 0, "rel=": 0, "rel>": 0,
             "fb_pred": 0, "unfed_obs": 0, "unfed_pred": 0}
    for run in RUNS:
        ev = [json.loads(l) for l in open(ROOT / "runs" / run / "events.jsonl", encoding="utf-8")]
        truth = {e["day"]: e["data"] for e in ev if e["kind"] == "truth_day_start"}
        for e in ev:
            if e["kind"] != "proposal":
                continue
            d, r = e["day"], e["round"]
            prof = profile_from_truth(truth[d])
            S = truth[d]["store"]
            p = IDS.index(e["data"]["proposer"])
            al = e["data"]["allocation"]
            own_obs = al.get(IDS[p], 0)
            rel_obs = sum(al.values())
            risk = [i for i, (a, h, s) in enumerate(prof) if a and h == 1 and s == 0]
            unfed_obs = sum(1 for i in risk if al.get(IDS[i], 0) == 0)
            plan = solve(d, S, prof, p, R - r + 1, policy, tie)[0]
            tally["n"] += 1
            tally["unfed_obs"] += unfed_obs
            if not plan[1]:
                tally["fb_pred"] += 1
                pred = "fallback"
            else:
                owns = sorted({rec[p] for rec, _ in plan[2]})
                rel = sum(plan[2][0][0])
                unfed = st.mean(sum(1 for i in risk if rec[i] == 0) for rec, _ in plan[2])
                tally["unfed_pred"] += unfed
                lo, hi = owns[0], owns[-1]
                tally["own<" if own_obs < lo else "own>" if own_obs > hi else "own="] += 1
                tally["rel<" if rel_obs < rel else "rel>" if rel_obs > rel else "rel="] += 1
                pred = f"agree {lo if lo == hi else f'{lo}-{hi}'} {rel} {unfed:g}"
            short = run.replace("pilot_n5_v2_", "")
            print(f"  {short:<12} d{d} r{r} S{S} n{len(al)} | {own_obs} {rel_obs} {unfed_obs} | {pred}")
    print("  summary:", tally)


def play(seed, policy, tie, rng):
    """One game with every day decided by the benchmark (truthful, no theft)."""
    s = engine.new_game(CFG, seed)
    plan_cache = {}
    while not s.done:
        reqs = engine.pending(s)
        ph = s.phase
        acts = {}
        if ph == MESSAGE:
            acts = {r.agent: {"text": ""} for r in reqs}
        elif ph in (PROPOSE, VOTE):
            key = (s.day, s.round)
            if key not in plan_cache:
                prof = tuple((int(st_.alive), st_.h if st_.alive else 0, st_.stock if st_.alive else 0)
                             for st_ in (s.agents[a] for a in IDS))
                p = IDS.index(engine.current_proposer(s))
                pl = solve(s.day, s.store, prof, p, R - s.round + 1, policy, tie)
                cont = pl[1][3] if len(pl) > 1 else None
                if pl[0][1]:
                    rec, _ = rng.choice(pl[0][2])
                    alloc = {IDS[i]: rec[i] for i in range(N) if prof[i][0]}
                else:
                    alloc = None
                plan_cache[key] = (alloc, pl[0], cont)
            alloc, p0, cont = plan_cache[key]
            if ph == PROPOSE:
                acts = {reqs[0].agent: ({"allocation": alloc} if alloc is not None else None)}
            else:
                acts = {r.agent: {"vote": "accept"} for r in reqs}   # the plan already passes
        elif ph == FALLBACK:
            acts = {r.agent: {"request": 1} for r in reqs}
        elif ph == TRANSFER:
            acts = {r.agent: {"action": "none", "target": None} for r in reqs}
        elif ph == MEAL:
            acts = {r.agent: {"eat": s.agents[r.agent].h >= 1} for r in reqs}
        engine.apply(s, acts)
    return summarize(s)


def simulate(policy, tie, seeds=range(200)):
    rng = random.Random(20261018)
    res = [play(sd, policy, tie, rng) for sd in seeds]
    surv = [r["survival"] for r in res]
    gap = [r["priority_gap"] for r in res if r["priority_gap"] is not None]
    matched = {r["seed"]: r["survival"] for r in res if r["seed"] in (1, 2, 3)}
    print(f"\n[{policy}, {tie}] 200 seeds played by the benchmark: survival {st.mean(surv):.2f}, "
          f"all alive {st.mean(r['all_alive'] for r in res):.2f}, "
          f"deaths {st.mean(len(r['deaths']) for r in res):.2f}, priority gap {st.mean(gap):.2f}; "
          f"seeds 1/2/3: " + " / ".join(f"{matched[k]:.2f}" for k in (1, 2, 3)))


def main():
    global ALPHA
    if "--alpha" in sys.argv:                       # sensitivity: python baselines/su_bf.py --alpha 0.05
        ALPHA = float(sys.argv[sys.argv.index("--alpha") + 1])
        print(f"Sensitivity: ALPHA = {ALPHA} (each player also values every other player's expected days)")
        for policy in ("need-first", "lottery"):
            observed(policy, "efficient")
            simulate(policy, "efficient")
        return
    print(f"Config: configs/pilot_n5_v2.yaml (N={N}, T={T}, R={R}, supply {list(SUPPLY)} i.i.d., "
          f"vote {CFG.vote_rule})")
    for policy in ("need-first", "lottery"):
        for tie in ("efficient", "minimal"):
            illustrative(policy, tie)
            observed(policy, tie)
            simulate(policy, tie)


if __name__ == "__main__":
    main()
