"""Scripted references and a Baron-Ferejohn benchmark under the frozen rules (pilot_n5_v2).

All five seats are scripted. Seeds 1-3 give the same supply and initial hunger as the LLM
pilot games pilot_n5_v2_*_seed1..3, so those rows are matched references.

1. Scripted rosters (sim/agents/scripted.py): need-first ("always truthful"), claim-hungry,
   thief, all-request lottery, equal split. The roster is rotated by seed (seed mod N), so a
   liar or thief does not always sit in seat A, which need-first proposers serve first.
2. Ex-post planner bound: knows the supply sequence and every h; maximises player-days alive.
   Feeding an h = 0 player is never better than storing the unit, so the planner only chooses
   how many h = 1 players to feed each day.
3. Baron-Ferejohn benchmark for one day's negotiation: players value units linearly, rounds
   R, proposer rotates, a proposal passes with votes_needed(n) accepts, and the fallback is the
   all-request lottery (each living player gets 1 unit with probability min(1, S/n)). Units are
   whole, so buying a vote costs ceil(continuation value). Food left in the store has no value
   in this one-day model, which ignores storage, hunger and theft: it is an interpretive
   benchmark, not a prediction for the multi-day game.
"""
import json
import math
import statistics as st
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim.agents import (ClaimHungryAgent, EqualSplitAgent, LotteryAgent,  # noqa: E402
                        NeedFirstAgent, ThiefAgent)
from sim.config import Config  # noqa: E402
from sim.engine import new_game, votes_needed  # noqa: E402
from sim.runner import run_game_sync  # noqa: E402

CFG = Config.from_yaml(ROOT / "configs" / "pilot_n5_v2.yaml")
SEEDS = range(200)
MATCHED = (1, 2, 3)
N, T = CFG.n_agents, CFG.days

ROSTERS = {
    "all truthful (need-first)": lambda: [NeedFirstAgent() for _ in range(N)],
    "1 claim-hungry": lambda: [ClaimHungryAgent()] + [NeedFirstAgent() for _ in range(N - 1)],
    "2 claim-hungry": lambda: [ClaimHungryAgent() for _ in range(2)] + [NeedFirstAgent() for _ in range(N - 2)],
    "all claim-hungry": lambda: [ClaimHungryAgent() for _ in range(N)],
    "1 thief (random target)": lambda: [ThiefAgent("random")] + [NeedFirstAgent() for _ in range(N - 1)],
    "1 thief (today's receiver)": lambda: [ThiefAgent("recent_receiver")] + [NeedFirstAgent() for _ in range(N - 1)],
    "equal split": lambda: [EqualSplitAgent() for _ in range(N)],
    "all lottery": lambda: [LotteryAgent() for _ in range(N)],
}


def planner(seed: int) -> float:
    s = new_game(CFG, seed)
    supply = s.supply
    n1 = sum(a.h for a in s.agents.values())

    @lru_cache(maxsize=None)
    def best(day, store, h0, h1):
        if day == T:
            return 0
        store += supply[day]
        return max(h0 + x + best(day + 1, store - x, x, h0) for x in range(min(h1, store) + 1))
    return best(0, 0, N - n1, n1) / (N * T)


def bf_day(S: int, n: int, rounds: int = CFG.rounds) -> tuple[list[float], bool]:
    """Expected units for each seat (seat 0 proposes in round 1, seat 1 in round 2, ...),
    and whether the round-1 proposal is accepted."""
    k = votes_needed(CFG, n) - 1                      # accepts needed besides the proposer
    v = [min(1.0, S / n)] * n                         # fallback lottery
    agreed = False
    for r in reversed(range(rounds)):
        p = r % n
        others = [j for j in range(n) if j != p]
        cost = {j: math.ceil(v[j] - 1e-9) for j in others}
        order = sorted(others, key=cost.get)
        total = sum(cost[j] for j in order[:k])
        agreed = not (total > S or S - total < v[p])
        if not agreed:
            continue                                  # no acceptable proposal: next round
        cut = cost[order[k - 1]] if k else None
        sure = [j for j in others if k and cost[j] < cut]
        tied = [j for j in others if k and cost[j] == cut]
        prob = (k - len(sure)) / len(tied) if tied else 0.0
        nv = [0.0] * n
        nv[p] = S - total
        for j in sure:
            nv[j] = cost[j]
        for j in tied:
            nv[j] = prob * cost[j]
        v = nv
    return v, agreed


def main():
    print(f"Config: configs/pilot_n5_v2.yaml (N={N}, T={T}, R={CFG.rounds}, supply {CFG.supply_values}, "
          f"vote {CFG.vote_rule}, k={CFG.initially_hungry}); {len(SEEDS)} seeds\n")
    print("Supply on matched seeds:", {sd: new_game(CFG, sd).supply for sd in MATCHED}, "\n")

    print(f"{'roster':<28} {'surv':>5} {'allAlive':>8} {'deaths':>6} {'prioGap':>7} | seeds 1/2/3 surv")
    rows = {}
    for name, make in ROSTERS.items():
        surv, alive, deaths, gap, matched = [], [], [], [], {}
        for seed in SEEDS:
            agents = make()
            r = seed % N
            agents = agents[-r:] + agents[:-r] if r else agents
            res = run_game_sync(CFG, dict(zip(CFG.agent_ids, agents)), seed)
            surv.append(res["survival"]); alive.append(res["all_alive"]); deaths.append(len(res["deaths"]))
            if res["priority_gap"] is not None:
                gap.append(res["priority_gap"])
            if seed in MATCHED:
                matched[seed] = res["survival"]
        rows[name] = matched
        print(f"{name:<28} {st.mean(surv):>5.2f} {st.mean(alive):>8.2f} {st.mean(deaths):>6.2f} "
              f"{st.mean(gap):>7.2f} | " + " / ".join(f"{matched[s]:.2f}" for s in MATCHED))
    pl = [planner(s) for s in SEEDS]
    print(f"{'ex-post planner bound':<28} {st.mean(pl):>5.2f} {st.mean(p == 1 for p in pl):>8.2f} "
          f"{'':>6} {'':>7} | " + " / ".join(f"{planner(s):.2f}" for s in MATCHED))

    print("\nLLM pilot games on the same seeds (survival, priority gap):")
    for run in ["pilot_n5_v2_cocoo_seed1", "pilot_n5_v2_cocoo_seed2", "pilot_n5_v2_cocoo_seed3",
                "pilot_n5_v2_ddddd_seed1"]:
        r = json.loads((ROOT / "runs" / run / "result.json").read_text())
        print(f"  {run:<26} {r['survival']:.2f}  {r['priority_gap']:.2f}")

    print("\nBaron-Ferejohn one-day benchmark: proposer's units / store, round-1 agreement")
    print("  n\\S " + " ".join(f"{S:>6}" for S in range(1, 9)))
    for n in (5, 4, 3, 2):
        cells = []
        for S in range(1, 9):
            v, agree = bf_day(S, n)
            cells.append(f"{v[0]:.0f}/{S}" if agree else "fallbk")
        print(f"  {n:>3} " + " ".join(f"{c:>6}" for c in cells))

    print("\nObserved proposals vs the benchmark (proposer units; players given 0):")
    for run in ["pilot_n5_v2_cocoo_seed1", "pilot_n5_v2_cocoo_seed2", "pilot_n5_v2_cocoo_seed3",
                "pilot_n5_v2_ddddd_seed1"]:
        ev = [json.loads(l) for l in open(ROOT / "runs" / run / "events.jsonl", encoding="utf-8")]
        store = {e["day"]: e["data"]["store"] for e in ev if e["kind"] == "day_start"}
        out = []
        for e in ev:
            if e["kind"] == "proposal":
                al, p, S = e["data"]["allocation"], e["data"]["proposer"], store[e["day"]]
                n = len(al)
                v, agree = bf_day(S, n)
                pred = f"{v[0]:.0f}" if agree else "fb"
                zeros = sum(1 for a, u in al.items() if u == 0 and a != p)
                out.append(f"d{e['day']} S{S} n{n}: {al.get(p, 0)} vs {pred}, left {S - sum(al.values())}, zeros {zeros}")
        print(f"  {run}:\n    " + "\n    ".join(out))


if __name__ == "__main__":
    main()
