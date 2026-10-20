"""Transfer settlement: simultaneous gives and takes on the pre-transfer snapshot.

The engine samples one contention outcome per seed; across many seeds the set of
outcomes must equal the specification's set of possible outcomes.
"""
from itertools import product

import pytest

from sim.engine import settle_transfers

IDS = "ABCD"
WAIT = ("wait", None)


def acts_from(spec):
    out = {}
    for i, (kind, target) in enumerate(spec):
        if kind == "wait":
            out[IDS[i]] = {"action": "none", "target": None}
        else:
            out[IDS[i]] = {"action": "give" if kind == "give" else "steal", "target": IDS[target]}
    return out


CASES = [
    ("reciprocal_theft", (1, 1, 0, 0), [("take", 1), ("take", 0), WAIT, WAIT], {(1, 1, 0, 0)}, 2),
    ("two_thieves_one_unit", (0, 0, 1, 0), [("take", 2), ("take", 2), WAIT, WAIT],
     {(1, 0, 0, 0), (0, 1, 0, 0)}, 1),
    ("empty_reciprocal_theft", (0, 0, 0, 0), [("take", 1), ("take", 0), WAIT, WAIT], {(0, 0, 0, 0)}, 0),
    ("no_same_round_retheft", (1, 0, 0, 0), [("take", 1), ("take", 2), ("take", 0), WAIT],
     {(0, 0, 1, 0)}, 1),
    ("gift_competes_with_theft", (1, 0, 0, 0), [("give", 1), WAIT, ("take", 0), WAIT],
     {(0, 1, 0, 0), (0, 0, 1, 0)}, 1),
    ("received_food_cannot_fund_gift", (1, 0, 0, 0), [("give", 1), ("give", 2), WAIT, WAIT],
     {(0, 1, 0, 0)}, 1),
]


@pytest.mark.parametrize("name,stock,spec,expected,count", CASES, ids=[c[0] for c in CASES])
def test_four_actor_edge_cases(name, stock, spec, expected, count):
    stock_d = dict(zip(IDS, stock))
    seen = set()
    for seed in range(300):
        final, done = settle_transfers(stock_d, acts_from(spec), seed, day=1)
        seen.add(tuple(final[a] for a in IDS))
        assert len(done) == count
    assert seen == expected


def test_three_actor_invariants_exhaustive():
    ids = "ABC"
    menus = [[("none", None)] + [(k, t) for k in ("give", "steal") for t in ids if t != a]
             for a in ids]
    profiles = 0
    for stock in product(range(3), repeat=3):
        sd = dict(zip(ids, stock))
        for prof in product(*menus):
            profiles += 1
            acts = {a: {"action": k, "target": t} for a, (k, t) in zip(ids, prof)}
            for seed in range(4):
                final, done = settle_transfers(sd, acts, seed, day=1)
                assert sum(final.values()) == sum(stock)                      # conservation
                assert all(v >= 0 for v in final.values())                    # nonnegative
                assert len({d[0] for d in done}) == len(done)                 # one action each
                for src in ids:                                               # snapshot capacity
                    intended = sum((k == "give" and a == src) or (k == "steal" and t == src)
                                   for a, (k, t) in zip(ids, prof))
                    assert sum(d[2] == src for d in done) == min(sd[src], intended)
    assert profiles == 3375


def test_contention_is_roughly_uniform():
    stock = {"A": 0, "B": 0, "C": 1, "D": 0}
    acts = acts_from([("take", 2), ("take", 2), WAIT, WAIT])
    wins = sum(settle_transfers(stock, acts, seed, 1)[0]["A"] for seed in range(4000))
    assert abs(wins / 4000 - 0.5) < 0.03
