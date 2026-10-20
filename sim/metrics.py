"""Game-level outcomes computed from the final state and ground-truth events."""
from __future__ import annotations

from .state import GameState


def normalized_survival(s: GameState) -> float:
    """Agent-days alive at the end of each day, divided by N*T (the earlier reference's metric).

    A game that stops early with one survivor counts that player as alive on the days not played.
    """
    return sum(s.alive_end_of_day) / (s.cfg.n_agents * s.cfg.days)


def priority_gap(s: GameState) -> float | None:
    """PROVISIONAL definition, to be fixed at the freeze.

    Share of units leaving the store (agreement or fallback) that reach agents truly at risk
    at the start of the day (h=1, no stock), minus the expected share under an all-request
    lottery on the same days: each day the lottery would release min(store, living) units
    uniformly over living agents, so it sends at_risk/living of them to at-risk agents.

    Only days that start with at least two living agents count: a lone agent has no one to
    be prioritized over. This also makes the value the same whether or not the engine ends a
    game early with one survivor (games logged before that change played those days).
    """
    at_risk, store, living = {}, {}, {}
    for e in s.events:
        if e.kind == "truth_day_start" and sum(e.data["alive"].values()) >= 2:
            at_risk[e.day] = set(e.data["at_risk"])
            store[e.day] = e.data["store"]
            living[e.day] = sum(e.data["alive"].values())
    released = to_risk = 0
    lot_released = lot_to_risk = 0.0
    for e in s.events:
        if e.kind == "release" and e.day in at_risk:
            units = sum(e.data["received"].values())
            released += units
            to_risk += sum(u for a, u in e.data["received"].items() if a in at_risk[e.day])
    for d in at_risk:
        m = min(store[d], living[d])
        lot_released += m
        lot_to_risk += m * len(at_risk[d]) / living[d] if living[d] else 0.0
    if released == 0 or lot_released == 0:
        return None
    return to_risk / released - lot_to_risk / lot_released


def summarize(s: GameState) -> dict:
    return {
        "seed": s.seed,
        "survival": normalized_survival(s),
        "all_alive": len(s.living()) == s.cfg.n_agents,
        "deaths": {a: st.died_day for a, st in s.agents.items() if not st.alive},
        "priority_gap": priority_gap(s),
        "steps": s.step,
    }
