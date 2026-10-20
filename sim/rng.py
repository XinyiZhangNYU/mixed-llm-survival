"""Named random streams and supply sampling.

Every random draw comes from a stream keyed by (game seed, source name, day, sub-key).
Draws therefore do not depend on how many other draws happened before, so rosters that
behave differently still share supply, initial hunger, proposer start, fallback draws,
contention draws and theft-identity reveals (common random numbers).
"""
from __future__ import annotations

import hashlib
import json
import random


def stream(seed: int, name: str, *keys) -> random.Random:
    blob = json.dumps([seed, name, *keys], separators=(",", ":")).encode()
    return random.Random(int.from_bytes(hashlib.sha256(blob).digest()[:8], "big"))


def need_by(day: int, n_agents: int, initially_hungry: int) -> int:
    """Minimum meals the group must have eaten by the end of `day` (1-based) for all to live.

    An agent starting at h=1 must eat on days 1, 3, 5, ...; one starting at h=0 on days 2, 4, ...
    """
    k = initially_hungry
    return k * ((day + 1) // 2) + (n_agents - k) * (day // 2)


def feasible(seq: list[int], n_agents: int, initially_hungry: int) -> bool:
    """True if cumulative supply covers cumulative minimum need on every prefix."""
    total = 0
    for day, x in enumerate(seq, 1):
        total += x
        if total < need_by(day, n_agents, initially_hungry):
            return False
    return True


def draw_supply(cfg, seed: int, max_attempts: int = 100_000) -> list[int]:
    """Supply sequence for one game; rejection-sampled when cfg.supply_feasible_only."""
    for attempt in range(max_attempts):
        r = stream(seed, "supply", attempt)
        seq = [r.choice(cfg.supply_values) for _ in range(cfg.days)]
        if not cfg.supply_feasible_only or feasible(seq, cfg.n_agents, cfg.initially_hungry):
            return seq
    raise RuntimeError("no feasible supply sequence found; check supply_values")
