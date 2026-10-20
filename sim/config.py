"""Game configuration.

Fields marked OPEN were not fixed by the proposal, and the defaults here are the
pre-pilot placeholders. The rules were frozen on 2026-10-18: configs/main.yaml holds the
frozen values and tests/test_frozen_rules.py checks them (docs/frozen_design.md).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    n_agents: int = 5
    days: int = 15
    rounds: int = 3
    supply_values: tuple[int, ...] = (2, 3, 4)
    # Keep only supply sequences under which perfect coordination keeps everyone alive.
    # Scripted reference runs only: model agents are told supply is independent across
    # days, which a filtered sequence is not, so rules_text refuses a filtered config.
    supply_feasible_only: bool = False
    initially_hungry: int = 2
    # Whether the rules text lists the possible daily supply values and their distribution.
    # False: players only learn that the amount is random, and see each day's amount on arrival.
    disclose_supply_values: bool = False

    # OPEN: probability that a theft victim privately learns the thief's identity.
    q_reveal_thief: float = 0.5
    # OPEN: "simultaneous" (nobody sees this round's messages before writing) or
    # "sequential" (seat order starting from the round's proposer).
    message_order: str = "simultaneous"
    # OPEN: message passes per negotiation round. Each living player posts once per pass;
    # with sequential order every pass uses the same order.
    message_passes: int = 1
    # OPEN: votes needed to pass, out of n living players (the proposer votes too):
    # "unanimous" n; "majority" more than half; "all_but_one" n - 1, but never fewer than a
    # majority (so with 2 players both must accept). See engine.votes_needed.
    vote_rule: str = "unanimous"
    # OPEN: whether the proposer may attach a public statement to the proposal.
    proposal_statement: bool = False
    # OPEN: whether each decision request asks the player to review the game so far before
    # replying (one fixed sentence, identical for every player and model).
    deliberation_prompt: bool = False
    # OPEN: whether each day after the first opens with a record of earlier days built only
    # from announced events (proposals, votes, releases, deaths).
    public_record: bool = False
    # OPEN: whether individual votes are public or only the pass/fail outcome.
    votes_public: bool = True
    # OPEN: whether a gift recipient learns who gave the unit.
    gift_reveals_giver: bool = True
    # OPEN: whether a proposal must allocate the whole store (False lets food stay in store).
    allocation_must_exhaust_store: bool = False
    # OPEN: actions used when an agent's output stays invalid after retries.
    invalid_defaults: dict = field(default_factory=lambda: {
        "message": {"text": ""},
        "propose": None,            # None = no proposal this round, so the round fails
        "vote": {"vote": "reject"},
        "request": {"request": 0},
        "transfer": {"action": "none", "target": None},
        "meal": {"eat": False},
    })
    max_retries: int = 2
    max_message_chars: int = 1200

    def __post_init__(self):
        if self.n_agents < 2 or self.n_agents > 26:
            raise ValueError("n_agents must be in [2, 26]")
        if not 0 <= self.initially_hungry <= self.n_agents:
            raise ValueError("initially_hungry out of range")
        if self.message_order not in ("simultaneous", "sequential"):
            raise ValueError("message_order must be simultaneous or sequential")
        if self.message_passes < 1:
            raise ValueError("message_passes must be at least 1")
        if self.vote_rule not in ("unanimous", "majority", "all_but_one"):
            raise ValueError("vote_rule must be unanimous, majority or all_but_one")
        if not 0.0 <= self.q_reveal_thief <= 1.0:
            raise ValueError("q_reveal_thief must be a probability")
        if min(self.supply_values) < 0:
            raise ValueError("supply values must be nonnegative")
        object.__setattr__(self, "supply_values", tuple(self.supply_values))

    @property
    def agent_ids(self) -> list[str]:
        return [chr(ord("A") + i) for i in range(self.n_agents)]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["supply_values"] = list(self.supply_values)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        d = dict(d)
        if "invalid_defaults" in d:
            merged = cls().invalid_defaults
            merged.update(d["invalid_defaults"])
            d["invalid_defaults"] = merged
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})
