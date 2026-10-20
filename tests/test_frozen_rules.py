"""Rules frozen on 2026-10-18 (docs/frozen_design.md). configs/main.yaml must keep the
frozen rule values, and the text players see (system prompt and every request) must match the
snapshot of a dry-run game at N = 5, T = 8, seed 1. Only parameters and models may change.
Regenerate the snapshot only for a deliberate, documented rule change: FREEZE_REGEN=1 pytest."""
import asyncio
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from dry_run_prompts import DryProvider  # noqa: E402

from sim.agents import LLMAgent
from sim.config import Config
from sim.runner import run_game

SNAP = ROOT / "docs" / "frozen_rules_20261018"
FROZEN = dict(rounds=2, supply_feasible_only=False, disclose_supply_values=False, q_reveal_thief=0.5,
              message_order="sequential", message_passes=2, vote_rule="all_but_one",
              proposal_statement=True, deliberation_prompt=True, public_record=True,
              votes_public=True, gift_reveals_giver=True, allocation_must_exhaust_store=False,
              max_retries=2, max_message_chars=1200, invalid_defaults=Config().invalid_defaults)


def test_main_config_keeps_frozen_rules():
    cfg = Config.from_yaml(ROOT / "configs" / "main.yaml")
    assert {k: getattr(cfg, k) for k in FROZEN} == FROZEN
    assert cfg.n_agents == 5
    pilot = Config.from_yaml(ROOT / "configs" / "pilot_n5_v2.yaml")
    assert {k: getattr(pilot, k) for k in FROZEN} == FROZEN


def _conversations():
    cfg = replace(Config.from_yaml(ROOT / "configs" / "main.yaml"), days=8, supply_values=(1, 2, 3, 4))
    roster = {a: LLMAgent(DryProvider(a)) for a in cfg.agent_ids}
    asyncio.run(run_game(cfg, roster, 1))
    return {a: "".join(f"===== {m['role'].upper()} =====\n{m['content']}\n\n" for m in roster[a].messages)
            for a in cfg.agent_ids}


def test_player_text_matches_frozen_snapshot():
    convs = _conversations()
    if os.environ.get("FREEZE_REGEN") == "1":
        SNAP.mkdir(parents=True, exist_ok=True)
        for a, text in convs.items():
            (SNAP / f"seat{a}.txt").write_text(text, encoding="utf-8", newline="\n")
    for a, text in convs.items():
        assert text == (SNAP / f"seat{a}.txt").read_text(encoding="utf-8"), f"seat {a} text changed"
