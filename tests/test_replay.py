"""A logged game can be rebuilt exactly from its action log, and checkpoints round-trip."""
import json

import pytest
from conftest import mixed_roster

from sim.config import Config
from sim.runner import load_checkpoint, replay, run_game_sync


@pytest.mark.parametrize("order", ["simultaneous", "sequential"])
@pytest.mark.parametrize("seed", [0, 3, 11])
def test_replay_matches(tmp_path, seed, order):
    cfg = Config(message_order=order)
    res = run_game_sync(cfg, mixed_roster(cfg), seed, log_dir=tmp_path)
    original = res["state"].to_dict()
    assert replay(tmp_path).to_dict() == original
    assert load_checkpoint(tmp_path).to_dict() == original
    logged = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert logged == original["events"]
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["survival"] == res["survival"]

