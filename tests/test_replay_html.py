import json

from sim.agents.scripted import LotteryAgent, NeedFirstAgent, ThiefAgent
from sim.config import Config
from sim.replay_html import build, payload
from sim.runner import run_game_sync


def test_runner_writes_replay_page(tmp_path):
    roster = {"A": ThiefAgent(), "B": ThiefAgent(), "C": LotteryAgent(), "D": NeedFirstAgent(), "E": LotteryAgent()}
    run_game_sync(Config(days=3), roster, 1, tmp_path)
    page = (tmp_path / "replay.html").read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and "http-equiv=\"refresh\"" not in page
    data = payload(tmp_path)
    assert data["finished"] and data["events"][-1]["kind"] == "game_end"
    assert "</script" not in json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    frag = build(tmp_path, tmp_path / "frag.html", standalone=False).read_text(encoding="utf-8")
    assert frag.startswith("<title>")


def test_unfinished_run_reloads(tmp_path):
    roster = {a: LotteryAgent() for a in "ABCDE"}
    run_game_sync(Config(days=2), roster, 1, tmp_path)
    (tmp_path / "result.json").unlink()
    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    (tmp_path / "events.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    assert "http-equiv=\"refresh\"" in build(tmp_path).read_text(encoding="utf-8")


def test_page_exists_before_the_first_request_and_tracks_every_step(tmp_path, monkeypatch):
    import sim.runner as runner
    from sim import engine
    from sim.runner import RunLogger, _setup
    roster = {a: LotteryAgent() for a in "ABCDE"}
    s, _ = _setup(Config(days=2), roster, 1, RunLogger(tmp_path))
    page = (tmp_path / "replay.html").read_text(encoding="utf-8")
    assert "http-equiv=\"refresh\"" in page and engine.pending(s)   # live page, nobody has answered yet
    builds = []
    monkeypatch.setattr(runner, "build_replay", lambda d: builds.append(d))
    run_game_sync(Config(days=2), roster, 1, tmp_path / "run")
    steps = len((tmp_path / "run" / "actions.jsonl").read_text(encoding="utf-8").splitlines())
    assert len(builds) == 1 + steps + 1   # setup, each step, result
