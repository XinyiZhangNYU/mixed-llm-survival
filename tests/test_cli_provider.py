"""Claude CLI provider with the subprocess mocked out: no real model calls."""
import asyncio
import json

from sim.agents import LLMAgent, LotteryAgent
from sim.agents.cli import ClaudeCLIProvider
from sim.config import Config
from sim.runner import run_game


def usage(fresh=10, read=0, created=0):
    return {"input_tokens": fresh, "cache_read_input_tokens": read,
            "cache_creation_input_tokens": created, "output_tokens": 5}


class MockCLI(ClaudeCLIProvider):
    """Records each invocation and simulates CLI sessions: --session-id creates one,
    --resume needs one to exist. reply(prompt) gives the result text, or None for a failure."""

    def __init__(self, reply, mode="persistent", tmp=None):
        super().__init__(executable="claude", workdir=tmp, mode=mode)
        self.reply = reply
        self.invocations = []
        self.sessions = {}

    async def _run(self, args, prompt):
        self.invocations.append({"args": args, "prompt": prompt})
        if "--resume" in args:
            sid = args[args.index("--resume") + 1]
            if sid not in self.sessions:
                return 1, json.dumps({"is_error": True, "result": "no such session"}), ""
        elif "--session-id" in args:
            sid = args[args.index("--session-id") + 1]
            self.sessions[sid] = []
        else:
            sid = None
        text = self.reply(prompt)
        if text is None:
            return 1, "", "boom"
        if sid:
            self.sessions[sid] += [prompt, text]
        return 0, json.dumps({"result": text, "total_cost_usd": 0.01, "is_error": False,
                              "usage": usage(read=100 if "--resume" in args else 0, created=50),
                              "modelUsage": {"m": {}}}), ""


def convo(*turns):
    msgs = [{"role": "system", "content": "RULES"}]
    for i, t in enumerate(turns):
        msgs.append({"role": "user" if i % 2 == 0 else "assistant", "content": t})
    return msgs


def run(coro):
    return asyncio.run(coro)


def test_persistent_sends_only_the_new_turn_and_resumes(tmp_path):
    p = MockCLI(lambda prompt: "r", tmp=tmp_path)
    t1, m1 = run(p.complete(convo("obs1")))
    t2, m2 = run(p.complete(convo("obs1", "r", "obs2")))
    t3, m3 = run(p.complete(convo("obs1", "r", "obs2", "r", "obs3")))
    first, second, third = p.invocations
    assert "--session-id" in first["args"] and "--resume" not in first["args"]
    assert second["args"][second["args"].index("--resume") + 1] == m1["session_id"]
    assert m1["session_id"] == m2["session_id"] == m3["session_id"]
    assert [i["prompt"] for i in p.invocations] == ["obs1", "obs2", "obs3"]
    for inv in p.invocations:
        a = inv["args"]
        assert "--no-session-persistence" not in a and "--system-prompt-file" in a
        assert a[a.index("--tools") + 1] == "" and "--strict-mcp-config" in a
        assert "--disable-slash-commands" in a
    assert all("session_restart" not in m for m in (m1, m2, m3))


def test_meta_keeps_cache_fields_separate(tmp_path):
    p = MockCLI(lambda prompt: "r", tmp=tmp_path)
    run(p.complete(convo("obs1")))
    _, meta = run(p.complete(convo("obs1", "r", "obs2")))
    assert (meta["input_tokens"], meta["cache_read_input_tokens"],
            meta["cache_creation_input_tokens"], meta["total_input_tokens"]) == (10, 100, 50, 160)
    assert meta["cli_mode"] == "persistent"


def test_history_mismatch_restarts_from_flattened_transcript(tmp_path):
    p = MockCLI(lambda prompt: "r", tmp=tmp_path)
    _, m1 = run(p.complete(convo("obs1")))
    _, m2 = run(p.complete(convo("obs1", "EDITED", "obs2")))
    assert m2["session_restart"] == "history_mismatch" and m2["session_id"] != m1["session_id"]
    last = p.invocations[-1]
    assert "--session-id" in last["args"]
    assert last["prompt"].endswith("--- CURRENT REQUEST ---\n\nobs2") and "EDITED" in last["prompt"]


def test_failed_call_drops_the_session(tmp_path):
    replies = iter(["r", None, "r2"])
    p = MockCLI(lambda prompt: next(replies), tmp=tmp_path)
    _, m1 = run(p.complete(convo("obs1")))
    text, m2 = run(p.complete(convo("obs1", "r", "obs2")))
    assert text == "" and "error" in m2
    _, m3 = run(p.complete(convo("obs1", "r", "obs2", "", "feedback")))
    assert m3["session_restart"] == "after_failure" and m3["session_id"] != m1["session_id"]
    assert "obs2" in p.invocations[-1]["prompt"] and "feedback" in p.invocations[-1]["prompt"]


def test_stateless_mode_unchanged(tmp_path):
    p = MockCLI(lambda prompt: "r", mode="stateless", tmp=tmp_path)
    run(p.complete(convo("obs1", "r", "obs2")))
    inv = p.invocations[0]
    assert "--no-session-persistence" in inv["args"]
    assert "--resume" not in inv["args"] and "--session-id" not in inv["args"]
    assert inv["prompt"].endswith("--- CURRENT REQUEST ---\n\nobs2") and "obs1" in inv["prompt"]


def test_game_with_retry_keeps_one_session_per_seat(tmp_path):
    """A's first reply is invalid; the bad reply and the feedback stay in its session."""
    good = {'"text"': '{"text": "hi"}', '"allocation"': '{"allocation": {}}', '"vote"': '{"vote": "reject"}',
            '"request"': '{"request": 1}', '"action"': '{"action": "none"}', '"eat"': '{"eat": true}'}
    state = {"first": True}

    def reply(prompt):
        if state["first"]:
            state["first"] = False
            return "not json"
        if prompt.startswith("Your previous reply was not accepted"):
            return good['"text"']
        tail = prompt.split("Reply:")[-1]
        return next(v for k, v in good.items() if k in tail)

    cfg = Config(days=2, rounds=1)
    roster = {x: LotteryAgent() for x in cfg.agent_ids}
    providers = {s: MockCLI(reply, tmp=tmp_path / s) for s in "AB"}
    for s, p in providers.items():
        roster[s] = LLMAgent(p)
    res = run(run_game(cfg, roster, 0, log_dir=tmp_path / "log"))
    assert res["state"].done
    for s, p in providers.items():
        assert len(p.sessions) == 1                          # one session per seat, never restarted
        assert sum("--session-id" in i["args"] for i in p.invocations) == 1
        (session,) = p.sessions.values()
        assert session == [x["content"] for x in roster[s].messages[1:]]
    a_session = next(iter(providers["A"].sessions.values()))
    assert a_session[1] == "not json" and a_session[2].startswith("Your previous reply was not accepted")
    calls = [json.loads(x) for x in (tmp_path / "log" / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    a_calls = [c for c in calls if c.get("agent") == "A" and "attempt" in c]
    assert a_calls[0]["error"] and a_calls[0]["raw"] == "not json" and a_calls[0]["observation"]
    assert all(c["meta"]["cli_mode"] == "persistent" for c in a_calls)
