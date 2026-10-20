import asyncio
import io
import json
import urllib.error

from sim.agents.anthropic_api import AnthropicAPIProvider

MSGS = [{"role": "system", "content": "RULES"},
        {"role": "user", "content": "day 1"},
        {"role": "assistant", "content": '{"text": "hi"}'},
        {"role": "user", "content": "day 1 vote"}]

OK = {"id": "msg_1", "model": "claude-x", "stop_reason": "end_turn",
      "content": [{"type": "text", "text": '{"vote": "accept"}'}],
      "usage": {"input_tokens": 10, "output_tokens": 5,
                "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20}}


def test_body_has_only_engine_content_and_cache_breakpoints():
    b = AnthropicAPIProvider("claude-x")._body(MSGS)
    assert "tools" not in b
    assert b["system"] == [{"type": "text", "text": "RULES", "cache_control": {"type": "ephemeral"}}]
    assert [m["role"] for m in b["messages"]] == ["user", "assistant", "user"]
    assert b["messages"][-1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert b["messages"][0]["content"] == "day 1"
    assert MSGS[-1]["content"] == "day 1 vote"          # caller's list is not mutated


def test_reply_meta_and_cost_without_key_leak(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-secret")
    p = AnthropicAPIProvider("claude-x", price_per_mtok={"input": 3, "output": 15,
                                                         "cache_write": 3.75, "cache_read": 0.3})
    monkeypatch.setattr(p, "_post", lambda body: OK)
    text, meta = asyncio.run(p.complete(MSGS))
    assert text == '{"vote": "accept"}'
    assert meta["cache_read_input_tokens"] == 100 and meta["cache_creation_input_tokens"] == 20
    assert meta["cost_usd"] == round((10 * 3 + 5 * 15 + 20 * 3.75 + 100 * 0.3) / 1e6, 6)
    assert "sk-test-secret" not in json.dumps(meta)


def test_retries_overload_then_succeeds(monkeypatch):
    p = AnthropicAPIProvider("claude-x", max_http_retries=2)
    calls = {"n": 0}

    def post(body):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError("u", 529, "overloaded", {}, io.BytesIO(b"{}"))
        return OK

    async def no_sleep(_):
        return None

    monkeypatch.setattr(p, "_post", post)
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    text, meta = asyncio.run(p.complete(MSGS))
    assert calls["n"] == 2 and "error" not in meta


def test_client_error_is_not_retried(monkeypatch):
    p = AnthropicAPIProvider("claude-x")
    calls = {"n": 0}

    def post(body):
        calls["n"] += 1
        raise urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b'{"error": "bad request"}'))

    monkeypatch.setattr(p, "_post", post)
    text, meta = asyncio.run(p.complete(MSGS))
    assert calls["n"] == 1 and text == "" and meta["error"].startswith("HTTP 400")


def test_missing_key_raises(monkeypatch, tmp_path):
    import sim.agents.anthropic_api as mod
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(mod, "KEY_FILE", tmp_path / "absent")
    try:
        AnthropicAPIProvider("claude-x")._post({})
    except RuntimeError as e:
        assert "ANTHROPIC_API_KEY" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_effort_in_body_and_thinking_kept_in_meta(monkeypatch):
    p = AnthropicAPIProvider("claude-x", max_tokens=8000, effort="medium")
    b = p._body(MSGS)
    assert b["output_config"] == {"effort": "medium"} and b["max_tokens"] == 8000
    assert "output_config" not in AnthropicAPIProvider("claude-x")._body(MSGS)
    assert "thinking" not in b
    shown = AnthropicAPIProvider("claude-x", thinking_display="summarized")._body(MSGS)
    assert shown["thinking"] == {"type": "adaptive", "display": "summarized"}
    reply = dict(OK, content=[{"type": "thinking", "thinking": "weigh it", "signature": "s"},
                              {"type": "text", "text": '{"vote": "accept"}'}])
    monkeypatch.setattr(p, "_post", lambda body: reply)
    text, meta = asyncio.run(p.complete(MSGS))
    assert text == '{"vote": "accept"}'
    assert meta["thinking"] == "weigh it" and meta["effort"] == "medium"
