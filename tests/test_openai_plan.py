import asyncio
import io
import json
import urllib.error

import pytest

import sim.agents.openai_plan as mod
from sim.agents.openai_plan import OpenAIPlanProvider, PlanUsageLimit

MSGS = [{"role": "system", "content": "RULES"},
        {"role": "user", "content": "day 1"},
        {"role": "assistant", "content": '{"text": "hi"}'},
        {"role": "user", "content": "day 1 vote"}]


def done(text="OK", **usage):
    return {"_event": "response.completed", "_text": text, "_request_id": "req_1", "status": "completed",
            "model": "gpt-x", "usage": {"input_tokens": 12, "output_tokens": 3,
                                        "input_tokens_details": {"cached_tokens": 0}, **usage}}


def test_body_has_only_engine_content():
    b = OpenAIPlanProvider("gpt-x")._body(MSGS)
    assert b["instructions"] == "RULES"
    assert [m["role"] for m in b["input"]] == ["user", "assistant", "user"]
    assert b["store"] is False and b["stream"] is True
    assert "tools" not in b and "temperature" not in b and "reasoning" not in b
    assert OpenAIPlanProvider("gpt-x", reasoning_effort="low")._body(MSGS)["reasoning"] == {"effort": "low"}


def test_reply_and_meta(monkeypatch):
    p = OpenAIPlanProvider("gpt-x")
    monkeypatch.setattr(p, "_post", lambda body: done('{"vote": "accept"}'))
    text, meta = asyncio.run(p.complete(MSGS))
    assert text == '{"vote": "accept"}'
    assert meta["input_tokens"] == 12 and meta["cached_input_tokens"] == 0 and meta["request_id"] == "req_1"


def test_usage_limit_raises(monkeypatch):
    p = OpenAIPlanProvider("gpt-x")
    body = b'{"error": {"code": "subscription_sharing_usage_limit_exceeded"}}'
    monkeypatch.setattr(p, "_post", lambda b: (_ for _ in ()).throw(
        urllib.error.HTTPError("u", 429, "limit", {}, io.BytesIO(body))))
    with pytest.raises(PlanUsageLimit):
        asyncio.run(p.complete(MSGS))


def test_unavailable_is_retried(monkeypatch):
    p = OpenAIPlanProvider("gpt-x", max_http_retries=2)
    calls = {"n": 0}

    def post(body):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"_event": "response.failed", "_text": "", "_request_id": None,
                    "error": {"code": "subscription_sharing_usage_unavailable"}}
        return done()

    async def no_sleep(_):
        return None

    monkeypatch.setattr(p, "_post", post)
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    text, meta = asyncio.run(p.complete(MSGS))
    assert calls["n"] == 2 and text == "OK" and "error" not in meta


def test_missing_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "CRED_FILE", tmp_path / "absent.json")
    with pytest.raises(RuntimeError, match="chatgpt_plan_login"):
        mod.access_token()


def test_store_rejects_grant_without_plan_scope(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "STORE", tmp_path)
    monkeypatch.setattr(mod, "CRED_FILE", tmp_path / "c.json")
    with pytest.raises(RuntimeError, match="not granted"):
        mod._store_token_response({"access_token": "a", "scope": "openid email"}, "cid")
    assert not (tmp_path / "c.json").exists()


def test_reasoning_summary_requested_and_kept(monkeypatch):
    p = OpenAIPlanProvider("gpt-x", reasoning_effort="xhigh", reasoning_summary="auto")
    assert p._body(MSGS)["reasoning"] == {"effort": "xhigh", "summary": "auto"}
    streamed = done('{"vote": "accept"}') | {"_summary": "weighing the split"}
    monkeypatch.setattr(p, "_post", lambda body: streamed)
    assert asyncio.run(p.complete(MSGS))[1]["thinking"] == "weighing the split"
    final_only = done('{"vote": "accept"}') | {"output": [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "from output"}]}]}
    monkeypatch.setattr(p, "_post", lambda body: final_only)
    assert asyncio.run(p.complete(MSGS))[1]["thinking"] == "from output"


def complete_sse(monkeypatch, events, output=None):
    """Replay synthetic wire events; never read credentials or contact a service."""
    p = OpenAIPlanProvider("gpt-x", reasoning_effort="xhigh", reasoning_summary="auto")
    final = {"type": "response.completed", "response": {
        "status": "completed", "model": "gpt-x", "output": output or [],
        "usage": {"output_tokens_details": {"reasoning_tokens": 10}}}}
    events = [*events, {"type": "response.output_text.delta", "delta": "OK"}, final]

    class Stream(io.BytesIO):
        headers = {"x-request-id": "req_fixture"}

    def urlopen(req, **kwargs):
        assert req.full_url == "https://api.openai.com/v1/responses"
        assert json.loads(req.data) == {
            "model": "gpt-x", "instructions": "RULES", "input": MSGS[1:],
            "store": False, "stream": True,
            "reasoning": {"effort": "xhigh", "summary": "auto"}}
        return Stream("".join("data: " + json.dumps(e) + "\n\n" for e in events).encode())

    monkeypatch.setattr(mod, "access_token", lambda: "offline-fixture")
    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen)
    return asyncio.run(p.complete(MSGS))


@pytest.mark.parametrize("event", [
    {"type": "response.reasoning_summary_text.done", "item_id": "rs_fixture",
     "summary_index": 0, "text": "full summary"},
    {"type": "response.reasoning_summary_part.done", "item_id": "rs_fixture",
     "summary_index": 0, "part": {"type": "summary_text", "text": "full summary"}},
    {"type": "response.output_item.done", "item": {"id": "rs_fixture", "type": "reasoning",
     "summary": [{"type": "summary_text", "text": "full summary"}]}},
])
def test_summary_present_only_in_done_event(monkeypatch, event):
    text, meta = complete_sse(monkeypatch, [event])
    assert text == "OK"
    assert meta["thinking"] == "full summary"
    assert meta["response_event_counts"] == {
        event["type"]: 1, "response.output_text.delta": 1, "response.completed": 1}


def test_completed_summary_replaces_partial_delta_without_duplication(monkeypatch):
    events = [
        {"type": "response.reasoning_summary_text.delta", "item_id": "rs_fixture",
         "summary_index": 0, "delta": "full "},
        {"type": "response.reasoning_summary_text.done", "item_id": "rs_fixture",
         "summary_index": 0, "text": "full summary"},
        {"type": "response.reasoning_summary_part.done", "item_id": "rs_fixture",
         "summary_index": 0, "part": {"type": "summary_text", "text": "full summary"}},
    ]
    output = [{"id": "rs_fixture", "type": "reasoning", "summary": [
        {"type": "summary_text", "text": "full summary"},
        {"type": "summary_text", "text": "second part"}]}]
    _, meta = complete_sse(monkeypatch, events, output)
    assert meta["thinking"] == "full summary\n\nsecond part"


def test_no_summary_is_not_synthesized_from_usage_or_encrypted_content(monkeypatch):
    output = [{"id": "rs_fixture", "type": "reasoning", "summary": [],
               "encrypted_content": "opaque-fixture"}]
    _, meta = complete_sse(monkeypatch, [], output)
    assert "thinking" not in meta
    assert meta["reasoning_tokens"] == 10
    assert meta["response_event_counts"] == {
        "response.output_text.delta": 1, "response.completed": 1}
    assert "opaque-fixture" not in json.dumps(meta)
