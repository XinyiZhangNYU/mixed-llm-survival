import asyncio
import io
import json
import urllib.error
from datetime import datetime, timezone

import sim.agents.deepseek_api as mod
from sim.agents.deepseek_api import DeepSeekAPIProvider, is_peak

MSGS = [{"role": "system", "content": "RULES"},
        {"role": "user", "content": "day 1"},
        {"role": "assistant", "content": '{"text": "hi"}'},
        {"role": "user", "content": "day 1 vote"}]

# shape of a real reply (2026-10-18)
OK = {"id": "r1", "model": "deepseek-v4-pro",
      "choices": [{"finish_reason": "stop",
                   "message": {"role": "assistant", "content": '{"vote": "accept"}',
                               "reasoning_content": "I should accept."}}],
      "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050,
                "prompt_tokens_details": {"cached_tokens": 800},
                "completion_tokens_details": {"reasoning_tokens": 40},
                "prompt_cache_hit_tokens": 800, "prompt_cache_miss_tokens": 200}}
PRICE = {"input": 0.66, "cache_hit": 0.022, "output": 1.98}


def test_body_has_only_engine_content():
    b = DeepSeekAPIProvider("deepseek-v4-pro")._body(MSGS)
    assert "tools" not in b and "temperature" not in b
    assert b["messages"] == MSGS and b["messages"] is not MSGS
    assert b["thinking"] == {"type": "enabled"} and b["reasoning_effort"] == "high"
    off = DeepSeekAPIProvider("deepseek-v4-pro", thinking=False)._body(MSGS)
    assert off["thinking"] == {"type": "disabled"} and "reasoning_effort" not in off


def test_reply_meta_cost_and_no_key_leak(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-secret")
    monkeypatch.setattr(mod, "is_peak", lambda t: False)
    p = DeepSeekAPIProvider("deepseek-v4-pro", price_per_mtok=PRICE)
    monkeypatch.setattr(p, "_post", lambda body: OK)
    text, meta = asyncio.run(p.complete(MSGS))
    assert text == '{"vote": "accept"}' and meta["thinking"] == "I should accept."
    assert (meta["input_tokens"], meta["cached_input_tokens"], meta["output_tokens"],
            meta["reasoning_tokens"]) == (1000, 800, 50, 40)
    assert meta["cost_usd"] == round((200 * 0.66 + 800 * 0.022 + 50 * 1.98) / 1e6, 6)
    assert "sk-test-secret" not in json.dumps(meta)
    monkeypatch.setattr(mod, "is_peak", lambda t: True)
    _, peak = asyncio.run(p.complete(MSGS))
    assert peak["peak_pricing"] and abs(peak["cost_usd"] - 2 * meta["cost_usd"]) < 2e-6


def test_peak_hours():
    mon = lambda h: datetime(2026, 10, 19, h, 30, tzinfo=timezone.utc)   # a Monday
    assert [is_peak(mon(h)) for h in (0, 1, 3, 4, 5, 6, 9, 10)] == [False, True, True, False,
                                                                    False, True, True, False]
    assert not is_peak(datetime(2026, 10, 18, 7, tzinfo=timezone.utc))  # Sunday


def test_retries_then_succeeds_and_client_error_not_retried(monkeypatch):
    async def no_sleep(_):
        return None
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    p = DeepSeekAPIProvider("deepseek-v4-pro", max_http_retries=2)
    calls = {"n": 0}

    def flaky(body):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError("u", 503, "busy", {}, io.BytesIO(b"{}"))
        return OK
    monkeypatch.setattr(p, "_post", flaky)
    _, meta = asyncio.run(p.complete(MSGS))
    assert calls["n"] == 2 and "error" not in meta

    calls["n"] = 0

    def bad(body):
        calls["n"] += 1
        raise urllib.error.HTTPError("u", 402, "pay", {}, io.BytesIO(b'{"error": "Insufficient Balance"}'))
    monkeypatch.setattr(p, "_post", bad)
    text, meta = asyncio.run(p.complete(MSGS))
    assert calls["n"] == 1 and text == "" and meta["error"].startswith("HTTP 402")


def test_missing_key_raises(monkeypatch, tmp_path):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(mod, "KEY_FILE", tmp_path / "absent")
    try:
        DeepSeekAPIProvider("deepseek-v4-pro")._post({})
    except RuntimeError as e:
        assert "DEEPSEEK_API_KEY" in str(e)
    else:
        raise AssertionError("expected RuntimeError")
