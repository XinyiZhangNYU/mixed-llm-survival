import asyncio
import io
import json
import urllib.error

import pytest

from sim.agents.chat_compat import ChatSetupError, GeminiAPIProvider, ZhipuAPIProvider

MSGS = [{"role": "system", "content": "RULES"},
        {"role": "user", "content": "day 1"},
        {"role": "assistant", "content": '{"text": "hi"}'},
        {"role": "user", "content": "day 1 vote"}]
PRICE = {"input": 1.0, "cache_hit": 0.1, "output": 4.0}

# reply shapes follow each vendor's docs (GLM: OpenAI-compatible; Gemini: native
# generateContent); check against a real reply with tools/check_seat.py before the first game
GLM_OK = {"id": "z1", "model": "glm-5.3-flash",
          "choices": [{"finish_reason": "stop",
                       "message": {"role": "assistant", "content": '{"vote": "accept"}',
                                   "reasoning_content": "I should accept."}}],
          "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050,
                    "prompt_tokens_details": {"cached_tokens": 800}}}
# thinking reported apart from the reply tokens; cached tokens are part of the prompt count
GEMINI_OK = {"responseId": "g1", "modelVersion": "gemini-3.8-flash",
             "candidates": [{"finishReason": "STOP",
                             "content": {"role": "model", "parts": [
                                 {"text": "Weighing the vote.", "thought": True},
                                 {"text": '{"vote": "reject"}'}]}}],
             "usageMetadata": {"promptTokenCount": 1000, "cachedContentTokenCount": 800,
                               "candidatesTokenCount": 20, "thoughtsTokenCount": 80,
                               "totalTokenCount": 1100}}


def http_error(code, body=b"{}"):
    return urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(body))


@pytest.fixture
def no_sleep(monkeypatch):
    async def _sleep(_):
        return None
    monkeypatch.setattr(asyncio, "sleep", _sleep)


def test_bodies_have_only_engine_content():
    z = ZhipuAPIProvider("glm-5.3-flash")._body(MSGS)
    assert "tools" not in z and "temperature" not in z
    assert z["messages"] == MSGS and z["messages"] is not MSGS
    assert z["reasoning_effort"] == "high" and z["stream"] is False
    assert z["thinking"] == {"type": "enabled"}

    g = GeminiAPIProvider("gemini-3.8-flash")._body(MSGS)
    assert set(g) == {"contents", "generationConfig", "systemInstruction"}
    assert g["systemInstruction"] == {"parts": [{"text": "RULES"}]}
    assert g["contents"] == [{"role": "user", "parts": [{"text": "day 1"}]},
                             {"role": "model", "parts": [{"text": '{"text": "hi"}'}]},
                             {"role": "user", "parts": [{"text": "day 1 vote"}]}]
    assert g["generationConfig"] == {"maxOutputTokens": 32000,
                                     "thinkingConfig": {"includeThoughts": True, "thinkingLevel": "high"}}


def test_gemini_endpoint_and_key_header(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gk-test-secret")
    p = GeminiAPIProvider("gemini-3.8-flash")
    assert p._endpoint().endswith("/v1beta/models/gemini-3.8-flash:generateContent")
    assert p._headers()["x-goog-api-key"] == "gk-test-secret" and "Authorization" not in p._headers()


def test_glm_reply_meta_cost_and_no_key_leak(monkeypatch):
    monkeypatch.setenv("ZHIPU_API_KEY", "zk-test-secret")
    p = ZhipuAPIProvider("glm-5.3-flash", price_per_mtok=PRICE)
    monkeypatch.setattr(p, "_post", lambda body: GLM_OK)
    text, meta = asyncio.run(p.complete(MSGS))
    assert text == '{"vote": "accept"}' and meta["thinking"] == "I should accept."
    assert (meta["input_tokens"], meta["cached_input_tokens"], meta["output_tokens"]) == (1000, 800, 50)
    assert meta["cost_usd"] == round((200 * 1.0 + 800 * 0.1 + 50 * 4.0) / 1e6, 6)
    assert p.model_id == "zhipu-api:glm-5.3-flash"
    assert "zk-test-secret" not in json.dumps(meta)


def test_gemini_bills_thinking_and_cache_hits(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "gk-test-secret")
    p = GeminiAPIProvider("gemini-3.8-flash", price_per_mtok=PRICE)
    monkeypatch.setattr(p, "_post", lambda body: GEMINI_OK)
    text, meta = asyncio.run(p.complete(MSGS))
    assert text == '{"vote": "reject"}' and meta["thinking"] == "Weighing the vote."
    assert (meta["input_tokens"], meta["cached_input_tokens"]) == (1000, 800)
    assert (meta["output_tokens"], meta["reasoning_tokens"]) == (100, 80)
    assert meta["cost_usd"] == round((200 * 1.0 + 800 * 0.1 + 100 * 4.0) / 1e6, 6)
    assert meta["usage"] == GEMINI_OK["usageMetadata"] and meta["stop_reason"] == "STOP"
    assert meta["models"] == ["gemini-3.8-flash"] and "gk-test-secret" not in json.dumps(meta)


def test_gemini_blocked_prompt_returns_empty_reply(monkeypatch):
    p = GeminiAPIProvider("gemini-3.8-flash")
    monkeypatch.setattr(p, "_post", lambda body: {"promptFeedback": {"blockReason": "SAFETY"},
                                                  "usageMetadata": {"promptTokenCount": 10}})
    text, meta = asyncio.run(p.complete(MSGS))
    assert text == "" and meta["stop_reason"] == "SAFETY" and meta["output_tokens"] == 0


def test_retries_rate_limit_then_succeeds(monkeypatch, no_sleep):
    p = GeminiAPIProvider("gemini-3.8-flash", max_http_retries=2)
    calls = {"n": 0}

    def flaky(body):
        calls["n"] += 1
        if calls["n"] == 1:
            raise http_error(429, b'{"error": {"status": "RESOURCE_EXHAUSTED"}}')
        return GEMINI_OK
    monkeypatch.setattr(p, "_post", flaky)
    _, meta = asyncio.run(p.complete(MSGS))
    assert calls["n"] == 2 and "error" not in meta


def test_gives_up_after_retries(monkeypatch, no_sleep):
    p = GeminiAPIProvider("gemini-3.8-flash", max_http_retries=1)

    def busy(body):
        raise http_error(503)
    monkeypatch.setattr(p, "_post", busy)
    text, meta = asyncio.run(p.complete(MSGS))
    assert text == "" and meta["error"].startswith("gave up after retries")


@pytest.mark.parametrize("cls,code,body", [
    (GeminiAPIProvider, 400, b'{"error": "API key not valid"}'),
    (GeminiAPIProvider, 403, b"{}"),
    (ZhipuAPIProvider, 401, b"{}"),
    (ZhipuAPIProvider, 429, '{"error": {"code": "1113", "message": "余额不足"}}'.encode("utf-8")),
])
def test_setup_errors_stop_the_run(monkeypatch, no_sleep, cls, code, body):
    p = cls("m")
    calls = {"n": 0}

    def bad(b):
        calls["n"] += 1
        raise http_error(code, body)
    monkeypatch.setattr(p, "_post", bad)
    with pytest.raises(ChatSetupError):
        asyncio.run(p.complete(MSGS))
    assert calls["n"] == 1


def test_missing_key_raises(monkeypatch, tmp_path):
    for cls, env in ((ZhipuAPIProvider, "ZHIPU_API_KEY"), (GeminiAPIProvider, "GEMINI_API_KEY")):
        monkeypatch.delenv(env, raising=False)
        monkeypatch.setattr(cls, "key_file", tmp_path / "absent")
        with pytest.raises(ChatSetupError, match=env):
            cls("m").api_key()


def test_run_pilot_registers_new_kinds():
    from tools.run_pilot import TAG, provider
    assert provider("glm").model_id == "zhipu-api:glm-5.3-flash"
    assert provider("gemini").model_id == "gemini-api:gemini-3.8-flash"
    assert provider("gemini_vertex").model_id == "gemini-vertex:gemini-3.8-flash"
    assert TAG["glm"] == "z" and TAG["gemini"] == "g" and TAG["gemini_vertex"] == "g"
    assert len(set(TAG.values())) == len(TAG) - 2   # openai_api and gemini_vertex share letters
