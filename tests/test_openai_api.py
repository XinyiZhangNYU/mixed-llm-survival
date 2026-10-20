import asyncio
import importlib.util
import io
import json
import urllib.error
from pathlib import Path

import pytest

import sim.agents.openai_api as mod
import sim.agents.openai_plan as plan_mod
from sim.agents.openai_api import OpenAIAPIProvider, OpenAIAPISetupError
from sim.agents.openai_plan import OpenAIPlanProvider

MSGS = [{"role": "system", "content": "RULES"},
        {"role": "user", "content": "day 1"},
        {"role": "assistant", "content": '{"text": "hi"}'},
        {"role": "user", "content": "day 1 vote"}]


def run_pilot():
    spec = importlib.util.spec_from_file_location(
        "run_pilot", Path(__file__).resolve().parents[1] / "tools" / "run_pilot.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def http_error(code, body):
    return lambda b: (_ for _ in ()).throw(urllib.error.HTTPError("u", code, "x", {}, io.BytesIO(body)))


def done(text="OK"):
    return {"_event": "response.completed", "_text": text, "_request_id": "req_1", "status": "completed",
            "model": "gpt-x", "usage": {"input_tokens": 1_000_000, "output_tokens": 100_000,
                                        "input_tokens_details": {"cached_tokens": 400_000}}}


@pytest.fixture
def no_sleep(monkeypatch):
    async def sleep(_):
        return None
    monkeypatch.setattr(asyncio, "sleep", sleep)


def test_api_seat_sends_the_same_request_as_the_plan_seat():
    rp = run_pilot()
    api, plan = rp.provider("openai_api"), rp.provider("openai")
    assert api._body(MSGS) == plan._body(MSGS)
    assert api.model == plan.model and api.model_id == "openai-api:gpt-6.1-sol"
    assert rp.TAG["openai_api"] == rp.TAG["openai"]


def test_key_from_environment_goes_only_in_the_header(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-fixture")
    seen = {}

    class Stream(io.BytesIO):
        headers = {"x-request-id": "req_fixture"}

    def urlopen(req, **kwargs):
        seen["auth"] = req.get_header("Authorization")
        seen["body"] = req.data.decode()
        final = {"type": "response.completed", "response": {"status": "completed", "model": "gpt-x"}}
        return Stream(("data: " + json.dumps({"type": "response.output_text.delta", "delta": "OK"}) + "\n\n"
                       + "data: " + json.dumps(final) + "\n\n").encode())

    monkeypatch.setattr(plan_mod.urllib.request, "urlopen", urlopen)
    text, meta = asyncio.run(OpenAIAPIProvider("gpt-x").complete(MSGS))
    assert text == "OK" and seen["auth"] == "Bearer test-key-fixture"
    assert "test-key-fixture" not in seen["body"] and "test-key-fixture" not in json.dumps(meta)


def test_missing_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(mod, "KEY_FILE", tmp_path / "absent")
    with pytest.raises(OpenAIAPISetupError, match="OPENAI_API_KEY"):
        mod.api_key()


@pytest.mark.parametrize("code,body", [
    (401, b'{"error": {"code": "invalid_api_key"}}'),
    (404, b'{"error": {"code": "model_not_found"}}'),
    (400, b'{"error": {"message": "organization must be verified"}}'),
    (429, b'{"error": {"code": "insufficient_quota"}}'),
])
def test_setup_errors_stop_the_run(monkeypatch, code, body):
    p = OpenAIAPIProvider("gpt-x")
    monkeypatch.setattr(p, "_post", http_error(code, body))
    with pytest.raises(OpenAIAPISetupError):
        asyncio.run(p.complete(MSGS))


def test_rate_limit_is_retried(monkeypatch, no_sleep):
    p = OpenAIAPIProvider("gpt-x", max_http_retries=2)
    calls = {"n": 0}

    def post(body):
        calls["n"] += 1
        if calls["n"] == 1:
            http_error(429, b'{"error": {"code": "rate_limit_exceeded"}}')(body)
        if calls["n"] == 2:
            return {"_event": "response.failed", "_text": "", "_request_id": None,
                    "error": {"code": "rate_limit_exceeded"}}
        return done()

    monkeypatch.setattr(p, "_post", post)
    text, meta = asyncio.run(p.complete(MSGS))
    assert calls["n"] == 3 and text == "OK" and "error" not in meta


def test_quota_in_stream_stops_the_run(monkeypatch):
    p = OpenAIAPIProvider("gpt-x")
    monkeypatch.setattr(p, "_post", lambda b: {"_event": "response.failed", "_text": "", "_request_id": None,
                                               "error": {"code": "insufficient_quota"}})
    with pytest.raises(OpenAIAPISetupError):
        asyncio.run(p.complete(MSGS))


def test_cost_estimate(monkeypatch):
    p = OpenAIAPIProvider("gpt-x", price_per_mtok={"input": 2.5, "cached_input": 0.1, "output": 10.0})
    monkeypatch.setattr(p, "_post", lambda b: done())
    _, meta = asyncio.run(p.complete(MSGS))
    assert meta["cost_usd"] == pytest.approx(0.6 * 2.5 + 0.4 * 0.1 + 0.1 * 10.0)
    monkeypatch.setattr(OpenAIPlanProvider, "_post", lambda self, b: done())
    assert "cost_usd" not in asyncio.run(OpenAIPlanProvider("gpt-x").complete(MSGS))[1]
