"""Provider that calls the DeepSeek Chat Completions API (OpenAI-compatible) directly.

Nothing is added to the model context beyond what the engine sends: the rules as the
system message, then the seat's own user/assistant turns. No tools are declared.

The API key is read from the DEEPSEEK_API_KEY environment variable, or else from
~/.gt6263/deepseek_api_key (written by tools/save_deepseek_key.py). The key is never
written to logs or metadata.

Thinking: on by default for current models; the request sets thinking {"type": "enabled"}
and reasoning_effort explicitly so the setting is recorded. The reply's reasoning_content
is the model's own chain of thought (not a summary, unlike the Anthropic and OpenAI
seats). It is kept in meta["thinking"] for the logs and is not sent back on later turns;
without tools the API ignores it anyway (api-docs.deepseek.com/guides/thinking_mode).

Caching is automatic on the provider side (prefix cache); usage reports hit and miss
tokens separately. Prices differ by time of day (peak rates are twice off-peak), so the
cost estimate uses the UTC time of the call.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API_URL = "https://api.deepseek.com/chat/completions"
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
KEY_FILE = Path.home() / ".gt6263" / "deepseek_api_key"


def is_peak(t: datetime) -> bool:
    """Peak hours per the pricing page (2026-10-18): 01:00-04:00 and 06:00-10:00 UTC,
    Monday to Friday. Chinese public holidays (off-peak) are not modelled."""
    t = t.astimezone(timezone.utc)
    return t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10)


class DeepSeekAPIProvider:
    def __init__(self, model: str, max_tokens: int = 32000, reasoning_effort: str | None = "high",
                 thinking: bool = True, timeout_s: float = 300.0, max_http_retries: int = 5,
                 price_per_mtok: dict | None = None):
        """price_per_mtok: optional {"input", "cache_hit", "output"} in USD per million
        tokens at OFF-PEAK rates; peak calls are charged double. Used only to estimate
        cost_usd for the runner's cost cap."""
        self.model = model
        self.model_id = f"deepseek-api:{model}"
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.thinking = thinking
        self.timeout_s = timeout_s
        self.max_http_retries = max_http_retries
        self.price = price_per_mtok

    def _body(self, messages: list[dict]) -> dict:
        body = {"model": self.model, "max_tokens": self.max_tokens, "stream": False,
                "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
                "thinking": {"type": "enabled" if self.thinking else "disabled"}}
        if self.thinking and self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def _post(self, body: dict) -> dict:
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key and KEY_FILE.is_file():
            key = KEY_FILE.read_text(encoding="utf-8").strip()
        if not key:
            raise RuntimeError(f"DEEPSEEK_API_KEY is not set and {KEY_FILE} does not exist")
        req = urllib.request.Request(
            API_URL, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {key}", "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read().decode("utf-8"))

    def _estimate_cost(self, meta: dict, peak: bool) -> float | None:
        if not self.price:
            return None
        p, k = self.price, 2.0 if peak else 1.0
        miss = meta["input_tokens"] - meta["cached_input_tokens"]
        return round(k * (miss * p["input"] + meta["cached_input_tokens"] * p["cache_hit"]
                          + meta["output_tokens"] * p["output"]) / 1e6, 6)

    async def complete(self, messages: list[dict]) -> tuple[str, dict]:
        body = self._body(messages)
        last_err = None
        for attempt in range(self.max_http_retries + 1):
            try:
                started = datetime.now(timezone.utc)
                data = await asyncio.to_thread(self._post, body)
                break
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")[:500]
                last_err = f"HTTP {e.code}: {detail}"
                if e.code not in RETRY_STATUS:
                    return "", {"error": last_err}
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = f"network: {e}"
            await asyncio.sleep(min(60.0, 2 ** attempt + random.random()))
        else:
            return "", {"error": f"gave up after retries; last: {last_err}"}

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        thinking = msg.get("reasoning_content") or ""
        u = data.get("usage") or {}
        peak = is_peak(started)
        meta = {
            "input_tokens": u.get("prompt_tokens", 0),
            "cached_input_tokens": u.get("prompt_cache_hit_tokens",
                                         (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)),
            "output_tokens": u.get("completion_tokens", 0),
            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0),
            "stop_reason": choice.get("finish_reason"),
            "models": [data.get("model")],
            "request_id": data.get("id"),
            "peak_pricing": peak,
        }
        if thinking:
            meta["thinking"] = thinking
        if self.thinking and self.reasoning_effort is not None:
            meta["effort"] = self.reasoning_effort
        cost = self._estimate_cost(meta, peak)
        if cost is not None:
            meta["cost_usd"] = cost
        return text, meta
