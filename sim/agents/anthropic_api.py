"""Provider that calls the Anthropic Messages API directly.

Nothing is added to the model context beyond what the engine sends: the rules as the
system prompt, then the seat's own user/assistant turns. No tools are declared.

The API key is read from the ANTHROPIC_API_KEY environment variable, or else from
~/.gt6263/anthropic_api_key (for example a key from the Console organization linked to
a Max plan, so usage draws on the plan's monthly API credits). The key is never written
to logs or metadata.

Prompt caching: the system prompt and the newest user turn carry cache_control
breakpoints, so each call reads the earlier conversation from cache.

Effort and thinking: with effort set, the request carries output_config.effort. With
thinking_display set, it also carries thinking {"type": "adaptive", "display": ...}; on
models where thinking is on by default the display default is "omitted" (empty thinking
text), so "summarized" is needed to see anything. The text is the provider's summary, not
the raw chain of thought. Thinking counts toward max_tokens, so max_tokens is set well
above the size of a JSON reply. Thinking text is kept in meta["thinking"] for the logs and
is not sent back to the model on later turns.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 529}
# Fallback when the environment variable is not set; kept outside the project folder.
KEY_FILE = Path.home() / ".gt6263" / "anthropic_api_key"


class AnthropicAPIProvider:
    def __init__(self, model: str, max_tokens: int = 1024, temperature: float | None = None,
                 timeout_s: float = 120.0, max_http_retries: int = 5,
                 price_per_mtok: dict | None = None, cache: bool = True,
                 effort: str | None = None, thinking_display: str | None = None):
        """temperature: None leaves it out of the request (current models reject the field).
        price_per_mtok: optional {"input", "output", "cache_write", "cache_read"} in USD
        per million tokens, used only to estimate cost_usd for the runner's cost cap."""
        self.model = model
        self.model_id = f"anthropic-api:{model}"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.max_http_retries = max_http_retries
        self.price = price_per_mtok
        self.cache = cache
        self.effort = effort
        self.thinking_display = thinking_display

    def _body(self, messages: list[dict]) -> dict:
        system, turns = messages[0]["content"], [dict(m) for m in messages[1:]]
        body = {"model": self.model, "max_tokens": self.max_tokens, "messages": turns}
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.effort is not None:
            body["output_config"] = {"effort": self.effort}
        if self.thinking_display is not None:
            body["thinking"] = {"type": "adaptive", "display": self.thinking_display}
        if self.cache:
            body["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
            last = turns[-1]
            last["content"] = [{"type": "text", "text": last["content"],
                                "cache_control": {"type": "ephemeral"}}]
        else:
            body["system"] = system
        return body

    def _post(self, body: dict) -> dict:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key and KEY_FILE.is_file():
            key = KEY_FILE.read_text(encoding="utf-8").strip()
        if not key:
            raise RuntimeError(f"ANTHROPIC_API_KEY is not set and {KEY_FILE} does not exist")
        req = urllib.request.Request(
            API_URL, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"x-api-key": key, "anthropic-version": API_VERSION,
                     "content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read().decode("utf-8"))

    def _estimate_cost(self, u: dict) -> float | None:
        if not self.price:
            return None
        p = self.price
        return round((u.get("input_tokens", 0) * p["input"]
                      + u.get("output_tokens", 0) * p["output"]
                      + u.get("cache_creation_input_tokens", 0) * p["cache_write"]
                      + u.get("cache_read_input_tokens", 0) * p["cache_read"]) / 1e6, 6)

    async def complete(self, messages: list[dict]) -> tuple[str, dict]:
        body = self._body(messages)
        last_err = None
        for attempt in range(self.max_http_retries + 1):
            try:
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

        blocks = data.get("content", [])
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        thinking = "".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
        u = data.get("usage", {})
        meta = {
            "input_tokens": u.get("input_tokens", 0),
            "cache_read_input_tokens": u.get("cache_read_input_tokens", 0),
            "cache_creation_input_tokens": u.get("cache_creation_input_tokens", 0),
            "output_tokens": u.get("output_tokens", 0),
            "thinking_tokens": (u.get("output_tokens_details") or {}).get("thinking_tokens", 0),
            "stop_reason": data.get("stop_reason"),
            "models": [data.get("model")],
            "request_id": data.get("id"),
        }
        if thinking:
            meta["thinking"] = thinking
        if self.effort is not None:
            meta["effort"] = self.effort
        cost = self._estimate_cost(u)
        if cost is not None:
            meta["cost_usd"] = cost
        return text, meta
