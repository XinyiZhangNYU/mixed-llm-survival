"""Providers for Zhipu GLM (OpenAI-compatible Chat Completions) and Google Gemini (native
generateContent endpoint).

Nothing is added to the model context beyond what the engine sends: the rules as the
system message, then the seat's own user/assistant turns. No tools are declared. Both
vendors document these endpoints officially (docs.bigmodel.cn; ai.google.dev/gemini-api/docs).

Keys are read from an environment variable, or else from a file under ~/.gt6263/. A key is
never written to logs or metadata.

Errors that mean the setup is wrong (bad key, no access to the model, no balance left)
stop the run instead of being logged as failed replies, so a game is never played on
engine defaults. Rate limits and server errors are retried with backoff.

Each reply's raw usage object is kept in meta["usage"], so the token accounting can be
checked against the vendor's billing page after the first games.
"""
from __future__ import annotations

import asyncio
import http.client
import json
import os
import random
import urllib.error
import urllib.request
from pathlib import Path

RETRY_STATUS = {408, 429, 500, 502, 503, 504}
FATAL_STATUS = {400, 401, 402, 403, 404}


class ChatSetupError(RuntimeError):
    """The key, the model access or the account balance is not usable; fix it and rerun."""


class ChatCompatProvider:
    url: str = ""
    key_env: str = ""
    key_file: Path = Path()
    prefix: str = ""
    # substrings of an error body that mean "no balance / no quota left" even on a 429
    fatal_markers: tuple[str, ...] = ()

    def __init__(self, model: str, max_tokens: int = 32000, reasoning_effort: str | None = "high",
                 timeout_s: float = 300.0, max_http_retries: int = 5,
                 price_per_mtok: dict | None = None):
        """price_per_mtok: optional {"input", "cache_hit", "output"} in USD per million
        tokens, used only to estimate cost_usd for the runner's cost cap."""
        self.model = model
        self.model_id = f"{self.prefix}:{model}"
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.timeout_s = timeout_s
        self.max_http_retries = max_http_retries
        self.price = price_per_mtok

    def _body(self, messages: list[dict]) -> dict:
        body = {"model": self.model, "max_tokens": self.max_tokens, "stream": False,
                "messages": [{"role": m["role"], "content": m["content"]} for m in messages]}
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def api_key(self) -> str:
        key = os.environ.get(self.key_env)
        if not key and self.key_file.is_file():
            key = self.key_file.read_text(encoding="utf-8").strip()
        if not key:
            raise ChatSetupError(f"{self.key_env} is not set and ~/.gt6263/{self.key_file.name} does not exist")
        return key

    def _endpoint(self) -> str:
        return self.url

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key()}", "content-type": "application/json"}

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(
            self._endpoint(), data=json.dumps(body).encode("utf-8"), method="POST",
            headers=self._headers())
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read().decode("utf-8"))

    def _usage(self, u: dict) -> dict:
        prompt = u.get("prompt_tokens", 0)
        completion = u.get("completion_tokens", 0)
        reasoning = (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        # Some endpoints leave thinking out of completion_tokens but count it in
        # total_tokens; bill whichever is larger so the cost cap never errs low.
        output = max(completion, u.get("total_tokens", 0) - prompt)
        return {"input_tokens": prompt,
                "cached_input_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
                "output_tokens": output,
                "reasoning_tokens": reasoning or max(0, output - completion)}

    def _estimate_cost(self, meta: dict) -> float | None:
        if not self.price:
            return None
        p = self.price
        miss = meta["input_tokens"] - meta["cached_input_tokens"]
        return round((miss * p["input"] + meta["cached_input_tokens"] * p["cache_hit"]
                      + meta["output_tokens"] * p["output"]) / 1e6, 6)

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
                if e.code in FATAL_STATUS or any(m in detail for m in self.fatal_markers):
                    raise ChatSetupError(last_err)
                if e.code not in RETRY_STATUS:
                    return "", {"error": last_err}
            except (urllib.error.URLError, TimeoutError, http.client.HTTPException, ConnectionError) as e:
                last_err = f"network: {e}"
            await asyncio.sleep(min(60.0, 2 ** attempt + random.random()))
        else:
            return "", {"error": f"gave up after retries; last: {last_err}"}

        text, meta = self._parse(data)
        if self.reasoning_effort is not None:
            meta["effort"] = self.reasoning_effort
        cost = self._estimate_cost(meta)
        if cost is not None:
            meta["cost_usd"] = cost
        return text, meta

    def _parse(self, data: dict) -> tuple[str, dict]:
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        u = data.get("usage") or {}
        meta = {**self._usage(u),
                "stop_reason": choice.get("finish_reason"),
                "models": [data.get("model")],
                "request_id": data.get("id"),
                "usage": u}
        thinking = msg.get("reasoning_content") or ""
        if thinking:
            meta["thinking"] = thinking
        return msg.get("content") or "", meta


class ZhipuAPIProvider(ChatCompatProvider):
    """Zhipu BigModel platform (open.bigmodel.cn). GLM-5.3 models always think; the request
    still sets thinking {"type": "enabled"} and reasoning_effort (low/high/max) so the
    setting is recorded. The reply's reasoning_content is logged in meta["thinking"]."""
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    key_env = "ZHIPU_API_KEY"
    key_file = Path.home() / ".gt6263" / "zhipu_api_key"
    prefix = "zhipu-api"
    fatal_markers = ("\"1113\"", "余额不足")   # insufficient balance arrives as a 429

    def _body(self, messages: list[dict]) -> dict:
        return {**super()._body(messages), "thinking": {"type": "enabled"}}


class GeminiAPIProvider(ChatCompatProvider):
    """Gemini API through its native generateContent endpoint. Gemini 3.x models always
    think; reasoning_effort is sent as thinkingConfig.thinkingLevel (low/medium/high).

    The native endpoint is used rather than the OpenAI-compatible one because only it
    reports implicit-cache hits (cachedContentTokenCount) and thinking tokens
    (thoughtsTokenCount) separately, so cost estimates match billing. System messages go
    to systemInstruction, assistant turns are sent with role "model". Thought summaries
    are requested and logged in meta["thinking"]; they are kept out of the reply text."""
    url = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    key_env = "GEMINI_API_KEY"
    key_file = Path.home() / ".gt6263" / "gemini_api_key"
    prefix = "gemini-api"

    def _endpoint(self) -> str:
        return self.url.format(model=self.model)

    def _headers(self) -> dict:
        return {"x-goog-api-key": self.api_key(), "content-type": "application/json"}

    def _body(self, messages: list[dict]) -> dict:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [{"role": "model" if m["role"] == "assistant" else "user",
                     "parts": [{"text": m["content"]}]}
                    for m in messages if m["role"] != "system"]
        thinking = {"includeThoughts": True}
        if self.reasoning_effort is not None:
            thinking["thinkingLevel"] = self.reasoning_effort
        body = {"contents": contents,
                "generationConfig": {"maxOutputTokens": self.max_tokens, "thinkingConfig": thinking}}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    def _usage(self, u: dict) -> dict:
        thoughts = u.get("thoughtsTokenCount", 0)
        return {"input_tokens": u.get("promptTokenCount", 0),
                "cached_input_tokens": u.get("cachedContentTokenCount", 0),
                "output_tokens": u.get("candidatesTokenCount", 0) + thoughts,
                "reasoning_tokens": thoughts}

    def _parse(self, data: dict) -> tuple[str, dict]:
        cand = (data.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        thinking = "".join(p.get("text", "") for p in parts if p.get("thought"))
        u = data.get("usageMetadata") or {}
        meta = {**self._usage(u),
                "stop_reason": cand.get("finishReason")
                or (data.get("promptFeedback") or {}).get("blockReason"),
                "models": [data.get("modelVersion")],
                "request_id": data.get("responseId"),
                "usage": u}
        if thinking:
            meta["thinking"] = thinking
        return text, meta


class GeminiVertexProvider(GeminiAPIProvider):
    """The same model and request body as GeminiAPIProvider, served by Vertex AI
    (aiplatform.googleapis.com, global endpoint) with a Google Cloud API key. Billing goes to
    the Cloud project instead of the Gemini API account. The project ID is read from
    GT6263_VERTEX_PROJECT, or else from ~/.gt6263/vertex_project."""
    url = ("https://aiplatform.googleapis.com/v1/projects/{project}/locations/global"
           "/publishers/google/models/{model}:generateContent")
    key_env = "VERTEX_API_KEY"
    key_file = Path.home() / ".gt6263" / "vertex_api_key"
    project_env = "GT6263_VERTEX_PROJECT"
    project_file = Path.home() / ".gt6263" / "vertex_project"
    prefix = "gemini-vertex"

    def __init__(self, *args, max_http_retries: int = 12, **kwargs):
        # Vertex shares on-demand capacity across projects and answers HTTP 429 when it is
        # busy. Twelve retries wait about seven minutes in total, so a busy period delays a
        # reply instead of turning it into an invalid one that the player is told about.
        super().__init__(*args, max_http_retries=max_http_retries, **kwargs)

    def _endpoint(self) -> str:
        project = os.environ.get(self.project_env)
        if not project and self.project_file.is_file():
            project = self.project_file.read_text(encoding="utf-8").strip()
        if not project:
            raise ChatSetupError(f"{self.project_env} is not set and ~/.gt6263/{self.project_file.name} does not exist")
        return self.url.format(project=project, model=self.model)
