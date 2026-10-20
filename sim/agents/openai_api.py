"""Provider that runs OpenAI models through the OpenAI API with an API key.

An alternative to openai_plan.py for anyone without a ChatGPT plan sign-in. It sends the
same request body to the same Responses endpoint (rules in `instructions`, the seat's own
turns in `input`, no tools, store=false, stream=true) and reads the reply the same way.
Only the credential, the error handling and the cost estimate differ.

The API key is read from the OPENAI_API_KEY environment variable, or else from
~/.gt6263/openai_api_key. The key is never written to logs or metadata.

Errors that mean the setup is wrong (bad key, no access to the model, no credit left)
stop the run instead of being logged as failed replies, so a game is never played on
engine defaults. Rate limits (429 without insufficient_quota) are retried with backoff.
"""
from __future__ import annotations

import os
from pathlib import Path

from .openai_plan import RETRY_STATUS, OpenAIPlanProvider

KEY_FILE = Path.home() / ".gt6263" / "openai_api_key"
# Every call sends the same kind of body, so a 400 (e.g. reasoning summaries not enabled for
# the organization) would repeat on every turn; stop instead of playing on defaults.
FATAL_STATUS = {400, 401, 403, 404}


class OpenAIAPISetupError(RuntimeError):
    """The key, the model access or the account balance is not usable; fix it and rerun."""


def api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key and KEY_FILE.is_file():
        key = KEY_FILE.read_text(encoding="utf-8").strip()
    if not key:
        raise OpenAIAPISetupError(f"OPENAI_API_KEY is not set and {KEY_FILE} does not exist")
    return key


class OpenAIAPIProvider(OpenAIPlanProvider):
    def __init__(self, model: str, reasoning_effort: str | None = None,
                 timeout_s: float = 300.0, max_http_retries: int = 4,
                 reasoning_summary: str | None = None, price_per_mtok: dict | None = None):
        """price_per_mtok: optional {"input", "cached_input", "output"} in USD per million
        tokens, used only to estimate cost_usd for the runner's cost cap."""
        super().__init__(model, reasoning_effort=reasoning_effort, timeout_s=timeout_s,
                         max_http_retries=max_http_retries, reasoning_summary=reasoning_summary)
        self.model_id = f"openai-api:{model}"
        self.price = price_per_mtok

    def _bearer(self) -> str:
        return api_key()

    def _http_error(self, code: int, detail: str) -> bool:
        if code in FATAL_STATUS or "insufficient_quota" in detail:
            raise OpenAIAPISetupError(f"HTTP {code}: {detail}")
        return code == 429 or code in RETRY_STATUS

    def _stream_error(self, code: str, err: str) -> bool:
        if code == "insufficient_quota":
            raise OpenAIAPISetupError(err)
        return code in ("rate_limit_exceeded", "server_error")

    def _add_cost(self, meta: dict) -> None:
        if not self.price:
            return
        p = self.price
        cached = meta["cached_input_tokens"]
        meta["cost_usd"] = round(((meta["input_tokens"] - cached) * p["input"] + cached * p["cached_input"]
                                  + meta["output_tokens"] * p["output"]) / 1e6, 6)
