"""Model agent behind the shared decide() interface. No model API is connected yet.

A Provider turns a chat message list into reply text. Real providers (one unified
gateway or per-vendor SDKs) are to be added at the pilot stage; until then
NotConnectedProvider raises, and FakeProvider serves tests.
"""
from __future__ import annotations

import time
from typing import Callable, Protocol

from ..observe import Observation
from ..render import render_observation, rules_text
from ..schemas import parse_reply
from .base import Agent, Decision


class Provider(Protocol):
    model_id: str

    async def complete(self, messages: list[dict]) -> tuple[str, dict]:
        """Return (reply text, metadata such as token counts)."""


class NotConnectedProvider:
    def __init__(self, model_id: str):
        self.model_id = model_id

    async def complete(self, messages):
        raise NotImplementedError(f"provider for {self.model_id} is not connected yet")


class FakeProvider:
    """Replies from a list or a function of the message list; for tests."""

    def __init__(self, replies: list[str] | Callable[[list[dict]], str], model_id: str = "fake"):
        self.replies, self.model_id = replies, model_id
        self.calls = 0

    async def complete(self, messages):
        self.calls += 1
        if callable(self.replies):
            return self.replies(messages), {}
        return self.replies[min(self.calls - 1, len(self.replies) - 1)], {}


class LLMAgent(Agent):
    """One conversation per agent per game: system rules, then for each request the new
    visible events plus the decision prompt, and the agent's own replies."""

    def __init__(self, provider: Provider, label: str | None = None):
        self.provider = provider
        self.label = label or provider.model_id

    def start_game(self, agent_id, cfg, seed):
        super().start_game(agent_id, cfg, seed)
        self.messages = [{"role": "system", "content": rules_text(cfg, agent_id)}]

    async def decide(self, obs: Observation, key: str, feedback: str | None = None) -> Decision:
        if feedback is None:
            content = render_observation(obs, self.cfg)
        else:
            content = (f"Your previous reply was not accepted: {feedback}. "
                       "Reply again with a single JSON object in the requested format.")
        self.messages.append({"role": "user", "content": content})
        t0 = time.perf_counter()
        raw, meta = await self.provider.complete(list(self.messages))
        meta = {**meta, "latency_s": round(time.perf_counter() - t0, 3),
                "model_id": self.provider.model_id}
        self.messages.append({"role": "assistant", "content": raw})
        try:
            action, reasoning = parse_reply(key, raw)
        except ValueError as e:
            return Decision(None, error=str(e), raw=raw, meta=meta)
        return Decision(action, raw=raw, reasoning=reasoning, meta=meta)
