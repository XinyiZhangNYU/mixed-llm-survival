"""Reply schemas for model agents, and conversion to engine actions.

Model-facing wording uses "take" for the engine's "steal" (neutral operational language).
Every reply may carry a "reasoning" string that is logged and shown to no other agent.
"""
from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, ValidationError


class _Reply(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reasoning: str = ""


class MessageReply(_Reply):
    text: str


class ProposeReply(_Reply):
    allocation: dict[str, int]
    statement: str = ""          # public; only accepted when cfg.proposal_statement


class VoteReply(_Reply):
    vote: Literal["accept", "reject"]


class RequestReply(_Reply):
    request: Literal[0, 1]


class TransferReply(_Reply):
    action: Literal["give", "take", "none"]
    target: Optional[str] = None


class MealReply(_Reply):
    eat: bool


SCHEMAS: dict[str, type[_Reply]] = {
    "message": MessageReply, "propose": ProposeReply, "vote": VoteReply,
    "request": RequestReply, "transfer": TransferReply, "meal": MealReply,
}


def to_engine(key: str, reply: _Reply) -> dict:
    d = reply.model_dump(exclude={"reasoning"})
    if key == "propose" and not d["statement"]:
        del d["statement"]
    if key == "transfer":
        d["action"] = {"take": "steal"}.get(d["action"], d["action"])
        if d["action"] == "none":
            d["target"] = None
    return d


def extract_json(raw: str) -> dict:
    """First JSON object in `raw` (tolerates code fences or text around it)."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(raw[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise ValueError("no JSON object found in reply")


def parse_reply(key: str, raw: str) -> tuple[dict, str]:
    """Return (engine action, reasoning); raise ValueError with a readable message."""
    obj = extract_json(raw)
    try:
        reply = SCHEMAS[key].model_validate(obj, strict=False)
    except ValidationError as e:
        msgs = "; ".join(f"{'.'.join(map(str, x['loc'])) or 'reply'}: {x['msg']}" for x in e.errors())
        raise ValueError(msgs) from None
    return to_engine(key, reply), reply.reasoning
