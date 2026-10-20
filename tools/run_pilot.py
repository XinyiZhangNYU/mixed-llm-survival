"""Run one pilot game with model seats.

Usage:
  python tools/run_pilot.py <config.yaml> --seats A=claude,B=openai,... [--seed S]
                            [--cap USD] [--out DIR]

Seat kinds (settings fixed here for the pilot):
  claude  Anthropic API, claude-sonnet-5-5, effort high, thinking summaries on
          (at medium it did no thinking on game turns)
  openai  ChatGPT plan, gpt-6.1-sol, reasoning effort xhigh, reasoning summaries on
  opus    Anthropic API, claude-opus-5-5, effort high, thinking summaries on
  astra   ChatGPT plan, gpt-6-astra, reasoning effort xhigh, reasoning summaries on
  deepseek DeepSeek API, deepseek-flash (DeepSeek-V4.1-Flash-1017), thinking on, reasoning
          effort high (its levels are low/high/max; xhigh maps to high); raw chain of
          thought is logged

The cost cap counts Anthropic and DeepSeek API spending (plan calls report no price).
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.agents import LLMAgent  # noqa: E402
from sim.agents.anthropic_api import AnthropicAPIProvider  # noqa: E402
from sim.agents.deepseek_api import DeepSeekAPIProvider  # noqa: E402
from sim.agents.openai_plan import OpenAIPlanProvider  # noqa: E402
from sim.config import Config  # noqa: E402
from sim.runner import run_game  # noqa: E402

# USD per million tokens, Claude Sonnet 5.5 (pricing page, 2026-10-17); 5-minute cache writes.
SONNET_PRICE = {"input": 2.0, "output": 10.0, "cache_write": 2.5, "cache_read": 0.10}
# Claude Opus 5.5 (pricing table, cached 2026-10-06); its effort default is medium, so set it.
OPUS_PRICE = {"input": 4.0, "output": 20.0, "cache_write": 5.0, "cache_read": 0.20}
# DeepSeek V4.1 Flash, off-peak USD rates (pricing page, 2026-10-18); peak calls cost double.
DEEPSEEK_PRICE = {"input": 0.15, "cache_hit": 0.003, "output": 0.6}
TAG = {"claude": "c", "openai": "o", "opus": "p", "astra": "a", "deepseek": "d"}


def provider(kind: str):
    if kind == "claude":
        return AnthropicAPIProvider("claude-sonnet-5-5", max_tokens=16000, effort="high",
                                    thinking_display="summarized", timeout_s=300,
                                    price_per_mtok=SONNET_PRICE)
    if kind == "opus":
        return AnthropicAPIProvider("claude-opus-5-5", max_tokens=16000, effort="high",
                                    thinking_display="summarized", timeout_s=300,
                                    price_per_mtok=OPUS_PRICE)
    if kind == "openai":
        return OpenAIPlanProvider("gpt-6.1-sol", reasoning_effort="xhigh", reasoning_summary="auto")
    if kind == "astra":
        return OpenAIPlanProvider("gpt-6-astra", reasoning_effort="xhigh", reasoning_summary="auto")
    if kind == "deepseek":
        return DeepSeekAPIProvider("deepseek-flash", reasoning_effort="high",
                                   price_per_mtok=DEEPSEEK_PRICE)
    raise ValueError(f"unknown seat kind {kind!r}")


def main():
    args = sys.argv[1:]
    opt = lambda k, d: args[args.index(k) + 1] if k in args else d
    cfg_path = Path(args[0])
    cfg = Config.from_yaml(cfg_path)
    seats = dict(x.split("=") for x in opt("--seats", "").split(","))
    seed = int(opt("--seed", 1))
    cap = float(opt("--cap", 3.0))
    tag = "".join(TAG[seats[a]] for a in cfg.agent_ids)
    out = Path(opt("--out", f"runs/{cfg_path.stem}_{tag}_seed{seed}"))
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} already holds a run; pass a new --out")
    roster = {a: LLMAgent(provider(seats[a])) for a in cfg.agent_ids}
    res = asyncio.run(run_game(cfg, roster, seed, log_dir=out, cost_cap_usd=cap))
    res.pop("state", None)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"logs: {out}")


if __name__ == "__main__":
    main()
