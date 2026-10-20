"""Check that an OpenAI API key can run the openai_api seat, before starting a game.

Usage:
  python tools/check_openai_api.py

Sends one short request with the seat's exact settings (gpt-6.1-sol, reasoning effort xhigh,
reasoning summaries on). It stops with the API's error if the key is missing or invalid, the
account cannot use the model, or the settings are refused. The key is never printed. The
check costs well under one US cent.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.agents.openai_api import OpenAIAPISetupError  # noqa: E402
from tools.run_pilot import provider  # noqa: E402


def main():
    p = provider("openai_api")
    try:
        text, meta = asyncio.run(p.complete([
            {"role": "system", "content": "Reply with the single word OK."},
            {"role": "user", "content": "Ready?"}]))
    except OpenAIAPISetupError as e:
        sys.exit(f"not usable: {e}")
    if "error" in meta:
        sys.exit(f"request failed: {meta['error']}")
    print(f"seat model: {p.model_id}; served by: {meta['models']}")
    print(f"reply: {text!r}")
    print(f"tokens in/out/reasoning: {meta['input_tokens']}/{meta['output_tokens']}/{meta['reasoning_tokens']}"
          f"; estimated cost: ${meta.get('cost_usd', 0):.4f}")
    print(f"reasoning summary returned: {'yes' if meta.get('thinking') else 'no (the game still runs; motive evidence will be thinner)'}")


if __name__ == "__main__":
    main()
