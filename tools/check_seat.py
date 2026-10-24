"""Check that a key can run the glm or gemini seat, before starting a game.

Usage:
  python tools/check_seat.py glm|gemini|gemini_vertex

Sends one short request with the seat's exact settings (model, reasoning effort). It stops
with the API's error if the key is missing or invalid, the account cannot use the model or
has no balance, or the settings are refused. The key is never printed. The check costs
well under one US cent.
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.agents.chat_compat import ChatSetupError  # noqa: E402
from tools.run_pilot import provider  # noqa: E402


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else ""
    if kind not in ("glm", "gemini", "gemini_vertex"):
        sys.exit(__doc__)
    p = provider(kind)
    try:
        text, meta = asyncio.run(p.complete([
            {"role": "system", "content": 'Reply with the JSON object {"ok": true} and nothing else.'},
            {"role": "user", "content": "Ready?"}]))
    except ChatSetupError as e:
        sys.exit(f"not usable: {e}")
    if "error" in meta:
        sys.exit(f"request failed: {meta['error']}")
    print(f"seat model: {p.model_id}; served by: {meta['models']}")
    print(f"reply: {text!r}")
    print(f"tokens in/out/reasoning: {meta['input_tokens']}/{meta['output_tokens']}/{meta['reasoning_tokens']}"
          f"; estimated cost: ${meta.get('cost_usd', 0):.5f}")
    print(f"raw usage: {json.dumps(meta['usage'])}")
    print(f"thinking text returned: {'yes' if meta.get('thinking') else 'no'}")


if __name__ == "__main__":
    main()
