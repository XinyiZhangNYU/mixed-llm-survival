"""Dry run: play one game with model seats whose provider never calls a model, and write the
exact conversation each seat would be sent, plus the request bodies each live provider would
send for seat A's last call.

Replies are fixed placeholders (message "(placeholder)", allocate one unit each in seat
order, accept, request 1, no transfer, eat when possible), except that seat B always rejects
and gives a unit to the first other living seat, and seat C takes a unit from it, so the
failed-vote, fallback, gift and theft wording also appears. The game path is not what a
model would do; the point is to see the system prompt and the user turns as sent.

Usage: python tools/dry_run_prompts.py <config.yaml> [--seed S] [--out DIR]
"""
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.agents import LLMAgent  # noqa: E402
from sim.agents.anthropic_api import AnthropicAPIProvider  # noqa: E402
from sim.agents.openai_plan import OpenAIPlanProvider  # noqa: E402
from sim.config import Config  # noqa: E402
from sim.runner import run_game  # noqa: E402


class DryProvider:
    model_id = "dry-run"

    def __init__(self, seat: str):
        self.seat = seat
        self.last: list[dict] = []

    async def complete(self, messages):
        self.last = messages
        ask = messages[-1]["content"]
        if "Post your public message" in ask:
            r = {"text": "(placeholder)"}
        elif "You are the proposer" in ask:
            living = re.search(r"living player \(([^)]*)\)", ask).group(1).split(", ")
            left = int(re.search(r"from the store of (\d+)", ask).group(1))
            alloc = {}
            for a in living:
                alloc[a] = 1 if left > 0 else 0
                left -= alloc[a]
            r = {"allocation": alloc}
        elif re.search(r"^Vote on", ask, re.M):
            r = {"vote": "reject" if self.seat == "B" else "accept"}
        elif "Request 0 or 1" in ask:
            r = {"request": 1}
        elif "transfer action" in ask:
            m = re.search(r"Possible targets: ([^.]*)\.", ask)
            first = m.group(1).split(", ")[0] if m else None
            if first and self.seat == "B":
                r = {"action": "give", "target": first}
            elif first and self.seat == "C":
                r = {"action": "take", "target": first}
            else:
                r = {"action": "none"}
        else:
            r = {"eat": True}
        return json.dumps(r), {}


def main():
    args = sys.argv[1:]
    opt = lambda k, d: args[args.index(k) + 1] if k in args else d
    cfg_path = Path(args[0])
    seed = int(opt("--seed", 1))
    out = Path(opt("--out", "docs/pilot_prompts"))
    cfg = Config.from_yaml(cfg_path)
    provs = {a: DryProvider(a) for a in cfg.agent_ids}
    roster = {a: LLMAgent(p) for a, p in provs.items()}
    res = asyncio.run(run_game(cfg, roster, seed))
    out.mkdir(parents=True, exist_ok=True)
    for seat in cfg.agent_ids:
        stem = f"{cfg_path.stem}_seed{seed}_seat{seat}"
        with open(out / f"{stem}.txt", "w", encoding="utf-8", newline="\n") as f:
            f.write(f"# {cfg_path.name}, seed {seed}, seat {seat}. Assistant turns are dry-run placeholders.\n"
                    f"# Outcome of the placeholder game: survival {res['survival']:.2f}, deaths {res['deaths']}\n\n")
            for m in roster[seat].messages:
                f.write(f"===== {m['role'].upper()} =====\n{m['content']}\n\n")
    last = provs["A"].last
    bodies = {"anthropic_api": AnthropicAPIProvider("MODEL")._body(last),
              "openai_plan": OpenAIPlanProvider("MODEL")._body(last)}
    (out / f"{cfg_path.stem}_seed{seed}_seatA_request_bodies.json").write_text(
        json.dumps(bodies, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{cfg_path.name}: survival {res['survival']:.2f}, deaths {res['deaths']}")


if __name__ == "__main__":
    main()
