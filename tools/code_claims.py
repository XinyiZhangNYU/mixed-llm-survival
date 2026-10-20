"""Code the self-reports in a game's public messages, blind to the true state.

Usage:
  python tools/code_claims.py <run_dir> [<run_dir> ...] [--cap USD]

Each public message is sent alone, with only the speaker's id, to a coder model
(DeepSeek-V4.1-Flash-1017 via the DeepSeek API) with the fixed prompt prompts/claim_coder.md.
The coder records what the speaker says about their own h, their own stock, whether they
deny earlier takes, and whether they promise not to take today. It never sees the true state.

Results go to <run_dir>/claims.jsonl, one line per message, keyed by the event seq. Messages
already coded are skipped, so a rerun only codes what is missing or failed.
"""
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.agents.deepseek_api import DeepSeekAPIProvider  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROMPT = (ROOT / "prompts" / "claim_coder.md").read_text(encoding="utf-8")
CODER = "deepseek-flash"
PRICE = {"input": 0.15, "cache_hit": 0.003, "output": 0.6}   # off-peak, 2026-10-18
KEYS = ("h", "stock", "denies_taking", "promises_no_take", "evidence")
CONCURRENCY = 8


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if set(d) != set(KEYS):
        return None
    if d["h"] not in (0, 1, None) or not (d["stock"] is None or isinstance(d["stock"], int)):
        return None
    if d["denies_taking"] not in (True, None) or d["promises_no_take"] not in (True, "conditional", None):
        return None
    return d


async def code_one(p, sem, e):
    prompt = PROMPT.format(speaker=e["data"]["from"], text=e["data"]["text"])
    async with sem:
        for _ in range(3):
            text, meta = await p.complete([{"role": "user", "content": prompt}])
            d = parse(text)
            if d is not None:
                return e, d, meta
    meta["raw_reply"] = text[:2000]
    return e, None, meta


async def code_run(run: Path, cap: float, spent: list) -> None:
    out = run / "claims.jsonl"
    kept = []
    if out.exists():                                  # keep coded records; failed ones are retried
        kept = [r for r in map(json.loads, out.read_text(encoding="utf-8").splitlines()) if r["coded"]]
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    done = {r["seq"] for r in kept}
    ev = [json.loads(l) for l in (run / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    todo = [e for e in ev if e["kind"] == "message" and e["seq"] not in done and e["data"]["text"].strip()]
    roster = json.loads((run / "meta.json").read_text(encoding="utf-8"))["roster"]
    p = DeepSeekAPIProvider(CODER, max_tokens=8000, reasoning_effort="high", price_per_mtok=PRICE)
    sem = asyncio.Semaphore(CONCURRENCY)
    bad = 0
    with out.open("a", encoding="utf-8") as f:
        for coro in asyncio.as_completed([code_one(p, sem, e) for e in todo]):
            e, d, meta = await coro
            spent[0] += meta.get("cost_usd") or 0.0
            if d is None:
                bad += 1
            rec = {"seq": e["seq"], "day": e["day"], "round": e["round"], "speaker": e["data"]["from"],
                   "speaker_model": roster.get(e["data"]["from"]), "coder": f"deepseek-api:{CODER}",
                   "coded": d, "request_id": meta.get("request_id"), "error": meta.get("error"), "raw_reply": meta.get("raw_reply")}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if spent[0] > cap:
                raise SystemExit(f"cost cap ${cap} reached")
    print(f"{run.name}: coded {len(todo) - bad}/{len(todo)} new messages, {bad} unparsed; "
          f"running cost ${spent[0]:.4f}")


def main():
    args = sys.argv[1:]
    cap = 1.0
    if "--cap" in args:
        i = args.index("--cap"); cap = float(args[i + 1]); del args[i:i + 2]
    spent = [0.0]
    for a in args:
        asyncio.run(code_run(Path(a), cap, spent))


if __name__ == "__main__":
    main()
