"""Print the exact conversation one seat's model received in a logged run.

Rebuilds LLMAgent's message list from the run's logs: the system prompt (rules filled in
from meta.json), then for each decision attempt the user turn (rendered from the logged
observation, or the "not accepted" feedback on a retry) and the model's raw reply.
This is what our code sends; a provider such as the Claude CLI may add its own text on top.

Usage: python tools/show_prompt.py <run_dir> <seat> [--days N]   (N: stop after day N)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.config import Config  # noqa: E402
from sim.observe import Observation  # noqa: E402
from sim.render import render_observation, rules_text  # noqa: E402


def conversation(run: Path, seat: str, max_day: int | None = None) -> list[dict]:
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    cfg = Config.from_dict(meta["cfg"]) if hasattr(Config, "from_dict") else Config(**{
        k: tuple(v) if isinstance(v, list) else v for k, v in meta["cfg"].items()})
    msgs = [{"role": "system", "content": rules_text(cfg, seat)}]
    last_err = None
    for line in (run / "calls.jsonl").read_text(encoding="utf-8").splitlines():
        c = json.loads(line)
        if c.get("agent") != seat or "phase" not in c:
            continue
        if max_day is not None and c["day"] > max_day:
            break
        if c["attempt"] == 0:
            user = render_observation(Observation(**c["observation"]))
        else:
            user = (f"Your previous reply was not accepted: {last_err}. "
                    "Reply again with a single JSON object in the requested format.")
        msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": c["raw"] or ""}]
        last_err = c.get("error")
    return msgs


if __name__ == "__main__":
    args = sys.argv[1:]
    max_day = int(args[args.index("--days") + 1]) if "--days" in args else None
    for m in conversation(Path(args[0]), args[1], max_day):
        print(f"===== {m['role'].upper()} =====\n{m['content']}\n")
