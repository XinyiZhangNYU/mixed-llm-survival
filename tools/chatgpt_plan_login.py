"""Sign in with ChatGPT so OpenAIPlanProvider can use the ChatGPT plan.

Usage:
  python tools/chatgpt_plan_login.py              sign in on this machine (opens the browser)
  python tools/chatgpt_plan_login.py start        print an authorize URL to open on any device
  python tools/chatgpt_plan_login.py finish URL   complete it with the 127.0.0.1 URL the browser
                                                  landed on after allowing access
Prints only non-secret facts (granted scopes, plan type); tokens stay in
~/.gt6263/chatgpt_plan/credentials.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sim.agents.openai_plan import finish_login, login, start_login  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["start"]:
        print(start_login(), flush=True)
    elif args[:1] == ["finish"] and len(args) == 2:
        print(json.dumps(finish_login(args[1]), indent=1), flush=True)
    else:
        print(json.dumps(login(), indent=1), flush=True)
