# Mixed-LLM Survival: a public granary with private hunger

This is a game-theory course project (NYU ECE-GY 6263). Language-model players share a public food store, negotiate over who gets released food, can quietly give or take units afterwards, and must eat to survive. Each player's hunger and stock are private, and every message is cheap talk. The project compares the players' behaviour with Baron–Ferejohn legislative bargaining (the daily proposal and vote) and Crawford–Sobel cheap talk (unverifiable claims about one's own need).

The repository holds:

- the engine;
- the rules, frozen on 2026-10-18;
- the twelve N = 5 pilot games played under those rules, with full logs and a replay page for each;
- the scripted and Baron–Ferejohn baselines;
- the commands to reproduce all of it.

## The game

Each of N players has a private stock and a private hunger level h ∈ {0, 1}. Every day runs through five stages:

1. **Negotiation**, up to 2 rounds. The day's supply arrives in a public store. Players post public messages in seat order, starting from the proposer, with two passes per round. The proposer then puts forward an allocation, with an optional one-line statement, and everyone votes. A proposal passes with all but one accept, and never with fewer than a simple majority. The proposer seat rotates every round.
2. **Fallback**, only if no proposal passed. Each player requests 0 or 1 unit, and a uniform draw settles requests that exceed the store.
3. **Transfers**, simultaneous. Each player gives one unit, takes one unit, or does nothing. All gives and takes are settled against the pre-transfer stocks with equal priority. A victim always learns of a loss, and learns the taker's identity with probability 0.5.
4. **Meal.** A player holding food may eat one unit.
5. **Hunger.** Eating sets h to 0. Not eating raises h by 1, and a player who reaches h = 2 dies at the end of the day. Deaths are announced, and a dead player's stock leaves the game unseen.

Messages "may be untrue: nothing checks them, and statements and promises are not enforced." Supply is i.i.d. uniform over a set the players are not told. The prompts are neutral and identical for every model. The full rules and their rationale are in [docs/frozen_design.md](docs/frozen_design.md). The exact text each seat receives is in [docs/frozen_rules_20261018/](docs/frozen_rules_20261018/).

## Layout

| Path | Contents |
|---|---|
| `sim/` | Engine. It covers configuration (`config.py`), game state (`state.py`), the phase machine and settlement (`engine.py`), and named random streams so every roster sees the same supply and hunger on a given seed (`rng.py`). It also covers per-player observations (`observe.py`), the rules and request text (`render.py`), reply schemas (`schemas.py`), the async game runner with JSONL logs and checkpoints (`runner.py`), metrics (`metrics.py`) and the replay page (`replay_html.py`). |
| `sim/agents/` | Player interfaces. These are the model providers (`anthropic_api.py`, `openai_plan.py`, `openai_api.py`, `deepseek_api.py`, `cli.py`) and the scripted reference players (`scripted.py`). |
| `configs/` | `pilot_n5_v2.yaml` (the pilots: N = 5, T = 8, supply {1, 2, 3, 4}) and `main.yaml` (the main condition, with the same frozen rules). |
| `prompts/` | `rules_en.md` (the rules template every seat receives) and `claim_coder.md` (the prompt of the blind claim coder). |
| `runs/` | The twelve pilot games. |
| `baselines/` | Scripted references, the ex-post planner bound and Baron–Ferejohn benchmarks, with their saved outputs. |
| `tools/` | Pilot runner, sign-in and key helpers, the OpenAI API key check, the prompt dry run and viewer, and claim coding and audit tables. |
| `tests/` | 110 tests: settlement, information redaction, order invariance, replay, rules, providers (mocked), and the frozen-rules snapshot. |
| `docs/` | The frozen design and the frozen prompt snapshot. |

## Setup

You need Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The engine, tests, baselines and replays need no network access and no credentials. Only live model seats and the claim coder call an API.

### Credentials for live seats

No keys are stored in this repository. The providers look for credentials in the environment first and then in a folder under your home directory, outside the project. They never write credentials to logs or run metadata.

| Seat kind | Credential |
|---|---|
| `claude` | `ANTHROPIC_API_KEY`, or the file `~/.gt6263/anthropic_api_key` |
| `deepseek` | `DEEPSEEK_API_KEY`, or the file `~/.gt6263/deepseek_api_key`. `python tools/save_deepseek_key.py` prompts for the key with hidden input, checks it, and saves it. |
| `openai` | A ChatGPT plan, through OpenAI's Sign in with ChatGPT flow for locally hosted apps. Run `python tools/chatgpt_plan_login.py` once: it opens the browser and stores the tokens in `~/.gt6263/chatgpt_plan/`. Use `start` / `finish URL` to sign in from another device. The logged games used this route. |
| `openai_api` | `OPENAI_API_KEY`, or the file `~/.gt6263/openai_api_key`. Same model, settings and request body as `openai`, billed to the API account. Run `python tools/check_openai_api.py` first (see [Without a ChatGPT plan](#without-a-chatgpt-plan)). |

## Check the installation

```bash
python -m pytest tests -q
```

All 110 tests should pass. `tests/test_frozen_rules.py` plays a dry-run game and compares every seat's full conversation with `docs/frozen_rules_20261018/`, so any change to the rules text or request wording fails the suite.

To see exactly what the models are sent, without calling any model:

```bash
python tools/dry_run_prompts.py configs/pilot_n5_v2.yaml --seed 1 --out dry_run
```

This writes each seat's conversation, plus the request body each provider would send, to `dry_run/`. To rebuild the conversation one seat received in a logged game:

```bash
python tools/show_prompt.py runs/pilot_n5_v2_cocoo_seed1 A --days 2
```

## Reproduce the pilot games

The pilots used `configs/pilot_n5_v2.yaml`, seeds 1–3, and four rosters. The roster tag lists the seat kinds of seats A to E: `c` is claude, `o` is openai (either route), `d` is deepseek.

| Tag | `--seats` |
|---|---|
| `ccccc` | `A=claude,B=claude,C=claude,D=claude,E=claude` |
| `ooooo` | `A=openai,B=openai,C=openai,D=openai,E=openai` |
| `ddddd` | `A=deepseek,B=deepseek,C=deepseek,D=deepseek,E=deepseek` |
| `cocoo` | `A=claude,B=openai,C=claude,D=openai,E=openai` |

One game:

```bash
python tools/run_pilot.py configs/pilot_n5_v2.yaml --seats A=claude,B=openai,C=claude,D=openai,E=openai --seed 1 --cap 3 --out runs/repro_cocoo_seed1
```

By default `--out` is `runs/pilot_n5_v2_<tag>_seed<S>`. The script refuses a folder that already holds a run, so pass a new `--out` to keep the logged games. `--cap` stops the game between steps before the estimated API spend (Anthropic, DeepSeek and OpenAI API) would exceed that many US dollars. ChatGPT-plan calls report no price. The logged pilots cost about $0.7–1.1 per ccccc game, $0.3–0.4 per cocoo game and $0.2–0.4 per ddddd game.

All twelve games (bash):

```bash
for s in 1 2 3; do
  python tools/run_pilot.py configs/pilot_n5_v2.yaml --seats A=claude,B=claude,C=claude,D=claude,E=claude --seed $s --out runs/repro_ccccc_seed$s
  python tools/run_pilot.py configs/pilot_n5_v2.yaml --seats A=openai,B=openai,C=openai,D=openai,E=openai --seed $s --out runs/repro_ooooo_seed$s
  python tools/run_pilot.py configs/pilot_n5_v2.yaml --seats A=deepseek,B=deepseek,C=deepseek,D=deepseek,E=deepseek --seed $s --out runs/repro_ddddd_seed$s
  python tools/run_pilot.py configs/pilot_n5_v2.yaml --seats A=claude,B=openai,C=claude,D=openai,E=openai --seed $s --out runs/repro_cocoo_seed$s
done
```

### Without a ChatGPT plan

Use the seat kind `openai_api` wherever a command says `openai`, with an OpenAI API key (see [Credentials](#credentials-for-live-seats)). It sends the same request to the same model with the same settings; only the credential and the billing differ. `meta.json` records the route of each seat (`openai-api:gpt-6.1-sol` instead of `openai-plan:gpt-6.1-sol`).

1. Check that the key can run the seat. This sends one short request with the seat's exact settings and costs well under one cent:

   ```bash
   python tools/check_openai_api.py
   ```

   It stops with the API's message if the key is missing or invalid, the account cannot use `gpt-6.1-sol`, or the account cannot get reasoning summaries.
2. Run games as above, for example:

   ```bash
   python tools/run_pilot.py configs/pilot_n5_v2.yaml --seats A=claude,B=openai_api,C=claude,D=openai_api,E=openai_api --seed 1 --cap 5 --out runs/repro_cocoo_api_seed1
   ```

Priced at standard API rates, the logged games' token counts come to $3.7–4.2 per all-`gpt-6.1-sol` game and $2.4–3.0 per cocoo game (Claude seats included), so set `--cap` above the default of 3. The estimate prices uncached input at the $2.50 cache-write rate rather than the $2.00 input rate, so it errs high. During a game, a bad key, a missing model or an exhausted balance stops the run with an error rather than letting seats fall back to default actions. Rate limits are retried.

**What reproduces exactly and what does not.** Some things depend only on the engine and the seed: the supply sequence, who starts hungry, the first proposer, and every random draw (fallback lottery, transfer contention, thief reveal). These are identical across rosters and reruns on a given seed, and that is what makes the rosters comparable. The models' replies are not deterministic, so a rerun produces a new game on the same environment, not a copy of the logged one. The logs in `runs/` are the record of the games reported.

### What a run folder holds

| File | Contents |
|---|---|
| `meta.json` | Config, seed and roster (provider and model per seat). |
| `events.jsonl` | Every engine event with its visibility list, including ground truth (`truth_*` events), which no player sees. |
| `calls.jsonl` | Every decision attempt: the observation sent, the raw reply, the model's reasoning text or summary when the provider returns one, the parsed action, any validation error, and token usage. |
| `actions.jsonl` | The accepted action per step. |
| `checkpoint.json` | Engine state after the last step. |
| `result.json` | Survival, whether everyone survived, deaths, priority gap, steps and cost. |
| `claims.jsonl` | Coded self-reports, one per public message (see below). |
| `replay.html` | Self-contained replay page. |

## Replays (progress boards)

Each run folder has a `replay.html`. The page follows the game as a moderator would. **God view** shows the hidden truth (hunger, stocks, transfers, meals, reasoning), and **Public view** shows only what every seat could see. The page is in English, and the CN button switches its labels to Chinese.

### Open a replay

GitHub shows an `.html` file as source code, so clicking `replay.html` on github.com does not show the page. Open the local copy instead. It is a single self-contained file and needs no install, server or network.

1. Get the repository. Either click **Code → Download ZIP** on the repository page and unzip it, or clone it:

   ```bash
   git clone https://github.com/XinyiZhangNYU/mixed-llm-survival.git
   ```

2. Open the folder of one game, for example `runs/pilot_n5_v2_cocoo_seed1/`.
3. Double-click `replay.html`. It opens in your default browser (Chrome, Edge, Firefox or Safari). If another program opens it, right-click it and choose **Open with** and then a browser.

### Watch a game while it runs

You can follow a game started with `tools/run_pilot.py` from day 1:

1. Start the game. Its folder is the `--out` path, or `runs/pilot_n5_v2_<tag>_seed<S>` by default.
2. Open `replay.html` in that folder, as above. The runner writes the page before the first model reply, then rebuilds it after every step: each message, proposal, vote, transfer and meal.
3. Leave the page open. While the game runs it shows **Game in progress**, reloads itself every 15 seconds, and scrolls to the latest entry. It stops reloading when the game ends.

To rebuild a page from a run's logs by hand:

```bash
python -m sim.replay_html runs/pilot_n5_v2_cocoo_seed1
```

### Hosted copies

The links below open hosted copies of the twelve pilot replays. They are private and open only for accounts they have been shared with. The local files are identical.

| Run | Survival | Priority gap | Replay |
|---|---|---|---|
| ccccc seed 1 | 1.00 | 0.41 | [link](https://claude.ai/artifact/9jzaJzLQXMm8AcLLkJhEgP) |
| ccccc seed 2 | 0.60 | 0.15 | [link](https://claude.ai/artifact/Go12wyzoqovnuxUZmJeyCp) |
| ccccc seed 3 | 0.85 | 0.09 | [link](https://claude.ai/artifact/CXpXCpRgxpQkjCfqyXUiS5) |
| ooooo seed 1 | 0.75 | 0.10 | [link](https://claude.ai/artifact/StT5cyjSqP5FtiKkw1aa9c) |
| ooooo seed 2 | 0.825 | 0.14 | [link](https://claude.ai/artifact/SWHvHcvL5G73jzcfVfNRVb) |
| ooooo seed 3 | 0.775 | 0.12 | [link](https://claude.ai/artifact/1icmCp3VeLnoU2RefGMcjn) |
| ddddd seed 1 | 0.75 | 0.22 | [link](https://claude.ai/artifact/AmJ5FL5hxUwvDKbKfnfYJF) |
| ddddd seed 2 | 0.75 | 0.05 | [link](https://claude.ai/artifact/5bajqtkfKPYnt1qBXQdk4U) |
| ddddd seed 3 | 0.675 | 0.05 | [link](https://claude.ai/artifact/XhS8kgfRuA9a2fhfdu7XAM) |
| cocoo seed 1 | 0.70 | 0.15 | [link](https://claude.ai/artifact/2AMPMWFKYHfk7uyzighZDS) |
| cocoo seed 2 | 0.775 | 0.14 | [link](https://claude.ai/artifact/Sz24o7AiJ5euTe15hTMbd1) |
| cocoo seed 3 | 0.90 | 0.24 | [link](https://claude.ai/artifact/Stb7WrNLxAJPWdR8b429A3) |

Survival is player-days alive divided by N·T. Priority gap is the share of released units that reached players truly at risk (h = 1, no stock), minus the share an all-request lottery would send them. It counts only days that start with at least two players alive. Three games per roster is a pilot, not an estimate.

## Claim coding and audit tables

Each public message is sent alone to a blind coder: DeepSeek-V4.1-Flash-1017 with the fixed prompt `prompts/claim_coder.md`. The coder never sees the true state. It records what the speaker says about their own h and stock, whether they deny past takes, and whether they promise not to take today. The coder needs `DEEPSEEK_API_KEY`. Messages already coded are skipped.

```bash
python tools/code_claims.py runs/pilot_n5_v2_*_seed* --cap 1
python tools/claims_report.py runs/pilot_n5_v2_*_seed*
python tools/audit_days.py audit_out runs/pilot_n5_v2_*_seed*
```

`claims_report.py` compares claims with the truth for each speaker model: h and stock accuracy, denials, broken promises, who receives units by claimed and true need, and how stock is disclosed. `audit_days.py` writes `days.csv` (one row per living player-day), `takes.csv` (one row per take attempt) and `flags.txt` to the output folder. The coded `claims.jsonl` from the logged games is already in each run folder, so the last two commands run offline.

## Baselines

These need no credentials. Each script reads `configs/pilot_n5_v2.yaml`, and the saved output sits next to it.

```bash
python baselines/references_n5.py    # -> baselines/references_n5_output.txt
python baselines/su_bf.py            # -> baselines/su_bf_output.txt
python baselines/su_bf.py --alpha 0.05   # -> baselines/su_bf_alpha0.05.txt (also 0.01, 0.2)
```

- `references_n5.py`:
  - **Scripted rosters**, all five seats scripted over 200 seeds: need-first (truthful), claim-hungry, thief, all-request lottery and equal split. Seeds 1–3 match the pilots' supply and hunger.
  - **Ex-post planner bound.**
  - **Linear-utility Baron–Ferejohn one-day benchmark**, compared with the observed proposals.
- `su_bf.py`: a Baron–Ferejohn one-day benchmark whose payoff is the game's own objective, expected days alive. It uses a need-first or lottery continuation and two tie-breaks. For every observed LLM proposal, it gives the prediction from that day's true state and a 200-seed run played by the benchmark. `--alpha` adds weight on other players' expected days as a sensitivity check.

The saved outputs were produced with the current engine, which ends a game early when one player is left (see [docs/frozen_design.md](docs/frozen_design.md)). Reruns reproduce them exactly.

Both benchmarks are interpretive. They assume complete information about the day and ignore storage and transfers, while the game has private hunger and an undisclosed supply set.

## Frozen design

The rules, the open parameters of the main experiment, the model settings, the metrics and the policy for theory sub-studies are in [docs/frozen_design.md](docs/frozen_design.md). In short:

- The rules are fixed as of 2026-10-18, and `tests/test_frozen_rules.py` enforces them.
- The main run's parameters, rosters, settings and game counts are fixed before the main runs start.
- Each sub-study changes one pre-declared item, uses the same seeds, and is reported alongside the main run, never in place of it.
