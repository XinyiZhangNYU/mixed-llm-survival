# Frozen design

The game rules were frozen on 2026-10-18. They include the rules text the players see and the wording of every request. The frozen rules are the rules of `configs/pilot_n5_v2.yaml`, which the twelve N = 5 pilot games in `runs/` were played under. `configs/main.yaml` carries the same rule values, and `tests/test_frozen_rules.py` checks both.

The freeze has two layers.

- **Main experiment.** The rules below are fixed. The parameters marked open in `configs/main.yaml` are T and the supply set. Those parameters, the rosters, the model settings, the game counts and the metric definitions are all fixed before the main runs start (2026-10-31). After that, nothing in the main experiment changes.
- **Theory sub-studies.** A sub-study may change one item relative to the main run, such as `vote_rule`, `message_order`, a parameter, or the initial hunger setup. Each sub-study must meet four conditions:
  - its item is declared in advance;
  - it uses the same supply seeds as the main run;
  - it is fixed before its own runs;
  - it is reported in full alongside the main run, never in place of it.

## Frozen rule items

| Item | Value | Meaning |
|---|---|---|
| `rounds` | 2 | Negotiation rounds per day before the fallback lottery. |
| `message_order` | sequential | Players speak in seat order, starting from the round's proposer. |
| `message_passes` | 2 | Each living player speaks twice per round, so accusations can be answered in the same round. |
| `vote_rule` | all_but_one | A proposal passes with n − 1 accepts, never fewer than a simple majority. With two players alive, both must accept. |
| `proposal_statement` | true | The proposer may attach one public sentence explaining the proposal. |
| `deliberation_prompt` | true | Every request ends with one fixed sentence asking the player to review what they have seen before replying. The sentence is the same for every seat and model. |
| `public_record` | true | From day 2, each day opens with a record of earlier announced events: proposals, votes, releases and deaths. It contains nothing that was not already announced. |
| `supply_feasible_only` | false | Supply is i.i.d. uniform over the supply set, with no filter guaranteeing that everyone can survive. |
| `disclose_supply_values` | false | Players learn only that the daily amount is random and may differ across days. They see each day's amount when it arrives. |
| `q_reveal_thief` | 0.5 | A player whose unit is taken always learns of the loss. With probability 0.5 they also learn who took it. |
| `votes_public` | true | Individual votes are announced. |
| `gift_reveals_giver` | true | A gift recipient learns who gave the unit. |
| `allocation_must_exhaust_store` | false | A proposal may leave food in the store, and that food carries over. |
| `max_retries` | 2 | Retries after an invalid reply, each with a "not accepted" notice. |
| `max_message_chars` | 1200 | Longest public message. |
| Invalid-reply defaults | engine defaults | These apply after the retries run out: empty message, no proposal, reject, request 0, no transfer, no meal. |
| Rules text and request wording | snapshot | `docs/frozen_rules_20261018/seat{A..E}.txt` holds the full conversation each seat receives in a dry-run game (N = 5, T = 8, seed 1). It covers every wording path: failed vote, fallback, gift, take and death. |

Rules fixed since the proposal and unchanged by the freeze:

- New food enters the public store and reaches private stocks only through an accepted proposal or the fallback lottery.
- Stocks and hunger are private. Eating sets h to 0. Not eating raises h by 1. A player whose h reaches 2 dies at the end of the day.
- Transfers are simultaneous: give one unit, take one unit, or do nothing. All gives and takes are settled against the pre-transfer stock with equal priority, and a uniform draw decides which succeed when they exceed that stock. A unit received in the transfer stage cannot be passed on in the same stage.
- Deaths are announced at the end of the day. A dead player's stock leaves the game and is not revealed.
- The first proposer is random. After that, the proposer moves to the next living seat every round.
- The prompts state only the rules and the objective ("stay alive for as many days as possible"). The wording is neutral, contains no moral framing, and is identical for every model.

## Why the rules are what they are

Each item below was settled during the pilots, before the freeze.

1. **No survival guarantee in the rules text.** An early design filtered supply to sequences under which perfect coordination keeps everyone alive, and the rules text said so. Players reasoned from that sentence from day 1, and with no transfers every h could be computed from the public record. That turned private hunger into public information and removed the strategic content of cheap talk. The filter is now off. `rules_text` refuses a filtered config, and the filter is used only by scripted reference runs.
2. **Supply values undisclosed.** Not knowing the environment strengthens both the motive to keep a reserve and the motive to misreport. The cost is that distrust from environmental uncertainty cannot be fully separated from distrust of other players. Baron–Ferejohn and Crawford–Sobel both assume the environment is common knowledge, so they serve here as interpretive benchmarks.
3. **"Player", not "agent".** The second word could cue a shared AI identity. Every model-facing string uses "player".
4. **Message credibility wording.** The rules state the fact plainly and point at no one: messages "may be untrue: nothing checks them, and statements and promises are not enforced."
5. **Negotiation rules v2.** Early N = 4 pilots used simultaneous one-pass messages and a unanimous vote. Almost every day passed in round 1 with no exchange. An accused player could not be answered in the same round. A single thief could veto every proposal that excluded them. Players also made bookkeeping errors about who had eaten. Five changes were introduced together:
   - sequential messages with two passes;
   - the n − 1 vote rule;
   - the proposer statement;
   - the deliberation sentence;
   - the daily public record.

   In the three cocoo pilot games under the frozen rules, the n − 1 rule decided the outcome twice: a proposal passed over one rejecting vote in seed 1 (day 2) and in seed 2 (day 6). All 24 proposals in those games carried a statement.
6. **Initial hunger k = 1** (a parameter, not a rule). With k = 2 and empty stocks, a day-1 supply of 1 kills a player no matter what anyone decides.
7. **Sole-survivor early end** (an engine change made after the freeze, with the rules text unchanged). Suppose that at the end of a day N > 1, one player is alive, and every supply value is at least 1. Then the game ends that day, and the survivor counts as alive on the days not played: there is no one left to negotiate with or take from, and at least one unit arrives each day. The `game_end` event records `reason` and `days_played`. Among the twelve pilots, only ccccc seed 2 had a sole survivor (days 7–8), and that player survived both days, so its survival figure is unchanged. Its priority gap counts days 1–6 only (see Metrics).

## Considered and not adopted

| Option | Why not |
|---|---|
| Persona prompts such as "you are selfish" | They break the neutral, identical-prompt principle and would measure persona compliance. A code-controlled scripted thief (`ThiefAgent`) is the alternative, as a separately labelled condition. |
| Revealing every taker publicly, or a random end date instead of a known T | Either changes the monitoring structure or the end-game effect. That is a different research question and belongs in its own treatment arm. |
| A public gift record | It makes takes partly detectable, weakens the cheap-talk benchmark, and reveals truth in real time. |
| Telling players to "watch for anomalies" | It steers behaviour. When asked directly, all tested models can derive the inferable facts. Whether they say them publicly is part of the measured behaviour. |

## Model settings

A seat's model and settings are parameters, and they stay fixed for a model throughout the main experiment:

| Seat kind | Model | Settings |
|---|---|---|
| `claude` | claude-sonnet-5-5 (Anthropic API) | effort high, adaptive thinking with summarized display |
| `openai` | gpt-6.1-sol (Responses API on a ChatGPT plan, via Sign in with ChatGPT) | reasoning effort xhigh, reasoning summary auto |
| `deepseek` | deepseek-flash, DeepSeek-V4.1-Flash-1017 (DeepSeek API) | thinking on, reasoning effort high; raw reasoning is logged |

Whether thinking is on changes behaviour, so the settings are reported with the results.

## Metrics

- **Survival:** player-days alive at the end of each day, divided by N·T (`sim/metrics.py`).
- **Priority gap:** the share of released units that reach players truly at risk at the start of the day (h = 1, no stock), minus the share an all-request lottery would send them on the same days. Only days that start with at least two players alive count: a lone player has no one to be prioritized over, and the value is then the same with or without the sole-survivor early end. The definition is provisional until the main-run freeze.
- **Claim accuracy:** a blind coder (`tools/code_claims.py`, with the prompt in `prompts/claim_coder.md`) records what each public message says about the speaker's own h, stock, past takes and promises. `tools/claims_report.py` compares these claims with the true state.
