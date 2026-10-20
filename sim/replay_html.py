"""Game replay page in the style of a werewolf moderator app.

build(run_dir) reads a run's logs (meta.json, events.jsonl, calls.jsonl, result.json) and
writes run_dir/replay.html: a self-contained page where the moderator announces each phase
(dawn, negotiation rounds, fallback, evening transfers, night meal, deaths) and the seats'
messages, proposals and votes appear as a group chat.

Two views, switchable on the page:
  God view     everything, including hidden ground truth (hunger, stock, transfers, meals),
               private notices, each seat's private `reasoning` field, and provider thinking summaries.
  Public view  only events visible to all seats, i.e. what every seat could see.

The runner writes the page before the first request and rebuilds it after every step (each
message, proposal, vote, transfer and meal), so an unfinished game can be watched as it is
played (the page reloads itself every 15 s until the game ends). Reads logs only; nothing is re-run.

Usage: python -m sim.replay_html <run_dir> [<run_dir> ...]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PAGE = "replay.html"
RELOAD_S = 15


def _load_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:      # a line still being written
                pass
    return out


def _seat_models(calls: list[dict], roster: dict) -> dict:
    """Model name actually reported by the provider, else the roster label."""
    found = {}
    for c in calls:
        m = c.get("meta")
        if isinstance(m, dict) and c.get("agent") not in found:
            models = [x for x in (m.get("models") or []) if x]
            if models:
                found[c["agent"]] = models[0]
    return {a: _display_name(found.get(a, lbl)) for a, lbl in roster.items()}


# Provider aliases that do not carry a version. The mapping is what the provider's own
# documentation said on the date given; check it again before new runs.
ALIASES = {"deepseek-flash": "DeepSeek-V4.1-Flash-1017"}   # DeepSeek pricing page, 2026-10-18


def _display_name(model_id: str) -> str:
    """Claude API ids write the version with a hyphen (claude-sonnet-5-5); show it as 5.5.
    Unversioned aliases are shown with the full model name they pointed to."""
    if model_id in ALIASES:
        return ALIASES[model_id]
    m = re.fullmatch(r"claude-([a-z]+)-(\d+)-(\d+)", model_id or "")
    return f"Claude {m[1].capitalize()} {m[2]}.{m[3]}" if m else model_id


def _call_index(calls: list[dict]) -> dict:
    """Per decision: the accepted attempt's private reasoning, attempt count, errors, default."""
    where = {}
    idx: dict[str, dict] = {}
    for c in calls:
        if "default_used" in c:
            d = where.get(c["step"])
            if d:
                k = f"{d[0]}|{d[1]}|{d[2]}|{c['agent']}"
                idx.setdefault(k, {"n": 0, "err": []})["def"] = c["default_used"]
            continue
        rnd = c.get("round") if c["phase"] in ("MESSAGE", "PROPOSE", "VOTE") else ""
        mp = (c.get("observation") or {}).get("message_pass")
        if c["phase"] == "MESSAGE" and mp:
            rnd = f"{rnd}.{mp}"   # several message passes per round: key each pass separately
        where[c["step"]] = (c["day"], c["phase"], rnd)
        k = f"{c['day']}|{c['phase']}|{rnd}|{c['agent']}"
        e = idx.setdefault(k, {"n": 0, "err": []})
        e["n"] += 1
        if c.get("error"):
            e["err"].append(str(c["error"])[:300])
        else:
            e["r"] = c.get("reasoning") or ""
        m = c.get("meta") if isinstance(c.get("meta"), dict) else {}
        if not c.get("error"):
            e["th"] = m.get("thinking") or ""
        e["tok"] = e.get("tok", 0) + int(m.get("input_tokens") or 0) + int(m.get("output_tokens") or 0)
    return idx


def payload(run: Path) -> dict:
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    events = _load_jsonl(run / "events.jsonl")
    calls = _load_jsonl(run / "calls.jsonl")
    res_p = run / "result.json"
    result = json.loads(res_p.read_text(encoding="utf-8")) if res_p.is_file() else None
    finished = result is not None or any(e["kind"] == "game_end" for e in events)
    return {"run": run.name, "cfg": meta["cfg"], "seed": meta["seed"], "roster": meta["roster"],
            "models": _seat_models(calls, meta["roster"]), "events": events,
            "calls": _call_index(calls), "result": result, "finished": finished}


HEAD = ('<!doctype html>\n<html><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>:root{padding-block:env(safe-area-inset-top,0px) env(safe-area-inset-bottom,0px)}'
        'body{margin:0}[hidden]{display:none!important}</style>\n')


def build(run_dir: str | Path, out: str | Path | None = None, standalone: bool = True) -> Path:
    """standalone=False leaves out the document skeleton (for hosts that add their own)."""
    run = Path(run_dir)
    data = payload(run)
    js = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    reload_tag = "" if data["finished"] else f'<meta http-equiv="refresh" content="{RELOAD_S}">'
    html = (TEMPLATE.replace("__RELOAD__", reload_tag)
            .replace("__TITLE__", f"Granary Replay {run.name}")
            .replace("__DATA__", js))
    if standalone:
        html = HEAD + html.replace("</style>\n\n<div", "</style>\n</head><body>\n<div", 1) + "</body></html>\n"
    dst = Path(out) if out else run / PAGE
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(html, encoding="utf-8")
    tmp.replace(dst)
    return dst


TEMPLATE = r"""<title>__TITLE__</title>
<meta name="claude:viewport-fit-sides" content="cover">
__RELOAD__
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500&family=IBM+Plex+Sans:wght@400;500;600&family=Zilla+Slab:wght@600;700&display=swap">
<style>
/* Layout: one centered feed like a group chat; the moderator speaks in full-width night bands,
   seats speak in bubbles, hidden truth sits in dashed "god view" boxes. */
:root {
  --bg: #eef1ee; --panel: #ffffff; --ink: #1c2320; --muted: #5b6761; --line: #d6ddd8;
  --grain: #9a5f14; --grain-soft: #f6ead7;
  --night: #24304d; --night-ink: #eef2fb; --night-muted: #b7c1da;
  --god: #7046a0; --god-soft: #f3edf9;
  --ok: #2b7a4b; --ok-soft: #e3f2e8; --warn: #9a6a0c; --warn-soft: #f7edd5; --bad: #b0353a; --bad-soft: #f8e3e3;
  --s0: #2f6fb0; --s1: #b0442f; --s2: #2f8a63; --s3: #8a4fb0; --s4: #b07a1c; --s5: #2f8f9a; --s6: #a83f78; --s7: #5d6b2f;
  --f-display: "Zilla Slab", Georgia, "Songti SC", "SimSun", serif;
  --f-body: "IBM Plex Sans", system-ui, -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111613; --panel: #1a211d; --ink: #e5ebe7; --muted: #98a59e; --line: #2b3430;
  --grain: #e2a656; --grain-soft: #2c2416; --night: #233052; --night-ink: #eef2fb; --night-muted: #a9b5d3;
  --god: #b892e6; --god-soft: #241d2e; --ok: #63c48c; --ok-soft: #16271d; --warn: #e2b450; --warn-soft: #2a2414;
  --bad: #ec7377; --bad-soft: #2e1a1b; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #111613; --panel: #1a211d; --ink: #e5ebe7; --muted: #98a59e; --line: #2b3430;
  --grain: #e2a656; --grain-soft: #2c2416; --night: #233052; --night-ink: #eef2fb; --night-muted: #a9b5d3;
  --god: #b892e6; --god-soft: #241d2e; --ok: #63c48c; --ok-soft: #16271d; --warn: #e2b450; --warn-soft: #2a2414;
  --bad: #ec7377; --bad-soft: #2e1a1b; color-scheme: dark }
body { background: var(--bg); color: var(--ink); font-family: var(--f-body); font-size: 15px; line-height: 1.55; }
.wrap { max-width: 820px; margin: 0 auto; padding-left: max(16px, env(safe-area-inset-left, 0px));
  padding-right: max(16px, env(safe-area-inset-right, 0px)); padding-block: 20px 48px; display: grid; gap: 18px; }
h1 { font-family: var(--f-display); font-size: 1.9rem; line-height: 1.15; margin: 0; text-wrap: balance; }
.sub { color: var(--muted); font-size: .9rem; }
.mono, .num { font-family: var(--f-mono); font-variant-numeric: tabular-nums; }
.label { font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }

/* header */
.head { display: grid; gap: 10px; }
.outcome { display: flex; flex-wrap: wrap; gap: 8px; }
.pill { display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px; border-radius: 999px; font-size: .8rem;
  border: 1px solid var(--line); background: var(--panel); white-space: nowrap; }
.pill.ok { color: var(--ok); background: var(--ok-soft); border-color: transparent; }
.pill.bad { color: var(--bad); background: var(--bad-soft); border-color: transparent; }
.pill.live { color: var(--warn); background: var(--warn-soft); border-color: transparent; }
.seats { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 8px; }
.seat { display: flex; gap: 10px; align-items: center; padding: 8px 10px; background: var(--panel); border: 1px solid var(--line);
  border-radius: 10px; min-width: 0; }
.seat .who { min-width: 0; }
.seat .model { font-size: .75rem; color: var(--muted); overflow-wrap: anywhere; }
.seat.dead { opacity: .55; }
.av { flex: none; width: 34px; height: 34px; border-radius: 50%; display: grid; place-items: center; color: #fff;
  font-family: var(--f-display); font-weight: 700; font-size: 1.05rem; }
.av.sm { width: 22px; height: 22px; font-size: .75rem; }

/* controls */
.bar { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 5; background: var(--bg); padding-block: 8px;
  display: grid; gap: 8px; border-bottom: 1px solid var(--line); }
.ctrls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.seg { display: inline-flex; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; background: var(--panel); }
.seg button { font: inherit; font-size: .85rem; border: 0; background: transparent; color: var(--ink); padding: 5px 12px; cursor: pointer; }
.seg button[aria-pressed="true"] { background: var(--night); color: var(--night-ink); }
.seg button:focus-visible, .days button:focus-visible { outline: 2px solid var(--grain); outline-offset: 1px; }
.check { display: inline-flex; gap: 6px; align-items: center; font-size: .85rem; cursor: pointer; }
.days { display: flex; gap: 5px; overflow-x: auto; padding-bottom: 2px; }
.days button { flex: none; font: inherit; font-family: var(--f-mono); font-size: .75rem; border: 1px solid var(--line);
  background: var(--panel); color: var(--ink); border-radius: 6px; padding: 3px 0; width: 42px; cursor: pointer; position: relative; }
.days button .r { display: block; font-size: .65rem; color: var(--muted); }
.days button.r1 { border-color: var(--ok); } .days button.r2, .days button.r3 { border-color: var(--warn); }
.days button.fb { border-color: var(--bad); }
.days button.cur { background: var(--night); color: var(--night-ink); box-shadow: 0 0 0 2px var(--grain); }
.days button.cur .r { color: var(--night-muted); }
.legend { font-size: .75rem; color: var(--muted); }
.days button.death::after { content: ""; position: absolute; top: 3px; right: 3px; width: 6px; height: 6px; border-radius: 50%; background: var(--bad); }

/* feed */
.feed { display: grid; gap: 10px; }
.band { background: var(--night); color: var(--night-ink); border-radius: 12px; padding: 12px 16px; display: grid; gap: 6px; margin-top: 8px; }
.band h2 { margin: 0; font-family: var(--f-display); font-size: 1.25rem; display: flex; gap: 10px; align-items: center; }
.band h2 svg { flex: none; }
.band p { margin: 0; }
.band .muted { color: var(--night-muted); font-size: .88rem; }
.band.end { background: var(--grain); color: #fff; }
.sub-band { display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: .82rem; letter-spacing: .06em; text-transform: uppercase; margin-top: 6px; }
.sub-band::before, .sub-band::after { content: ""; flex: 1; height: 1px; background: var(--line); }
.say { display: flex; gap: 10px; align-items: flex-start; }
.say .body { min-width: 0; flex: 1; display: grid; gap: 4px; }
.say .meta { font-size: .8rem; color: var(--muted); display: flex; gap: 8px; flex-wrap: wrap; align-items: baseline; }
.say .meta b { color: var(--ink); font-weight: 600; }
.bubble { background: var(--panel); border: 1px solid var(--line); border-radius: 4px 14px 14px 14px; padding: 9px 13px;
  white-space: pre-wrap; overflow-wrap: anywhere; }
.bubble.empty { color: var(--muted); font-style: italic; }
.card { border: 1px solid var(--line); background: var(--panel); border-radius: 10px; padding: 10px 14px; display: grid; gap: 8px; }
.card.prop { border-color: var(--grain); background: var(--grain-soft); }
.card .ttl { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; font-weight: 600; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; }
.chip { display: inline-flex; gap: 6px; align-items: center; border: 1px solid var(--line); background: var(--panel);
  border-radius: 999px; padding: 2px 9px 2px 3px; font-size: .82rem; }
.chip .n { font-family: var(--f-mono); font-variant-numeric: tabular-nums; }
.chip.zero { opacity: .55; }
.chip.yes { border-color: var(--ok); color: var(--ok); } .chip.no { border-color: var(--bad); color: var(--bad); }
.verdict { margin-left: auto; }
.god { border: 1.5px dashed var(--god); background: var(--god-soft); border-radius: 10px; padding: 8px 12px; font-size: .88rem; display: grid; gap: 5px; }
.god .tag { color: var(--god); font-size: .7rem; letter-spacing: .08em; text-transform: uppercase; font-weight: 600; }
.why { border-left: 2px solid var(--god); padding: 2px 0 2px 10px; color: var(--muted); font-size: .85rem; white-space: pre-wrap; overflow-wrap: anywhere; }
.why .tag { color: var(--god); font-size: .7rem; letter-spacing: .08em; text-transform: uppercase; font-weight: 600; margin-right: 6px; }
.flag { color: var(--warn); font-size: .78rem; }
.statgrid { overflow-x: auto; }
.statgrid table { border-collapse: collapse; font-family: var(--f-mono); font-size: .78rem; font-variant-numeric: tabular-nums; }
.statgrid td, .statgrid th { padding: 2px 9px; text-align: center; border-bottom: 1px solid var(--line); }
.statgrid th { font-weight: 500; color: var(--muted); }
.risk { color: var(--bad); font-weight: 600; }
.is-public .godonly { display: none; }
.no-why .why { display: none; }
@media (max-width: 480px) { h1 { font-size: 1.5rem; } .band h2 { font-size: 1.1rem; } }
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }
</style>

<div class="wrap" id="app">
  <header class="head">
    <div class="label" id="eyebrow"></div>
    <h1 id="title"></h1>
    <div class="sub" id="cfgline"></div>
    <div class="outcome" id="outcome"></div>
    <div class="seats" id="seats"></div>
  </header>
  <div class="bar">
    <div class="ctrls">
      <div class="seg" role="group" aria-label="view">
        <button id="v-god" type="button" aria-pressed="true"></button>
        <button id="v-pub" type="button" aria-pressed="false"></button>
      </div>
      <label class="check godonly-ctl"><input type="checkbox" id="c-why"> <span id="c-why-l"></span></label>
      <div class="seg" role="group" aria-label="language">
        <button id="l-en" type="button" aria-pressed="true">EN</button>
        <button id="l-zh" type="button" aria-pressed="false">CN</button>
      </div>
    </div>
    <nav class="days" id="days" aria-label="days"></nav>
    <div class="legend" id="legend"></div>
  </div>
  <main class="feed" id="feed"></main>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
(function () {
const D = JSON.parse(document.getElementById("data").textContent);
const T = {
  zh: {
    eyebrow: "粮仓生存博弈 · 对局复盘", god: "上帝视角", pub: "公开视角", why: "显示思考内容和私下备注", whyTip: "思考内容：Claude 和 OpenAI 席位是服务商给出的思考摘要（Claude 的 thinking 摘要、OpenAI 的 reasoning summary），不是原始思维链，摘要由另一个模型生成，不保证完整；DeepSeek 席位是接口返回的原始思维链（reasoning_content）。私下备注：模型在回复 JSON 里自愿附带的 reasoning 字段，规则告诉它其他玩家看不到。两者都显示在对应的发言、提案、投票、傍晚行动和用餐旁边。",
    seed: "种子", days: "天", rounds: "轮/天", players: "名玩家", supply: "每日供给",
    live: "对局进行中（每 15 秒自动刷新）", survived: "全部存活", deathsN: n => `${n} 人死亡`, survival: "存活率",
    gap: "优先差距（暂定）", cost: "名义成本",
    alive: "存活", diedOn: d => `第 ${d} 天死亡`,
    gameStart: "游戏开始", gameStartP: (n, k) => `${n} 名玩家入座。其中 ${k} 人开局处于饥饿状态，但只有他们自己知道。`,
    gtStart: "开局真实状态", supplySeq: "供给序列",
    dawn: d => `第 ${d} 天 · 清晨`, dawnP: (x, s) => `天亮了。今日运来 ${x} 份粮食，公共粮仓现有 ${s} 份。`,
    livingNow: "存活玩家", gtDawn: "清晨真实状态", hunger: "饥饿 h", stock: "私人存粮", atRisk: "今天分不到粮会饿死（h = 1 且没有存粮）",
    round: r => `第 ${r} 轮协商`, speaks: "发言", silent: "（未发言）", pass: p => `第 ${p} 圈`,
    record: "往日公开记录", recordP: "每天开始时发给所有玩家，只含已公开的事件。", statement: "提案说明",
    proposal: "分配提案", proposer: "提案人", noProp: p => `${p} 本轮放弃提案`,
    vote: "投票结果", voteOn: p => `对 ${p} 的提案投票`, passed: "通过", rejected: "被否决", accept: "同意", reject: "反对", secret: "（个人投票不公开）",
    nextRound: "提案未通过，进入下一轮。",
    release: "粮仓发放", releaseFb: "抽签发放", left: s => `粮仓剩余 ${s} 份`, nobody: "无人获得粮食",
    fallback: "协商破裂 · 抽签申领", fallbackP: "所有协商轮都没有通过提案。每位玩家申领 0 或 1 份，粮食不够时随机抽签。",
    requesters: "申领者",
    evening: "傍晚 · 私下行动", eveningP: "玩家可以私下赠送一份粮食、偷取一份粮食，或什么都不做。没有人会公开看到发生了什么。",
    gtTransfer: "私下行动真相", give: "赠送给", steal: "偷取", none: "无行动", ok: "成功", fail: "失败",
    notice: "私密通知", giftFrom: (to, g) => g ? `${to} 收到 ${g} 赠送的 1 份粮食` : `${to} 收到 1 份匿名赠粮`,
    theftOn: (v, t) => t ? `${v} 被偷走 1 份粮食，${v} 得知小偷是 ${t}` : `${v} 被偷走 1 份粮食，${v} 不知道是谁偷的`,
    tResult: (a, act, tg, ok) => `${a} 的${act === "give" ? "赠送" : "偷取"}（对象 ${tg}）${ok ? "成功" : "失败"}`,
    night: "夜晚 · 用餐", meals: "用餐情况", ate: "吃了饭", skipped: "没吃",
    dusk: d => `第 ${d} 天 · 结算`, deathP: ds => `${ds} 饿死了。`, peace: "平安夜，无人饿死。",
    gtEnd: "日终真实状态", store: "粮仓",
    over: "游戏结束", overP: s => s.length ? `幸存者：${s.join("、")}` : "无人幸存",
    tries: n => `重答 ${n - 1} 次`, deflt: "无效回复，使用默认动作", tok: "tokens", reasoning: "私下备注", thinking: "思考",
    vis: "仅上帝可见",
    legend: "R1/R2/R3 = 当天在第几轮协商通过分配提案 · FB = 所有轮都没通过，抽签发放 · 红点 = 当天有人饿死",
    tipR: r => `第 ${r} 轮协商通过提案`, tipFb: "协商全部失败，抽签发放", tipNone: "当天尚未发放", tipDeath: "，有人饿死",
  },
  en: {
    eyebrow: "Granary survival game · replay", god: "God view", pub: "Public view", why: "Show thinking and private notes", whyTip: "Thinking: for Claude and OpenAI seats, the provider's summary of the model's thinking (Claude thinking summaries, OpenAI reasoning summaries); it is not the raw chain of thought, another model writes it and it may leave things out. For DeepSeek seats, the raw chain of thought returned by the API (reasoning_content). Private note: the optional reasoning field in the reply JSON, which the rules say no other player sees. Both are shown next to the matching message, proposal, vote, evening action and meal.",
    seed: "seed", days: "days", rounds: "rounds/day", players: "players", supply: "daily supply",
    live: "Game in progress (reloads every 15 s)", survived: "Everyone survived", deathsN: n => `${n} died`, survival: "survival",
    gap: "priority gap (provisional)", cost: "notional cost",
    alive: "alive", diedOn: d => `died day ${d}`,
    gameStart: "Game start", gameStartP: (n, k) => `${n} players take their seats. ${k} of them start hungry, and only they know it.`,
    gtStart: "True starting state", supplySeq: "supply sequence",
    dawn: d => `Day ${d} · Dawn`, dawnP: (x, s) => `Morning. ${x} units arrived today; the public granary now holds ${s}.`,
    livingNow: "Living", gtDawn: "True state at dawn", hunger: "hunger h", stock: "private stock", atRisk: "dies tonight unless given food (h = 1, no stock)",
    round: r => `Negotiation round ${r}`, speaks: "says", silent: "(said nothing)", pass: p => `pass ${p}`,
    record: "Record of earlier days", recordP: "Sent to every player at the start of the day; announced events only.", statement: "Statement",
    proposal: "Allocation proposal", proposer: "proposer", noProp: p => `${p} makes no proposal this round`,
    vote: "Vote", voteOn: p => `Vote on ${p}'s proposal`, passed: "Passed", rejected: "Rejected", accept: "accept", reject: "reject", secret: "(individual votes not announced)",
    nextRound: "No agreement; next round.",
    release: "Granary release", releaseFb: "Lottery release", left: s => `${s} left in the granary`, nobody: "nobody received food",
    fallback: "Talks failed · fallback requests", fallbackP: "No proposal passed in any round. Each player requests 0 or 1 unit; a lottery decides if food runs short.",
    requesters: "requested",
    evening: "Evening · private actions", eveningP: "Players may secretly give one unit, steal one unit, or do nothing. Nobody publicly sees what happens.",
    gtTransfer: "What really happened", give: "gives to", steal: "steals from", none: "no action", ok: "succeeded", fail: "failed",
    notice: "Private notice", giftFrom: (to, g) => g ? `${to} receives 1 unit from ${g}` : `${to} receives 1 unit from an unknown giver`,
    theftOn: (v, t) => t ? `${v} lost 1 unit and was told the thief was ${t}` : `${v} lost 1 unit and was not told who took it`,
    tResult: (a, act, tg, ok) => `${a}'s ${act} (target ${tg}) ${ok ? "succeeded" : "failed"}`,
    night: "Night · meal", meals: "Who ate", ate: "ate", skipped: "did not eat",
    dusk: d => `Day ${d} · Settlement`, deathP: ds => `${ds} starved to death.`, peace: "A quiet night. Nobody starved.",
    gtEnd: "True state at day end", store: "granary",
    over: "Game over", overP: s => s.length ? `Survivors: ${s.join(", ")}` : "No survivors",
    tries: n => `${n - 1} retries`, deflt: "invalid reply, default action used", tok: "tokens", reasoning: "note", thinking: "thinking",
    vis: "god only",
    legend: "R1/R2/R3 = round in which the day's allocation passed · FB = no round passed, lottery release · red dot = someone starved",
    tipR: r => `Allocation passed in round ${r}`, tipFb: "All rounds failed; lottery release", tipNone: "No release yet", tipDeath: "; someone starved",
  },
};
let lang = "en", view = "god";
try { lang = localStorage.getItem("gr-lang-en") || "en"; view = localStorage.getItem("gr-view") || "god";
  document.getElementById("c-why").checked = localStorage.getItem("gr-why") === "1"; } catch (e) {}

const ids = D.cfg && D.events.length ? D.events[0].data.agent_ids : Object.keys(D.roster);
const color = a => `var(--s${ids.indexOf(a) % 8})`;
const el = (tag, cls, txt) => { const n = document.createElement(tag); if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; };
const av = (a, sm) => { const n = el("span", "av" + (sm ? " sm" : ""), a); n.style.background = color(a); n.setAttribute("aria-hidden", "true"); return n; };
const ICON = {
  sun: '<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="10" cy="10" r="3.6" fill="none"/><path d="M10 1.5v2.6M10 15.9v2.6M1.5 10h2.6M15.9 10h2.6M4 4l1.8 1.8M14.2 14.2L16 16M4 16l1.8-1.8M14.2 5.8L16 4"/></svg>',
  moon: '<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M15.5 12.5A6.5 6.5 0 0 1 7.5 4.5a6.5 6.5 0 1 0 8 8z" fill="none"/></svg>',
  dusk: '<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M2 14h16M5 14a5 5 0 0 1 10 0" fill="none"/><path d="M10 4v3M3.5 8l1.7 1.2M16.5 8l-1.7 1.2"/></svg>',
  grain: '<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M10 18V6"/><path d="M10 9c-2.5 0-4-1.5-4-4 2.5 0 4 1.5 4 4zM10 9c2.5 0 4-1.5 4-4-2.5 0-4 1.5-4 4zM10 13.5c-2.5 0-4-1.5-4-4 2.5 0 4 1.5 4 4zM10 13.5c2.5 0 4-1.5 4-4-2.5 0-4 1.5-4 4z" fill="none"/></svg>',
  skull: '<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M10 2.5a6 6 0 0 0-6 6c0 2.2 1.1 3.7 2.5 4.6V16h7v-2.9c1.4-.9 2.5-2.4 2.5-4.6a6 6 0 0 0-6-6z" fill="none"/><circle cx="7.6" cy="9" r="1.2" fill="currentColor"/><circle cx="12.4" cy="9" r="1.2" fill="currentColor"/></svg>',
  flag: '<svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M4.5 18V2.5M4.5 3h10l-2.5 3.5 2.5 3.5h-10" fill="none"/></svg>',
};
function band(icon, title, lines, cls) {
  const b = el("section", "band" + (cls ? " " + cls : ""));
  const h = el("h2"); h.innerHTML = ICON[icon] || ""; h.append(el("span", null, title)); b.append(h);
  for (const l of lines || []) { if (l instanceof Node) b.append(l); else if (l) b.append(el("p", null, l)); }
  return b;
}
function godBox(title, rows) {
  const g = el("div", "god godonly"); g.append(el("div", "tag", title + " · " + T[lang].vis));
  for (const r of rows) g.append(r instanceof Node ? r : el("div", null, r));
  return g;
}
function chip(a, val, cls) { const c = el("span", "chip" + (cls ? " " + cls : "")); c.append(av(a, true)); if (val != null) c.append(el("span", "n", String(val))); return c; }
function stateTable(st) {
  const t = T[lang], wrap = el("div", "statgrid"), tb = el("table"), live = ids.filter(a => st.alive[a]);
  const hr = el("tr"); hr.append(el("th", null, "")); live.forEach(a => hr.append(el("th", null, a))); tb.append(hr);
  const r1 = el("tr"); r1.append(el("th", null, t.hunger)); live.forEach(a => r1.append(el("td", st.at_risk.includes(a) ? "risk" : null, String(st.h[a])))); tb.append(r1);
  const r2 = el("tr"); r2.append(el("th", null, t.stock)); live.forEach(a => r2.append(el("td", null, String(st.stock[a])))); tb.append(r2);
  wrap.append(tb); return wrap;
}
function decision(day, phase, rnd, a) { return D.calls[`${day}|${phase}|${rnd == null ? "" : rnd}|${a}`]; }
function whyNode(c) {
  if (!c) return null;
  const t = T[lang], frag = document.createDocumentFragment();
  if (c.th) { const w = el("div", "why godonly"); w.append(el("span", "tag", t.thinking), document.createTextNode(c.th)); frag.append(w); }
  if (c.r) { const w = el("div", "why godonly"); w.append(el("span", "tag", t.reasoning), document.createTextNode(c.r)); frag.append(w); }
  if (c.n > 1 || c.def !== undefined) {
    const f = el("div", "flag godonly", [c.n > 1 ? t.tries(c.n) : "", c.def !== undefined ? t.deflt : ""].filter(Boolean).join(" · "));
    frag.append(f);
  }
  return frag;
}

// One tagged line per seat with a private note for this decision.
function notes(box, day, phase, rnd, seats) {
  const t = T[lang];
  for (const a of seats) {
    const dc = decision(day, phase, rnd, a);
    if (!dc) continue;
    if (dc.th) { const w = el("div", "why godonly"); w.append(el("span", "tag", `${a} · ${t.thinking}`), document.createTextNode(dc.th)); box.append(w); }
    if (dc.r) { const w = el("div", "why godonly"); w.append(el("span", "tag", `${a} · ${t.reasoning}`), document.createTextNode(dc.r)); box.append(w); }
  }
}
function render() {
  const t = T[lang], cfg = D.cfg, feed = document.getElementById("feed");
  document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
  document.getElementById("app").classList.toggle("is-public", view === "pub");
  document.getElementById("app").classList.toggle("no-why", !document.getElementById("c-why").checked);
  document.getElementById("v-god").textContent = t.god; document.getElementById("v-pub").textContent = t.pub;
  document.getElementById("v-god").setAttribute("aria-pressed", view === "god"); document.getElementById("v-pub").setAttribute("aria-pressed", view === "pub");
  document.getElementById("l-zh").setAttribute("aria-pressed", lang === "zh"); document.getElementById("l-en").setAttribute("aria-pressed", lang === "en");
  document.getElementById("c-why-l").textContent = t.why; document.querySelector(".godonly-ctl").title = t.whyTip;
  document.querySelector(".godonly-ctl").hidden = view === "pub";
  document.getElementById("eyebrow").textContent = t.eyebrow;
  document.getElementById("title").textContent = D.run;
  document.getElementById("cfgline").textContent =
    `${cfg.n_agents} ${t.players} · ${cfg.days} ${t.days} · ${cfg.rounds} ${t.rounds} · ${t.supply} ${cfg.supply_values.join("/")} · q=${cfg.q_reveal_thief} · ${t.seed} ${D.seed}`;

  // outcome + seats
  const died = {}; D.events.filter(e => e.kind === "deaths").forEach(e => e.data.died.forEach(a => died[a] = e.day));
  const oc = document.getElementById("outcome"); oc.replaceChildren();
  if (!D.finished) oc.append(el("span", "pill live", t.live));
  const nd = Object.keys(died).length;
  if (D.finished) oc.append(el("span", "pill " + (nd ? "bad" : "ok"), nd ? t.deathsN(nd) : t.survived));
  if (D.result) {
    if (D.result.survival != null) oc.append(el("span", "pill num", `${t.survival} ${(D.result.survival * 100).toFixed(1)}%`));
    if (D.result.priority_gap != null) oc.append(el("span", "pill num", `${t.gap} ${D.result.priority_gap.toFixed(3)}`));
    if (D.result.cost_usd) oc.append(el("span", "pill num", `${t.cost} $${D.result.cost_usd.toFixed(2)}`));
  }
  const seats = document.getElementById("seats"); seats.replaceChildren();
  ids.forEach(a => {
    const s = el("div", "seat" + (died[a] ? " dead" : "")), w = el("div", "who");
    w.append(el("div", null, a + " · " + (died[a] ? t.diedOn(died[a]) : t.alive)), el("div", "model", D.models[a] || ""));
    s.append(av(a), w); seats.append(s);
  });

  // day strip
  const dayInfo = {};
  D.events.forEach(e => {
    const di = dayInfo[e.day] = dayInfo[e.day] || {};
    if (e.kind === "release") di.how = e.data.source === "fallback" ? "fb" : "r" + e.round;
    if (e.kind === "deaths" && e.data.died.length) di.death = true;
  });
  const days = document.getElementById("days"); days.replaceChildren();
  Object.keys(dayInfo).map(Number).filter(d => d > 0).forEach(d => {
    const di = dayInfo[d], b = el("button", [di.how || "", di.death ? "death" : ""].join(" ").trim());
    b.type = "button"; b.dataset.day = d;
    b.append(document.createTextNode("D" + d), el("span", "r", di.how === "fb" ? "FB" : di.how ? di.how.toUpperCase() : "…"));
    b.title = (di.how === "fb" ? t.tipFb : di.how ? t.tipR(di.how.slice(1)) : t.tipNone) + (di.death ? t.tipDeath : "");
    b.addEventListener("click", () => {
      const tgt = document.getElementById("day-" + d); if (!tgt) return;
      const barH = document.querySelector(".bar").getBoundingClientRect().height;   // the sticky bar is taller on phones
      window.scrollTo(0, tgt.getBoundingClientRect().top + window.scrollY - barH - 12);
    });
    days.append(b);
  });
  document.getElementById("legend").textContent = t.legend;
  curDay = null; spy();

  // feed
  feed.replaceChildren();
  let roundKey = null, phaseSeen = {};
  const once = (k, f) => { if (!phaseSeen[k]) { phaseSeen[k] = 1; f(); } };
  for (const e of D.events) {
    const d = e.data, day = e.day, priv = e.visible_to !== "all";
    if (["MESSAGE", "PROPOSE", "VOTE"].includes(e.phase) && e.round != null) {
      const k = day + "-" + e.round;
      if (k !== roundKey) { roundKey = k; feed.append(el("div", "sub-band", t.round(e.round))); }
    }
    switch (e.kind) {
      case "game_start":
        feed.append(band("flag", t.gameStart, [t.gameStartP(d.agent_ids.length, d.initially_hungry)])); break;
      case "truth_game_start":
        feed.append(godBox(t.gtStart, [
          el("div", null, `${t.hunger}: ` + ids.map(a => `${a}=${d.h[a]}`).join("  ")),
          el("div", "mono", `${t.supplySeq}: ${d.supply.join(" ")}`)])); break;
      case "initial_status": break;
      case "public_record": {
        const c = el("div", "card"); c.append(el("div", "ttl", t.record), el("div", "sub", t.recordP));
        d.days.forEach(x => {
          const parts = [`D${x.day} · supply ${x.supply}`];
          x.rounds.forEach(r => parts.push(r.allocation == null ? `R${r.round} ${r.proposer}: —`
            : `R${r.round} ${r.proposer} ${Object.entries(r.allocation).map(([a, v]) => a + v).join("/")} ${r.passed ? "✓" : "✗"}`
              + ((rj => rj.length ? ` (${rj.join(" ")} reject)` : "")(Object.entries(r.votes || {}).filter(([, v]) => v !== "accept").map(([a]) => a)))));
          if (x.release) parts.push((x.release.source === "fallback" ? "FB " : "→ ") + Object.entries(x.release.received).filter(([, v]) => v > 0).map(([a, v]) => a + v).join(" "));
          parts.push(x.died && x.died.length ? "† " + x.died.join(",") : "† —");
          c.append(el("div", "mono", parts.join(" · ")));
        });
        feed.append(c); break;
      }
      case "day_start": {
        roundKey = null;
        const b = band("sun", t.dawn(day), [t.dawnP(d.supply, d.store)]);
        const live = el("div", "chips"); d.living.forEach(a => live.append(chip(a))); b.append(live);
        b.id = "day-" + day; b.style.scrollMarginTop = "150px"; feed.append(b); break;
      }
      case "truth_day_start":
        feed.append(godBox(t.gtDawn, [stateTable(d), d.at_risk.length ? el("div", "risk", `${t.atRisk}: ${d.at_risk.join(", ")}`) : null].filter(Boolean))); break;
      case "message": {
        const row = el("div", "say"), body = el("div", "body"), meta = el("div", "meta");
        meta.append(el("b", null, d.from), el("span", null, D.models[d.from] || ""));
        if (d.pass) meta.append(el("span", null, "· " + t.pass(d.pass)));
        body.append(meta, el("div", "bubble" + (d.text ? "" : " empty"), d.text || t.silent));
        const w = whyNode(decision(day, "MESSAGE", d.pass ? `${e.round}.${d.pass}` : e.round, d.from)); if (w) body.append(w);
        row.append(av(d.from), body); feed.append(row); break;
      }
      case "proposal": {
        const c = el("div", "card prop"), ttl = el("div", "ttl");
        ttl.append(av(d.proposer, true), el("span", null, `${t.proposal} · ${t.proposer} ${d.proposer}`));
        const ch = el("div", "chips"); Object.entries(d.allocation).forEach(([a, v]) => ch.append(chip(a, v, v ? "" : "zero")));
        c.append(ttl, ch);
        if (d.statement) { const s = el("div", "bubble"); s.append(el("b", null, t.statement + ": "), d.statement); c.append(s); }
        const w = whyNode(decision(day, "PROPOSE", e.round, d.proposer)); if (w) c.append(w);
        feed.append(c); break;
      }
      case "no_proposal": {
        const c = el("div", "card"); c.append(el("div", "ttl", t.noProp(d.proposer)));
        const w = whyNode(decision(day, "PROPOSE", e.round, d.proposer)); if (w) c.append(w); feed.append(c); break;
      }
      case "vote_result": {
        const c = el("div", "card"), ttl = el("div", "ttl");
        const pe = [...D.events].reverse().find(x => x.kind === "proposal" && x.day === day && x.round === e.round);
        if (pe) ttl.append(av(pe.data.proposer, true));
        ttl.append(el("span", null, pe ? t.voteOn(pe.data.proposer) : t.vote), el("span", "pill verdict " + (d.passed ? "ok" : "bad"), d.passed ? t.passed : t.rejected));
        c.append(ttl);
        if (d.votes) {
          const ch = el("div", "chips");
          Object.entries(d.votes).forEach(([a, v]) => { const x = chip(a, null, v === "accept" ? "yes" : "no"); x.append(el("span", null, v === "accept" ? t.accept : t.reject)); ch.append(x); });
          c.append(ch);
          notes(c, day, "VOTE", e.round, Object.keys(d.votes));
        } else c.append(el("div", "sub", t.secret));
        if (!d.passed) c.append(el("div", "sub", t.nextRound));
        feed.append(c); break;
      }
      case "truth_fallback_requests":
        feed.append(band("dusk", t.fallback, [t.fallbackP]));
        feed.append(godBox(t.requesters, [d.requesters.join(", ") || "—"])); break;
      case "release": {
        if (d.source === "fallback" && !D.events.some(x => x.day === day && x.kind === "truth_fallback_requests"))
          feed.append(band("dusk", t.fallback, [t.fallbackP]));
        const got = Object.entries(d.received).filter(([, v]) => v > 0);
        const c = el("div", "card"), ttl = el("div", "ttl"); ttl.innerHTML = ICON.grain; ttl.append(el("span", null, d.source === "fallback" ? t.releaseFb : t.release));
        const ch = el("div", "chips"); got.forEach(([a, v]) => ch.append(chip(a, v)));
        c.append(ttl, got.length ? ch : el("div", "sub", t.nobody), el("div", "sub", t.left(d.store)));
        feed.append(c); break;
      }
      case "transfer_result": case "gift_received": case "theft_loss":
        once("ev" + day, () => feed.append(band("dusk", t.evening, [t.eveningP])));
        {
          const who = e.visible_to[0];
          const txt = e.kind === "gift_received" ? t.giftFrom(who, d.from) : e.kind === "theft_loss" ? t.theftOn(who, d.thief) : t.tResult(who, d.action, d.target, d.success);
          const g = el("div", "god godonly"); g.append(el("div", "tag", `${t.notice} → ${who} · ${t.vis}`), el("div", null, txt)); feed.append(g);
        } break;
      case "truth_transfers": {
        once("ev" + day, () => feed.append(band("dusk", t.evening, [t.eveningP])));
        const rows = Object.entries(d.actions).map(([a, x]) => {
          const ok = d.successful.some(s => s[0] === a);
          const r = el("div", "chips"); r.append(chip(a));
          if (x.action === "none") r.append(el("span", "sub", t.none));
          else { r.append(el("span", null, x.action === "give" ? t.give : t.steal), chip(x.target), el("span", "pill " + (ok ? "ok" : "bad"), ok ? t.ok : t.fail)); }
          return r;
        });
        const acted = Object.values(d.actions).some(x => x.action !== "none");
        const gb = godBox(t.gtTransfer, acted ? rows.filter((_, i) => Object.values(d.actions)[i].action !== "none") : [t.none]);
        notes(gb, day, "TRANSFER", null, Object.keys(d.actions)); feed.append(gb);
        break;
      }
      case "truth_meals": {
        const g = godBox(t.meals, []);
        const ch = el("div", "chips");
        const lastDawn = [...D.events].reverse().find(x => x.kind === "truth_day_start" && x.day === day);
        const living = lastDawn ? ids.filter(a => lastDawn.data.alive[a]) : ids;
        living.forEach(a => { const x = chip(a, null, d.ate.includes(a) ? "yes" : "zero"); x.append(el("span", null, d.ate.includes(a) ? t.ate : t.skipped)); ch.append(x); });
        g.append(ch); notes(g, day, "MEAL", null, living); const nb = band("moon", t.night, []); nb.classList.add("godonly"); feed.append(nb, g); break;
      }
      case "deaths": {
        const lines = [d.died.length ? t.deathP(d.died.join(lang === "zh" ? "、" : ", ")) : t.peace];
        feed.append(band(d.died.length ? "skull" : "moon", t.dusk(day), lines)); break;
      }
      case "truth_day_end":
        feed.append(godBox(t.gtEnd, [stateTable(d), el("div", "mono", `${t.store}: ${d.store}`)])); break;
      case "game_end":
        feed.append(band("flag", t.over, [t.overP(d.survivors)], "end")); break;
      default:
        if (priv) { const g = godBox(e.kind, [JSON.stringify(d)]); feed.append(g); }
        else feed.append(el("div", "sub", `${e.kind}: ${JSON.stringify(d)}`));
    }
  }
}
// Highlight the day whose dawn banner was passed last, and keep its button in view in the strip.
let curDay = null;
function spy() {
  const barBottom = document.querySelector(".bar").getBoundingClientRect().bottom;
  let d = null;
  for (const b of document.querySelectorAll(".feed .band[id^='day-']")) {
    if (b.getBoundingClientRect().top <= barBottom + 48) d = b.id.slice(4); else break;
  }
  if (d === curDay) return;
  curDay = d;
  const strip = document.getElementById("days");
  for (const btn of strip.children) {
    const on = btn.dataset.day === d;
    btn.classList.toggle("cur", on);
    if (on) { btn.setAttribute("aria-current", "true");
      const l = btn.offsetLeft - strip.offsetLeft, r = l + btn.offsetWidth;
      if (l < strip.scrollLeft || r > strip.scrollLeft + strip.clientWidth) strip.scrollLeft = l - (strip.clientWidth - btn.offsetWidth) / 2;
    } else btn.removeAttribute("aria-current");
  }
}
window.addEventListener("scroll", spy, { passive: true });   // one rect read per day banner; cheap enough unthrottled
window.addEventListener("resize", () => { curDay = null; spy(); });
function set(k, v) { if (k === "lang") lang = v; else view = v; try { localStorage.setItem(k === "lang" ? "gr-lang-en" : "gr-view", v); } catch (e) {} render(); }
document.getElementById("v-god").onclick = () => set("view", "god");
document.getElementById("v-pub").onclick = () => set("view", "pub");
document.getElementById("l-zh").onclick = () => set("lang", "zh");
document.getElementById("l-en").onclick = () => set("lang", "en");
document.getElementById("c-why").onchange = () => { try { localStorage.setItem("gr-why", document.getElementById("c-why").checked ? "1" : "0"); } catch (e) {} render(); };
render();
if (!D.finished) window.scrollTo(0, document.body.scrollHeight);
})();
</script>
"""


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--fragment"]:      # page body only, e.g. for publishing as an artifact
        print(build(args[1], args[2], standalone=False))
    else:
        for arg in args:
            print(build(arg))
