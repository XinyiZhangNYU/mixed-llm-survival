"""Game loop: ask pending agents, validate with retries, apply, log.

Within a phase every pending agent is asked from the same state, in parallel
(asyncio.gather); the engine settles only after all answers are in.

Log directory layout (when log_dir is given):
  meta.json        config, seed, roster labels
  events.jsonl     every engine event, with its visibility list
  calls.jsonl      every decision attempt: observation, raw reply, parsed action, error
  actions.jsonl    the final action batch of each step (input for replay)
  checkpoint.json  game state after the latest step
  result.json      outcome summary
  replay.html      moderator-style replay page (sim/replay_html.py), rebuilt at key events

run_game can stop early at a cost cap (provider-reported cost_usd). It stops only between
steps, so checkpoint.json and actions.jsonl stay consistent, and result.json records the stop.
Model conversations are not resumable from the checkpoint yet.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from . import engine
from .agents.base import Agent, Decision
from .config import Config
from .metrics import summarize
from .observe import observe
from .replay_html import build as build_replay
from .state import GameState


REPLAY_TRIGGERS = {"day_start", "release", "deaths", "game_end"}


class RunLogger:
    def __init__(self, log_dir: str | Path | None):
        self.dir = Path(log_dir) if log_dir else None
        self.cost_usd = 0.0          # sum of provider-reported cost over all attempts
        self.max_call_cost = 0.0
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)
            for name in ("events.jsonl", "calls.jsonl", "actions.jsonl"):
                (self.dir / name).write_text("", encoding="utf-8")

    def _append(self, name, rec):
        if self.dir:
            with open(self.dir / name, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _write(self, name, obj):
        if self.dir:
            tmp = self.dir / (name + ".tmp")
            tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, self.dir / name)

    def meta(self, cfg: Config, seed: int, roster: dict[str, Agent]):
        self._write("meta.json", {"cfg": cfg.to_dict(), "seed": seed,
                                  "roster": {a: ag.label for a, ag in roster.items()}})

    def call(self, rec):
        self._append("calls.jsonl", rec)

    def events(self, events):
        for ev in events:
            self._append("events.jsonl", asdict(ev))

    def step(self, s: GameState, phase: str, day: int, rnd, actions: dict, events):
        if not self.dir:
            return
        self._append("actions.jsonl", {"step": s.step, "day": day, "phase": phase,
                                       "round": rnd, "actions": actions})
        self.events(events)
        self._write("checkpoint.json", s.to_dict())
        if any(ev.kind in REPLAY_TRIGGERS for ev in events):
            self.replay()

    def replay(self):
        # The page is a view of the logs; a failure here must never stop a run.
        try:
            build_replay(self.dir)
        except Exception:
            pass

    def add_cost(self, d: Decision):
        c = (d.meta or {}).get("cost_usd") or 0.0
        self.cost_usd += c
        self.max_call_cost = max(self.max_call_cost, c)

    def result(self, res):
        self._write("result.json", res)
        if self.dir:
            self.replay()


def _check(s: GameState, agent: str, d: Decision) -> str | None:
    return d.error or engine.validate(s, agent, d.action)


def _call_record(s, req, obs, attempt, d: Decision, err):
    return {"step": s.step + 1, "day": s.day, "phase": s.phase, "round": obs.round,
            "agent": req.agent, "attempt": attempt, "observation": obs.to_dict(),
            "raw": d.raw, "reasoning": d.reasoning, "action": d.action, "error": err,
            "meta": d.meta}


async def _resolve(s, req, agent: Agent, since: int, log: RunLogger):
    obs = observe(s, req.agent, since)
    feedback = None
    for attempt in range(s.cfg.max_retries + 1):
        d = await agent.decide(obs, req.key, feedback)
        log.add_cost(d)
        err = _check(s, req.agent, d)
        if log.dir:
            log.call(_call_record(s, req, obs, attempt, d, err))
        if err is None:
            return d.action, obs.last_seq
        feedback = err
    default = s.cfg.invalid_defaults[req.key]
    log.call({"step": s.step + 1, "agent": req.agent, "default_used": default})
    return default, obs.last_seq


def _resolve_sync(s, req, agent: Agent, since: int, log: RunLogger):
    obs = observe(s, req.agent, since)
    feedback = None
    for attempt in range(s.cfg.max_retries + 1):
        d = agent.decide_sync(obs, req.key, feedback)
        log.add_cost(d)
        err = _check(s, req.agent, d)
        if log.dir:
            log.call(_call_record(s, req, obs, attempt, d, err))
        if err is None:
            return d.action, obs.last_seq
        feedback = err
    default = s.cfg.invalid_defaults[req.key]
    log.call({"step": s.step + 1, "agent": req.agent, "default_used": default})
    return default, obs.last_seq


def _setup(cfg, roster, seed, log):
    if sorted(roster) != cfg.agent_ids:
        raise ValueError(f"roster must cover seats {cfg.agent_ids}")
    s = engine.new_game(cfg, seed)
    for a, ag in roster.items():
        ag.start_game(a, cfg, seed)
    log.meta(cfg, seed, roster)
    log.events(s.events)
    return s, {a: -1 for a in cfg.agent_ids}


def _commit(s, reqs, results, last_seen, log):
    actions = {}
    for req, (action, seen) in zip(reqs, results):
        actions[req.agent] = action
        last_seen[req.agent] = seen
    phase, day, rnd = s.phase, s.day, s.round
    events = engine.apply(s, actions)
    log.step(s, phase, day, rnd, actions, events)


async def run_game(cfg: Config, roster: dict[str, Agent], seed: int,
                   log_dir: str | Path | None = None, cost_cap_usd: float | None = None) -> dict:
    """With cost_cap_usd, stop before any step that could push spending past the cap,
    estimating each pending request at the most expensive single call so far. The estimate
    can be low (later calls carry longer histories, retries cost extra), so a small overshoot
    is possible."""
    log = RunLogger(log_dir)
    s, last_seen = _setup(cfg, roster, seed, log)
    stopped = None
    while not s.done:
        reqs = engine.pending(s)
        if cost_cap_usd is not None and log.cost_usd + len(reqs) * log.max_call_cost > cost_cap_usd:
            stopped = "cost_cap"
            break
        results = await asyncio.gather(*[
            _resolve(s, r, roster[r.agent], last_seen[r.agent], log) for r in reqs])
        _commit(s, reqs, results, last_seen, log)
    res = summarize(s) | {"cost_usd": round(log.cost_usd, 4)}
    if stopped:
        # survival over N*T is not meaningful for an unfinished game
        res |= {"stopped": stopped, "cost_cap_usd": cost_cap_usd, "survival": None,
                "priority_gap": None, "stopped_at": {"day": s.day, "phase": s.phase, "round": s.round},
                "days_completed": len(s.alive_end_of_day), "living": s.living()}
    log.result(res)
    return res | {"state": s}


def run_game_sync(cfg: Config, roster: dict[str, Agent], seed: int,
                  log_dir: str | Path | None = None) -> dict:
    """Same loop without asyncio, for scripted-only rosters (fast reference runs)."""
    log = RunLogger(log_dir)
    s, last_seen = _setup(cfg, roster, seed, log)
    while not s.done:
        reqs = engine.pending(s)
        results = [_resolve_sync(s, r, roster[r.agent], last_seen[r.agent], log) for r in reqs]
        _commit(s, reqs, results, last_seen, log)
    res = summarize(s)
    log.result(res)
    return res | {"state": s}


def replay(log_dir: str | Path) -> GameState:
    """Rebuild a game from meta.json and actions.jsonl using only the engine."""
    d = Path(log_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    s = engine.new_game(Config.from_dict(meta["cfg"]), meta["seed"])
    for line in (d / "actions.jsonl").read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if (rec["phase"], rec["day"]) != (s.phase, s.day):
            raise ValueError(f"replay diverged at step {rec['step']}")
        engine.apply(s, rec["actions"])
    return s


def load_checkpoint(log_dir: str | Path) -> GameState:
    return GameState.from_dict(json.loads((Path(log_dir) / "checkpoint.json").read_text(encoding="utf-8")))
