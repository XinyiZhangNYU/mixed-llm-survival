"""Provider that calls the Claude Code CLI in print mode.

Two modes, both with no tools, no MCP servers, no slash commands, and an empty working
directory, and with the game rules as the system prompt:

- "persistent" (default): one CLI session per provider, so use one provider per seat. The
  first call starts the session with --session-id; later calls --resume it and send only the
  newest user turn. The prompt prefix stays fixed, so the prompt cache can be reused.
- "stateless": each call is a fresh process with --no-session-persistence, and the agent's
  conversation so far is flattened into one prompt. The prefix changes on every call.

The persistent mode checks that the agent's message list matches what the session holds.
The session is restarted from a flattened transcript (logged in meta as "session_restart")
when they differ or after a failed call, since a failed call may leave a half-recorded turn.

Invalid replies: LLMAgent keeps the invalid reply and the "not accepted" feedback in the
conversation, the same way for every agent and every provider. The session holds them too.

The CLI may add its own baseline instructions, so this path is for plumbing checks and
pilots, not for main runs.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import uuid
from pathlib import Path

MODES = ("persistent", "stateless")


def flatten(messages: list[dict]) -> str:
    """Render the non-system turns as one prompt; the last user turn is the current request."""
    turns = messages[1:]
    parts = []
    if len(turns) > 1:
        parts.append("Earlier turns of this game (your past requests and your replies):")
        for m in turns[:-1]:
            tag = "REQUEST" if m["role"] == "user" else "YOUR REPLY"
            parts.append(f"--- {tag} ---\n{m['content']}")
        parts.append("--- CURRENT REQUEST ---")
    parts.append(turns[-1]["content"])
    return "\n\n".join(parts)


def usage_meta(data: dict) -> dict:
    usage = data.get("usage", {})
    fresh = usage.get("input_tokens", 0)
    read = usage.get("cache_read_input_tokens", 0)
    created = usage.get("cache_creation_input_tokens", 0)
    return {"cost_usd": data.get("total_cost_usd"),
            "input_tokens": fresh,                   # uncached input only
            "cache_read_input_tokens": read,
            "cache_creation_input_tokens": created,
            "total_input_tokens": fresh + read + created,
            "output_tokens": usage.get("output_tokens"),
            "models": list(data.get("modelUsage", {})),
            "is_error": data.get("is_error")}


class ClaudeCLIProvider:
    def __init__(self, model: str | None = None, timeout_s: float = 240.0,
                 executable: str = "claude", workdir: str | Path | None = None,
                 mode: str = "persistent"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.exe = shutil.which(executable) or executable
        self.model = model
        self.mode = mode
        self.model_id = f"claude-cli:{model or 'default'}"
        self.timeout_s = timeout_s
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="gt6263_cli_"))
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.session_id: str | None = None
        self.session_turns: list[dict] = []   # the agent's non-system turns the session covers
        self.session_system: str | None = None

    def _base_args(self, sys_path: str) -> list[str]:
        args = [self.exe, "-p", "--output-format", "json", "--system-prompt-file", sys_path,
                "--tools", "", "--strict-mcp-config", "--disable-slash-commands"]
        if self.model:
            args += ["--model", self.model]
        return args

    async def _run(self, args: list[str], prompt: str) -> tuple[int | None, str, str]:
        """Run the CLI once; returns (exit code, stdout, stderr). Tests replace this."""
        proc = await asyncio.create_subprocess_exec(
            *args, cwd=str(self.workdir), stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(prompt.encode("utf-8")), self.timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            raise
        return proc.returncode, out.decode("utf-8", errors="replace"), err.decode("utf-8", errors="replace")

    async def complete(self, messages: list[dict]) -> tuple[str, dict]:
        if self.mode == "stateless":
            return await self._call(["--no-session-persistence"], flatten(messages), messages[0]["content"])
        return await self._complete_persistent(messages)

    async def _complete_persistent(self, messages: list[dict]) -> tuple[str, dict]:
        system, history, new = messages[0]["content"], messages[1:-1], messages[-1]
        restart = None
        if self.session_id is not None and (history != self.session_turns or system != self.session_system):
            restart = "history_mismatch"
        elif self.session_id is None and history:
            restart = "after_failure" if self.session_system is not None else "late_start"
        if self.session_id is None or restart:
            self.session_id = str(uuid.uuid4())
            self.session_system = system
            extra, prompt = ["--session-id", self.session_id], flatten(messages)
        else:
            extra, prompt = ["--resume", self.session_id], new["content"]
        text, meta = await self._call(extra, prompt, system)
        meta |= {"cli_mode": "persistent", "session_id": self.session_id}
        if restart:
            meta["session_restart"] = restart
        if "error" in meta:
            self.session_id = None                # the session may hold a half-recorded turn
            self.session_turns = []
        else:
            self.session_turns = history + [new, {"role": "assistant", "content": text}]
        return text, meta

    async def _call(self, extra: list[str], prompt: str, system: str) -> tuple[str, dict]:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8",
                                         dir=self.workdir) as f:
            f.write(system)
            sys_path = f.name
        try:
            code, out, err = await self._run(self._base_args(sys_path) + extra, prompt)
        except asyncio.TimeoutError:
            return "", {"error": f"timeout after {self.timeout_s}s"}
        finally:
            Path(sys_path).unlink(missing_ok=True)
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            return "", {"error": f"exit {code}: {(out or err)[:500]}"}
        meta = usage_meta(data)
        if data.get("is_error"):
            meta["error"] = str(data.get("result"))[:500]
            return "", meta
        return data.get("result", ""), meta
