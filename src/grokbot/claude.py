"""Runs Claude Code in the background on your Claude plan and turns its output into events.

grokbot drives the regular `claude` CLI (headless, streaming JSON), signed in with your
claude.ai account. Anything that would switch Claude Code to API billing is removed from
its environment, and grokbot refuses to start if the CLI isn't on a Claude plan.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from typing import Any

# Environment variables that make Claude Code use an API account or cloud provider instead of your plan.
API_BILLING_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS",
)


class PlanError(RuntimeError):
    """Claude Code isn't available on a Claude plan."""


def find_claude() -> str:
    exe = shutil.which("claude")
    if not exe:
        raise PlanError("Can't find the `claude` command. Install Claude Code and sign in first.")
    return exe


def plan_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in API_BILLING_VARS}


def check_plan(exe: str) -> dict[str, Any]:
    """Confirms the CLI is signed in with a claude.ai account and returns its auth status."""
    done = subprocess.run(
        [exe, "auth", "status"], capture_output=True, text=True, encoding="utf-8",
        errors="replace", env=plan_env(), timeout=60,
    )
    try:
        status = json.loads(done.stdout)
    except json.JSONDecodeError:
        output = (done.stdout or done.stderr)[:300]
        raise PlanError(f"`claude auth status` didn't answer as expected:\n{output}") from None
    if not status.get("loggedIn"):
        raise PlanError("Claude Code isn't signed in. Run `claude`, type /login, and pick your Claude account.")
    if status.get("authMethod") != "claude.ai":
        raise PlanError(
            f"Claude Code is signed in with {status.get('authMethod')!r}, not a Claude plan, so it would bill "
            "an API account. Run `claude`, type /login, and choose your Claude subscription."
        )
    return status


# ---- events -------------------------------------------------------------------------------


@dataclass(frozen=True)
class Ready:
    """Claude Code started a turn (it reports this at the start of every turn)."""

    session_id: str
    model: str
    api_key_source: str | None
    permission_mode: str | None = None


@dataclass(frozen=True)
class Thinking:
    """Claude is reasoning. `tokens` is Claude Code's running estimate for the current thought;
    `text` is thought text, when Claude Code shares it (current models usually keep it hidden)."""

    tokens: int | None = None
    text: str = ""


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class TextDone:
    """A complete block of reply text, for showing on screen."""

    text: str


@dataclass(frozen=True)
class ToolCall:
    name: str
    detail: str
    id: str = ""
    nested: bool = False  # run by a helper agent rather than the main conversation
    input: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True)
class ToolResult:
    id: str
    ok: bool
    text: str  # the start of the output; worth showing when it failed


@dataclass(frozen=True)
class PermissionRequest:
    request_id: str
    tool: str
    detail: str
    tool_input: dict[str, Any] = field(hash=False)
    suggestions: list[Any] = field(hash=False)


@dataclass(frozen=True)
class TurnDone:
    ok: bool
    subtype: str
    session_id: str | None
    denials: int


@dataclass(frozen=True)
class Usage:
    """Plan usage for the current rate-limit window."""

    status: str
    window: str | None
    utilization: float | None


@dataclass(frozen=True)
class ModeChanged:
    """The permission mode changed: because we asked, or because Claude Code changed it itself."""

    mode: str


@dataclass(frozen=True)
class Status:
    """Claude Code's own status: "compacting" while it shrinks a long conversation, else ""."""

    status: str


@dataclass(frozen=True)
class ControlError:
    """Claude Code refused one of our control requests, such as a mode switch."""

    message: str
    request: str  # which kind of request: "set_permission_mode", "interrupt", …


@dataclass(frozen=True)
class Exited:
    code: int | None
    stderr: str


Event = (
    Ready | Thinking | TextDelta | TextDone | ToolCall | ToolResult | PermissionRequest | TurnDone | Usage
    | ModeChanged | Status | ControlError | Exited
)


class StreamParser:
    """Turns Claude Code's stream-json messages into events."""

    def __init__(self) -> None:
        self._blocks: dict[int, list[str]] = {}

    def parse(self, msg: dict[str, Any]) -> list[Event]:
        kind = msg.get("type")
        if kind == "stream_event":
            if msg.get("parent_tool_use_id"):
                return []  # a helper agent's own words aren't part of the reply
            return self._stream(msg.get("event") or {})
        if kind == "assistant":
            content = (msg.get("message") or {}).get("content") or []
            nested = bool(msg.get("parent_tool_use_id"))
            return [
                ToolCall(b.get("name", "?"), describe_tool(b.get("name", ""), b.get("input") or {}),
                         b.get("id", ""), nested, b.get("input") or {})
                for b in content if b.get("type") == "tool_use"
            ]
        if kind == "user":
            content = (msg.get("message") or {}).get("content")
            return [
                ToolResult(b.get("tool_use_id", ""), not b.get("is_error"), _result_text(b.get("content")))
                for b in (content if isinstance(content, list) else [])
                if isinstance(b, dict) and b.get("type") == "tool_result"
            ]
        if kind == "system":
            return self._system(msg)
        if kind == "control_response":
            response = msg.get("response") or {}
            if response.get("subtype") == "error":
                request = str(response.get("request_id") or "").removeprefix("grokbot-").rpartition("-")[0]
                return [ControlError(str(response.get("error") or "unknown error"), request)]
            mode = (response.get("response") or {}).get("mode")
            return [ModeChanged(mode)] if mode else []
        if kind == "control_request":
            req = msg.get("request") or {}
            if req.get("subtype") == "can_use_tool":
                tool, data = req.get("tool_name", "?"), req.get("input") or {}
                return [PermissionRequest(
                    msg.get("request_id", ""), tool, describe_tool(tool, data, full=True),
                    data, req.get("permission_suggestions") or [],
                )]
        if kind == "result":
            return [TurnDone(
                ok=not msg.get("is_error"), subtype=msg.get("subtype", ""),
                session_id=msg.get("session_id"), denials=len(msg.get("permission_denials") or []),
            )]
        if kind == "rate_limit_event":
            info = msg.get("rate_limit_info") or {}
            window = info.get("rateLimitType")
            used = ((info.get("unifiedWindows") or {}).get(window) or {}).get("utilization")
            return [Usage(info.get("status", ""), window, used)]
        return []

    def _system(self, msg: dict[str, Any]) -> list[Event]:
        subtype = msg.get("subtype")
        if subtype == "init":
            return [Ready(msg.get("session_id", ""), msg.get("model", ""), msg.get("apiKeySource"),
                          msg.get("permissionMode"))]
        if subtype == "thinking_tokens":
            return [Thinking(tokens=msg.get("estimated_tokens"))]
        if subtype == "status":
            events: list[Event] = [Status(msg.get("status") or "")]
            if msg.get("permissionMode"):
                events.append(ModeChanged(msg["permissionMode"]))
            return events
        return []

    def _stream(self, event: dict[str, Any]) -> list[Event]:
        kind, index = event.get("type"), event.get("index")
        if kind == "message_start":
            self._blocks.clear()
        elif kind == "content_block_start":
            block_type = (event.get("content_block") or {}).get("type")
            if block_type == "text":
                self._blocks[index] = []
            elif block_type == "thinking":
                return [Thinking()]
        elif kind == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "thinking_delta" and delta.get("thinking"):
                return [Thinking(text=delta["thinking"])]
            if delta.get("type") == "text_delta" and delta.get("text") and index in self._blocks:
                self._blocks[index].append(delta["text"])
                return [TextDelta(delta["text"])]
        elif kind == "content_block_stop" and index in self._blocks:
            return [TextDone("".join(self._blocks.pop(index)))]
        return []


def describe_tool(name: str, data: dict[str, Any], full: bool = False) -> str:
    """A short label for a tool call, e.g. "main.py" or the command it runs."""
    limit = 400 if full else 70
    if name in ("Bash", "PowerShell"):
        command = str(data.get("command") or "")
        text = command if full else (data.get("description") or command)
        return _clip(str(text), limit)
    for key in ("file_path", "notebook_path", "path"):
        if data.get(key):
            return str(data[key]) if full else PurePath(str(data[key])).name
    for key in ("pattern", "url", "query", "description", "prompt"):
        if data.get(key):
            return _clip(str(data[key]), limit)
    return ""


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _result_text(content: Any, limit: int = 300) -> str:
    """A tool result's content is a string or a list of blocks; keep the start of its text."""
    if isinstance(content, list):
        content = " ".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
    return str(content or "").strip()[:limit]


# ---- the session ----------------------------------------------------------------------------


class ClaudeSession:
    """One long-running `claude -p` process; you send it messages and it streams events back."""

    def __init__(
        self,
        *,
        cwd: Path,
        persona: Path,
        on_event: Callable[[Event], None],
        model: str | None = None,
        effort: str | None = None,
        permission_mode: str | None = None,
        resume: str | None = None,
        continue_session: bool = False,
    ):
        self._on_event = on_event
        self._cwd = cwd
        self._args = [
            "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
            "--include-partial-messages",
            "--permission-prompt-tool", "stdio",  # send permission prompts to us, so you can answer out loud
            "--append-system-prompt-file", str(persona),
            "--disallowedTools", "AskUserQuestion",  # questions should be spoken, not menus
            "-n", "grokbot",
        ]
        for flag, value in (("--model", model), ("--effort", effort), ("--permission-mode", permission_mode),
                            ("--resume", resume)):
            if value:
                self._args += [flag, value]
        if continue_session:
            self._args.append("--continue")
        self._proc: subprocess.Popen[str] | None = None
        self._write_lock = threading.Lock()
        self._stderr: deque[str] = deque(maxlen=40)

    def start(self) -> None:
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0  # keep Ctrl+C ours
        self._proc = subprocess.Popen(
            [find_claude(), *self._args], cwd=self._cwd, env=plan_env(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1, creationflags=flags,
        )
        threading.Thread(target=self._read, name="claude-out", daemon=True).start()
        threading.Thread(target=self._read_stderr, name="claude-err", daemon=True).start()

    def send(self, text: str) -> None:
        self._write({"type": "user", "message": {"role": "user", "content": text}})

    def interrupt(self) -> None:
        self._control({"subtype": "interrupt"})

    def set_mode(self, mode: str) -> None:
        """Switches the permission mode mid-conversation ("default", "acceptEdits", "auto", "plan")."""
        self._control({"subtype": "set_permission_mode", "mode": mode})

    def answer(
        self,
        request_id: str,
        allow: bool,
        *,
        tool_input: dict[str, Any] | None = None,
        message: str = "",
        remember: list[Any] | None = None,
    ) -> None:
        """Replies to a PermissionRequest. `remember` applies Claude Code's own "don't ask again" suggestions."""
        if allow:
            response: dict[str, Any] = {"behavior": "allow", "updatedInput": tool_input or {}}
            if remember:
                response["updatedPermissions"] = remember
        else:
            response = {"behavior": "deny", "message": message or "The user said no."}
        self._write({"type": "control_response",
                     "response": {"subtype": "success", "request_id": request_id, "response": response}})

    def close(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.close()  # type: ignore[union-attr]
            proc.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()

    def _control(self, request: dict[str, Any]) -> None:
        request_id = f"grokbot-{request['subtype']}-{uuid.uuid4().hex[:12]}"  # the kind is read back on errors
        self._write({"type": "control_request", "request_id": request_id, "request": request})

    def _write(self, message: dict[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        with self._write_lock:
            try:
                proc.stdin.write(json.dumps(message) + "\n")
                proc.stdin.flush()
            except (OSError, ValueError):
                pass  # the process is gone; _read reports that

    def _read(self) -> None:
        assert self._proc and self._proc.stdout
        parser = StreamParser()
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "control_request" and (msg.get("request") or {}).get("subtype") != "can_use_tool":
                self._write({"type": "control_response", "response": {
                    "subtype": "error", "request_id": msg.get("request_id"), "error": "grokbot doesn't handle this"}})
                continue
            for event in parser.parse(msg):
                self._on_event(event)
        code = self._proc.wait()
        self._on_event(Exited(code, "".join(self._stderr)))

    def _read_stderr(self) -> None:
        assert self._proc and self._proc.stderr
        for line in self._proc.stderr:
            self._stderr.append(line)
