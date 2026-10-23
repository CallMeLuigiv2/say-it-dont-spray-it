"""What Claude is up to during a reply: for the status line, the end-of-turn summary, and spoken check-ins."""

from __future__ import annotations

import re
from pathlib import PurePath
from typing import Any
from urllib.parse import urlparse

from grokbot.claude import ToolCall, ToolResult

# Bash and PowerShell descriptions are written as commands ("Run tests"). These verbs read
# better as "running tests"; anything else becomes "running <description>".
_VERBS = frozenset(
    "add analyze apply benchmark build call check clean clone commit compare compile confirm convert copy count "
    "create delete deploy diff display download ensure execute export extract fetch filter find format generate "
    "get grep import inspect install kill launch lint list load look make measure merge move open parse ping "
    "print publish pull push query read rebase remove rename rerun reset restore retry run save search send set "
    "show sort start stash stop sync tag test touch try uninstall unpack update upgrade upload validate verify "
    "view wait watch write".split()
)
_DOUBLED = frozenset({"run", "rerun", "set", "get", "stop", "commit", "tag", "grep", "plan", "ship", "drop", "put"})


def activity(tool: str, data: dict[str, Any]) -> str:
    """A tool call in a few spoken words: "editing app.py", "running tests", "searching the code"."""
    file = data.get("file_path") or data.get("notebook_path")
    name = PurePath(str(file)).name if file else ""
    if tool == "Read":
        return f"reading {name}" if name else "reading a file"
    if tool in ("Edit", "MultiEdit", "NotebookEdit"):
        return f"editing {name}" if name else "editing a file"
    if tool == "Write":
        return f"writing {name}" if name else "writing a file"
    if tool in ("Bash", "PowerShell"):
        return describe_command(str(data.get("description") or ""))
    if tool in ("Grep", "Glob", "LSP"):
        return "searching the code"
    if tool == "WebSearch":
        return "searching the web"
    if tool == "WebFetch":
        host = urlparse(str(data.get("url") or "")).netloc.removeprefix("www.")
        return f"reading {host}" if host else "reading a web page"
    if tool in ("Task", "Agent"):
        what = str(data.get("description") or "").strip().rstrip(".")
        return f"sending a helper to {_lower_first(what)}" if what else "sending a helper"
    if tool.startswith("mcp__"):
        tool = tool.split("__")[-1].replace("_", " ")
    return f"using {tool}"


def describe_command(description: str) -> str:
    """ "Run the tests" -> "running the tests"; "Git status" -> "running git status"."""
    description = " ".join(description.split()).rstrip(".")
    if not description:
        return "running a command"
    first, _, rest = description.partition(" ")
    verb = first.lower()
    if verb.endswith("ing"):
        return _lower_first(description)
    if verb not in _VERBS:
        return f"running {_lower_first(description)}"
    if verb in _DOUBLED:
        gerund = verb + verb[-1] + "ing"
    elif verb.endswith("e") and not verb.endswith("ee"):
        gerund = verb[:-1] + "ing"
    else:
        gerund = verb + "ing"
    return f"{gerund} {rest}".strip()


def clock(seconds: float) -> str:
    """8 -> "8s", 65 -> "1:05"."""
    seconds = int(seconds)
    return f"{seconds}s" if seconds < 60 else f"{seconds // 60}:{seconds % 60:02d}"


class Progress:
    """Follows one reply from the moment you finish talking until Claude is done."""

    def __init__(self) -> None:
        self.started: float | None = None
        self.end()

    @property
    def active(self) -> bool:
        return self.started is not None

    def begin(self, now: float) -> None:
        if self.started is None:
            self.end()
            self.started = now

    def end(self) -> None:
        self.started = None
        self.phase = "thinking"  # thinking | working | writing
        self.steps = 0
        self.failed = 0
        self.doing = ""  # the latest step, e.g. "editing app.py"
        self.thought_tokens = 0
        self.thought = ""  # the latest line of visible thinking, if any
        self._running: dict[str, str] = {}  # tool call id -> tool name, until its result arrives
        self._last_check_in = ""

    # ---- events --------------------------------------------------------------------------

    def thinking(self, tokens: int | None, text: str) -> None:
        if tokens is None and not text:  # a new thought starts
            self.thought_tokens, self.thought = 0, ""
        if tokens:
            self.thought_tokens = tokens
        if text:
            lines = (self.thought + text).splitlines() or [""]
            self.thought = lines[-1][-200:] if lines[-1].strip() else self.thought
        if not self._running:
            self.phase = "thinking"

    def writing(self) -> None:
        self.phase = "writing"

    def tool(self, call: ToolCall) -> None:
        self.steps += 1
        self.phase = "working"
        doing = activity(call.name, call.input)
        self.doing = f"helper: {doing}" if call.nested else doing
        if call.id:
            self._running[call.id] = call.name

    def result(self, result: ToolResult) -> str | None:
        """Notes a finished step. Returns the tool's name if it failed."""
        name = self._running.pop(result.id, None)
        if not self._running:
            self.phase = "thinking"
        if result.ok or name is None:
            return None
        self.failed += 1
        return name

    # ---- reporting -----------------------------------------------------------------------

    def elapsed(self, now: float) -> float:
        return now - self.started if self.started is not None else 0.0

    def detail(self, now: float) -> str:
        """For the status line: "0:23 · 4 steps · editing app.py"."""
        if self.started is None:
            return ""
        parts = [clock(self.elapsed(now))]
        if self.steps:
            parts.append(f"{self.steps} step{'s' if self.steps != 1 else ''}")
        if self.phase == "working" and self.doing:
            parts.append(self.doing)
        elif self.phase == "thinking":
            if self.thought:
                parts.append(f"“{_clip(self.thought, 60)}”")
            elif self.thought_tokens:
                parts.append(f"{self.thought_tokens:,} tokens of thought")
        return " · ".join(parts)

    def summary(self, now: float) -> str:
        """For the end of a turn: "done in 42s · 6 steps · 1 failed"."""
        parts = [f"done in {clock(self.elapsed(now))}"]
        if self.steps:
            parts.append(f"{self.steps} step{'s' if self.steps != 1 else ''}")
        if self.failed:
            parts.append(f"{self.failed} failed")
        return " · ".join(parts)

    def check_in(self) -> str:
        """A short spoken update for when Claude has been quiet a while."""
        if self.phase == "working" and self.doing:
            line = f"Still working. {_upper_first(self.doing)}."
        elif self.phase == "writing":
            line = "Still writing."  # e.g. a long code block, which I don't read out
        elif self.steps:
            line = f"Still working, {self.steps} step{'s' if self.steps != 1 else ''} in."
        else:
            line = "Still thinking."
        if line == self._last_check_in:
            line = "Still on it."
        self._last_check_in = line
        return line


def _lower_first(text: str) -> str:
    # "List files" -> "list files"; acronyms like "NPM" stay as they are.
    return text[:1].lower() + text[1:] if text[1:2].islower() else text


def _upper_first(text: str) -> str:
    return text[:1].upper() + text[1:]


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else "…" + text[-(limit - 1):]
