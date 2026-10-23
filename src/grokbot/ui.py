"""Terminal display: the conversation scrolls up, a live status line stays at the bottom."""

from __future__ import annotations

import math
import re
import time

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

SPIN = "spin"  # icon placeholder: an animated spinner, so you can see it's alive
_SPINNER = "◐◓◑◒"

# state: (icon, label, style)
STATES = {
    "starting": ("◌", "starting up", "dim"),
    "listening": ("●", "listening", "green"),
    "hearing": ("●", "hearing you", "bold green"),
    "transcribing": (SPIN, "catching that", "cyan"),
    "thinking": (SPIN, "thinking", "magenta"),
    "working": (SPIN, "working", "magenta"),
    "writing": (SPIN, "writing", "magenta"),
    "compacting": (SPIN, "tidying up the conversation", "magenta"),
    "stopping": (SPIN, "stopping", "dim"),
    "speaking": ("▶", "speaking", "blue"),
    "approval": ("?", "yes · no · always · auto", "bold yellow"),
    "muted": ("✕", "mic muted, /mute to unmute", "red"),
    "typing": ("✎", "typing, the mic waits", "cyan"),
}
# permission mode: (badge, style, what it means)
MODES = {
    "default": ("ask", "dim", "I'll ask before edits and commands"),
    "acceptEdits": ("auto-edit", "bold cyan", "I'll edit files without asking, and still ask before commands"),
    "auto": ("AUTO", "bold black on yellow", "I'll work without asking; Claude Code still checks risky actions"),
    "plan": ("plan", "bold blue", "I'll look and plan, without changing anything"),
    "dontAsk": ("no prompts", "yellow", "anything that would need a yes is skipped"),
    "bypassPermissions": ("NO CHECKS", "bold white on red", "nothing is checked"),
}
KEYS = "esc stop · shift+tab mode · /help"
PLACEHOLDER = "talk, or type here · Enter sends · /mute /repeat /quit"
DRAFT_LINES = 4  # how much of a long paste the input line shows
YOU_LINES = 12  # how much of a long typed message stays on screen once sent
_BARS = "▁▂▃▄▅▆▇█"


class UI:
    def __init__(self, console: Console | None = None):
        self.console = console or Console(highlight=False)
        self._state = "starting"
        self._detail = ""
        self._level = 0.0
        self._mode = "default"
        self._preview = ""
        self._draft = ""
        self._live = Live(console=self.console, refresh_per_second=12, transient=True,
                          get_renderable=self._render)

    def start(self) -> None:
        self._live.start()

    def stop(self) -> None:
        self._live.stop()

    def status(self, state: str, detail: str = "") -> None:
        self._state, self._detail = state, detail

    def level(self, value: float) -> None:
        self._level = value

    def mode(self, mode: str) -> None:
        self._mode = mode

    def preview(self, text: str) -> None:
        """The reply as it streams in, shown above the status line until it's complete."""
        self._preview = text

    def draft(self, text: str) -> None:
        """What you're typing, shown on the input line."""
        self._draft = text

    def you(self, text: str) -> None:
        lines = text.splitlines()
        if len(lines) > YOU_LINES:  # a big paste: all of it goes to Claude, the start stays on screen
            text = "\n".join(lines[: YOU_LINES - 2]) + f"\n… (+{len(lines) - YOU_LINES + 2} more lines)"
        self._row(Text("you", style="bold cyan"), Text(text))

    def bot(self, markdown: str) -> None:
        self._preview = ""
        if markdown.strip():
            self._row(Text("grokbot", style="bold magenta"), Markdown(markdown))

    def tool(self, name: str, detail: str, nested: bool = False) -> None:
        icon = "  ↳ " if nested else "⚙ "
        self._row(Text(""), Text.assemble((icon, "dim"), (name, "dim bold"), (f"  {detail}" if detail else "", "dim")))

    def tool_failed(self, name: str, message: str) -> None:
        lines = [line.strip() for line in message.splitlines() if line.strip()]
        if len(lines) > 1 and re.fullmatch(r"Exit code -?\d+", lines[0]):
            lines = [f"{lines[0]} · {lines[1]}"]  # the exit code alone says little; show the first output line too
        first = lines[0] if lines else ""
        first = first if len(first) <= 160 else first[:159] + "…"
        self._row(Text(""), Text.assemble(("✗ ", "red"), (f"{name} failed", "red"), (f"  {first}" if first else "", "dim")))

    def turn_done(self, summary: str) -> None:
        self._row(Text(""), Text(f"✓ {summary}", style="dim green"))

    def permission(self, tool: str, detail: str) -> None:
        self._row(Text("allow?", style="bold yellow"),
                  Text.assemble((tool, "bold"), (f"  {detail}" if detail else "", ""), ("   y/n/a · shift+tab for auto", "dim")))

    def mode_notice(self, mode: str) -> None:
        badge, style, meaning = MODES.get(mode, (mode, "bold", ""))
        self.console.print(Text.assemble(("mode ", "dim"), (f" {badge} ", style), (f"  {meaning}" if meaning else "", "dim")))

    def notice(self, text: str, style: str = "yellow") -> None:
        self.console.print(Text(text, style=style))

    def error(self, text: str) -> None:
        self.console.print(Text(text, style="bold red"))

    def _row(self, label: Text, body: object) -> None:
        grid = Table.grid(padding=(0, 1))
        grid.add_column(width=8, no_wrap=True)
        grid.add_column(ratio=1)
        grid.add_row(label, body)
        self.console.print(grid)

    def _render(self) -> RenderableType:
        parts: list[RenderableType] = []
        if self._preview.strip():
            keep = max(80, (self.console.width - 10) * 3)  # about three lines
            tail = " ".join(self._preview.split())
            tail = tail if len(tail) <= keep else "…" + tail[-keep:]
            grid = Table.grid(padding=(0, 1))
            grid.add_column(width=8, no_wrap=True)
            grid.add_column(ratio=1)
            grid.add_row(Text("grokbot", style="dim magenta"), Text(tail, style="dim"))
            parts.append(grid)
        parts.extend(self._input_lines())
        parts.append(self._status_line())
        return Group(*parts)

    def _input_lines(self) -> list[Text]:
        """The input line: what you're typing, or a hint when it's empty. A long paste shows its last lines."""
        width = self.console.width - 1
        if not self._draft:
            return [Text.assemble((" › ", "bold cyan"), (PLACEHOLDER, "dim"))]
        lines = self._draft.split("\n")
        shown = lines[-DRAFT_LINES:]
        out = []
        if len(lines) > DRAFT_LINES:
            out.append(Text(f"   … {len(lines)} lines", style="dim"))
        for i, line in enumerate(shown):
            prompt = " › " if i == 0 and len(lines) <= DRAFT_LINES else "   "
            room = width - len(prompt) - 1
            line = line.expandtabs(4)
            if len(line) > room:
                line = "…" + line[-(room - 1):]
            text = Text.assemble((prompt, "bold cyan"), (line, ""))
            if i == len(shown) - 1:
                text.append("▌", style="cyan")  # the cursor
            out.append(text)
        return out

    def _status_line(self) -> Text:
        icon, label, style = STATES[self._state]
        if icon == SPIN:
            icon = _SPINNER[int(time.monotonic() * 6) % len(_SPINNER)]
        badge, badge_style, _ = MODES.get(self._mode, (self._mode, "bold", ""))
        line = Text()
        line.append(f" {badge} ", style=badge_style)
        line.append(f" {icon} {label}", style=style)
        if self._state in ("listening", "hearing", "approval"):
            line.append(" " + meter(self._level), style=style)
        if self._detail:
            line.append(f"  {self._detail}", style="dim")
        line.append(f"    {KEYS}", style="dim")
        line.truncate(self.console.width - 1, overflow="ellipsis")  # one line, even in a narrow terminal
        return line


def meter(level: float, width: int = 6) -> str:
    """A tiny mic level bar; level is RMS loudness from 0 to 1."""
    db = 20 * math.log10(max(level, 1e-6))
    filled = max(0.0, min(1.0, (db + 55) / 45)) * width
    return "".join(_BARS[min(7, int((filled - i) * 8))] if filled > i else " " for i in range(width))
