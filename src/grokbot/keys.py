"""Keyboard input read straight from the Windows console: what you type, plus a few named keys."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

try:
    import msvcrt
except ImportError:  # not Windows: no typing, voice still works
    msvcrt = None  # type: ignore[assignment]

PASTE_GAP = 0.015  # keys that arrive closer together than this came from one paste
_SPECIAL = {"\x0f": "shift-tab"}  # the second character of a two-character key
_CONTROL = {"\r": "enter", "\x08": "backspace", "\x1b": "esc", "\x03": "ctrl-c"}


@dataclass(frozen=True)
class Typed:
    text: str


Event = Typed | str  # typed text, or a key's name: "enter", "backspace", "esc", "shift-tab", "ctrl-c"


def parse(burst: list[str]) -> list[Event]:
    """Turns one burst of keystrokes into typed text and named keys.

    A burst is everything that arrived together. A paste arrives all at once, so inside a burst
    Enter is a line break; only an Enter on its own sends. Ctrl+Enter (a bare "\\n") is a line break too.
    """
    events: list[Event] = []

    def typed(text: str) -> None:
        if events and isinstance(events[-1], Typed):
            events[-1] = Typed(events[-1].text + text)
        else:
            events.append(Typed(text))

    pasted = len(burst) > 1
    previous = ""
    for ch in burst:
        if not ch:
            continue  # a key with no meaning here, like an arrow
        if ch == "\r" and not pasted:
            events.append("enter")
        elif ch in ("\r", "\n"):
            if not (ch == "\n" and previous == "\r"):  # \r\n is one line break
                typed("\n")
        elif ch in _CONTROL:
            events.append(_CONTROL[ch])
        elif len(ch) > 1:
            events.append(ch)  # already named, like "shift-tab"
        elif ch == "\t" or ch.isprintable():
            typed(ch)
        previous = ch
    return events


class KeyReader:
    def __init__(self, on_key: Callable[[Event], None]):
        self._on_key = on_key
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="keys", daemon=True)

    def start(self) -> None:
        if msvcrt is not None:
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            if not msvcrt.kbhit():
                time.sleep(0.02)
                continue
            for event in parse(self._burst()):
                self._on_key(event)

    def _burst(self) -> list[str]:
        """Everything waiting, plus whatever follows within a moment (a big paste can come in pieces)."""
        burst: list[str] = []
        while True:
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):  # arrow and function keys arrive as two characters
                    ch = _SPECIAL.get(msvcrt.getwch(), "")
                burst.append(ch)
            time.sleep(PASTE_GAP)
            if not msvcrt.kbhit():
                return burst
