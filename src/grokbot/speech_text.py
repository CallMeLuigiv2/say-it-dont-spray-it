"""Turns Claude's streamed markdown into short sentences that sound right out loud."""

from __future__ import annotations

import re
import unicodedata
from urllib.parse import urlparse

CODE_NOTICE = "I put the code on your screen."

# A sentence ends at . ! ? or … (plus any closing quote or bracket) followed by whitespace,
# or at a line break. Requiring whitespace keeps "3.5" and "main.py" in one piece.
_BOUNDARY = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s)|\n+")
_ABBREVIATIONS = frozenset({"e.g", "i.e", "etc", "vs", "mr", "mrs", "ms", "dr", "st", "approx", "fig"})
_SOFT_BREAKS = (", ", "; ", ": ", " - ", " — ")


class SentenceSplitter:
    """Feed it text deltas as they stream in; it returns speakable sentences as soon as they're complete.

    Code blocks are skipped (with a one-time "it's on your screen" instead), and the first
    sentence of a reply is cut early at a comma if it runs long, so the voice starts sooner:
    the voice takes about 0.1 s per 10 characters before it can start playing a piece.
    """

    def __init__(self, first_max: int = 60, max_chars: int = 240):
        self._first_max = first_max
        self._max_chars = max_chars
        self.reset()

    def reset(self) -> None:
        self._raw = ""  # not yet examined; may end in half a ``` fence
        self._prose = ""  # waiting for a sentence boundary
        self._in_code = False
        self._mentioned_code = False
        self._spoken = 0

    def feed(self, delta: str) -> list[str]:
        self._raw += delta
        return self._drain(final=False)

    def flush(self) -> list[str]:
        """End of a text block: return whatever is left."""
        out = self._drain(final=True)
        self._raw = ""
        self._in_code = False
        return out

    def end_turn(self) -> list[str]:
        out = self.flush()
        self.reset()
        return out

    def _drain(self, final: bool) -> list[str]:
        out: list[str] = []
        while True:
            if self._in_code:
                end = self._raw.find("```")
                if end < 0:
                    self._raw = "" if final else self._raw[-2:]
                    break
                self._raw = self._raw[end + 3 :]
                self._in_code = False
                continue
            start = self._raw.find("```")
            if start < 0:
                keep = 0 if final else _trailing_backticks(self._raw)
                self._prose += self._raw[: len(self._raw) - keep]
                self._raw = self._raw[len(self._raw) - keep :]
                break
            self._prose += self._raw[:start]
            self._raw = self._raw[start + 3 :]
            self._in_code = True
            out += self._sentences(final=True)
            if not self._mentioned_code:
                self._mentioned_code = True
                out.append(CODE_NOTICE)
        return out + self._sentences(final)

    def _sentences(self, final: bool) -> list[str]:
        out: list[str] = []
        while (cut := self._find_cut()) is not None:
            self._emit(self._prose[:cut], out)
            self._prose = self._prose[cut:]
        if final:
            self._emit(self._prose, out)
            self._prose = ""
        return out

    def _find_cut(self) -> int | None:
        for match in _BOUNDARY.finditer(self._prose):
            if match.group().startswith("\n") or not _is_abbreviation(self._prose, match.start()):
                return match.end()
        limit = self._first_max if self._spoken == 0 else self._max_chars
        if len(self._prose) <= limit:
            return None
        window = self._prose[:limit]
        for sep in _SOFT_BREAKS:
            at = window.rfind(sep)
            if at > limit // 3:
                return at + len(sep)
        at = window.rfind(" ")
        return at + 1 if at > 0 else limit

    def _emit(self, piece: str, out: list[str]) -> None:
        spoken = to_speakable(piece)
        if any(ch.isalnum() for ch in spoken):
            out.append(spoken)
            self._spoken += 1


def _trailing_backticks(text: str) -> int:
    count = len(text) - len(text.rstrip("`"))
    return min(count, 2)


def _is_abbreviation(text: str, period_at: int) -> bool:
    before = text[:period_at].split()
    if not before:
        return False
    word = before[-1].lstrip("([\"'").lower()
    return word in _ABBREVIATIONS or word.isdigit() or (len(word) == 1 and word.isalpha())


_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL = re.compile(r"https?://[^\s)>\]]+")
_INLINE_CODE = re.compile(r"`([^`]*)`")
_PATH = re.compile(r"(?:[A-Za-z]:)?[\\/]?(?:[\w.~-]+[\\/])+[\w.-]+")
_SNAKE = re.compile(r"(?<=[A-Za-z0-9])_(?=[A-Za-z0-9])")
_EMPHASIS = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1")
_HEADING = re.compile(r"^\s*#{1,6}\s*", re.M)
_LIST_MARK = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.M)
_SPOKEN_SYMBOLS = (
    ("e.g.", "for example"), ("i.e.", "that is"), ("etc.", "et cetera"),
    ("->", " to "), ("=>", " to "), ("→", " to "), ("&", " and "), ("|", ", "),
)


def to_speakable(text: str) -> str:
    """Strips markdown and shortens things that sound awful read aloud (URLs, long paths)."""
    text = _LINK.sub(r"\1", text)
    text = _URL.sub(_domain, text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _PATH.sub(_short_path, text)
    text = _SNAKE.sub(" ", text)
    text = _EMPHASIS.sub(r"\2", text)
    text = _HEADING.sub("", text)
    text = _LIST_MARK.sub("", text)
    for symbol, words in _SPOKEN_SYMBOLS:
        text = text.replace(symbol, words)
    text = re.sub(r"~(?=\d)", "about ", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) not in ("So", "Cs", "Mn") and ch not in "*#`~")
    # A trailing comma stays: it tells the voice there's more coming.
    return re.sub(r"\s+", " ", text).strip().lstrip(",").strip()


def _domain(match: re.Match[str]) -> str:
    host = urlparse(match.group()).netloc
    return host.removeprefix("www.") or "a link"


def _short_path(match: re.Match[str]) -> str:
    path = match.group()
    parts = re.split(r"[\\/]", path)
    name = parts[-1]
    # "and/or" stays as is; real paths ("src/app/main.py", "C:\x\y") become just the file name.
    if len(parts) > 2 or "." in name:
        return name
    return path
