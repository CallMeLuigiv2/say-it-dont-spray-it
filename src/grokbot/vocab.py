"""Words speech recognition should expect: your project's file names, and names Claude has used lately.

Whisper takes a short text prompt. Listing these words in it makes "config.toml" come out as
"config.toml" instead of "config.tommel". Whisper keeps only the end of a long prompt, so the
most relevant words (the recent ones) go last.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections import OrderedDict
from pathlib import Path, PurePath
from typing import Any

ALWAYS = ("auto mode", "grokbot", "Claude Code")
PREFIX = "Words that may come up: "
MAX_CHARS = 480  # about 150 tokens; Whisper allows 223
_SKIP_DIRS = frozenset({".git", ".venv", "venv", "env", "node_modules", "__pycache__", "build", "dist", "target",
                        ".idea", ".vscode", ".pytest_cache", ".ruff_cache", ".mypy_cache", "site-packages"})
_CODE_SPAN = re.compile(r"`([A-Za-z_][\w./\\-]{2,60})`")


class Vocabulary:
    def __init__(self, cwd: Path | None = None, recent: int = 40):
        self._project = project_terms(cwd) if cwd else []
        self._recent: OrderedDict[str, None] = OrderedDict()
        self._max_recent = recent

    def add(self, term: str) -> None:
        term = term.strip().strip(".,:;")
        if not 3 <= len(term) <= 40 or " " in term:
            return
        self._recent.pop(term, None)
        self._recent[term] = None
        while len(self._recent) > self._max_recent:
            self._recent.popitem(last=False)

    def add_tool(self, data: dict[str, Any]) -> None:
        """Files Claude reads or edits are likely to come up next."""
        for key in ("file_path", "notebook_path"):
            if data.get(key):
                self.add(PurePath(str(data[key])).name)

    def add_text(self, text: str) -> None:
        """Names Claude put in `code` in its reply, so you can say them back."""
        for term in _CODE_SPAN.findall(text):
            self.add(re.split(r"[\\/]", term)[-1])

    def prompt(self) -> str:
        terms: list[str] = []
        seen: set[str] = set()
        for term in reversed([*self._project, *ALWAYS, *self._recent]):  # newest first, keep each once
            if term.lower() not in seen:
                seen.add(term.lower())
                terms.append(term)
        kept: list[str] = []
        size = len(PREFIX)
        for term in terms:
            size += len(term) + 2
            if size > MAX_CHARS:
                break
            kept.append(term)
        return PREFIX + ", ".join(reversed(kept)) + "."


def project_terms(cwd: Path, limit: int = 60) -> list[str]:
    """File and folder names from the project, shallowest first."""
    paths = _git_files(cwd) or _walk(cwd)
    paths.sort(key=lambda p: (len(p.parts), str(p).lower()))
    names: dict[str, None] = {cwd.name: None} if cwd.name else {}
    for path in paths:
        for part in (*path.parts[:-1], path.name):
            if 3 <= len(part) <= 40 and not part.startswith("."):
                names.setdefault(part, None)
        if len(names) >= limit:
            break
    return list(names)[:limit]


def _git_files(cwd: Path) -> list[PurePath]:
    try:
        done = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=cwd, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=3, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if done.returncode != 0:
        return []
    return [PurePath(line) for line in done.stdout.splitlines()[:5000] if line]


def _walk(cwd: Path, depth: int = 2, budget: int = 2000) -> list[PurePath]:
    found: list[PurePath] = []
    pending = [(cwd, 0)]
    while pending and len(found) < budget:
        folder, level = pending.pop()
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith(".") or entry.name in _SKIP_DIRS:
                continue
            rel = PurePath(entry.path).relative_to(cwd)
            found.append(rel)
            if level < depth and entry.is_dir(follow_symlinks=False):
                pending.append((Path(entry.path), level + 1))
    return found
