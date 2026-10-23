"""Where grokbot keeps its models and settings on disk."""

from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """Downloaded models live here: %LOCALAPPDATA%/grokbot on Windows."""
    base = os.environ.get("LOCALAPPDATA") or Path.home() / ".local" / "share"
    return Path(base) / "grokbot"


def config_path() -> Path:
    """Optional settings file: %APPDATA%/grokbot/config.toml on Windows."""
    base = os.environ.get("APPDATA") or Path.home() / ".config"
    return Path(base) / "grokbot" / "config.toml"


def kokoro_dir() -> Path:
    return data_dir() / "models" / "kokoro"
