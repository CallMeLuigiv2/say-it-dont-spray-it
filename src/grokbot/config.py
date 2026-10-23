"""Settings: built-in defaults, then config.toml, then command-line flags."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from grokbot.paths import config_path


@dataclass
class Settings:
    # Claude Code
    cwd: Path = field(default_factory=Path.cwd)
    model: str | None = None  # None = whatever Claude Code defaults to
    effort: str | None = None
    permission_mode: str | None = None
    resume: str | None = None
    continue_session: bool = False
    persona: Path | None = None  # None = the built-in persona.md
    # Voice out
    voice: str = "af_heart"
    speed: float = 1.05
    # Voice in
    whisper_model: str | None = None  # None = large-v3-turbo on a GPU, base.en on a CPU
    whisper_device: str = "auto"  # "auto" uses an NVIDIA GPU when the gpu extra is installed
    stt_hints: bool = True  # tell speech recognition your project's file names, so it spells them right
    end_silence: float = 0.7  # seconds of quiet that end your turn
    vad_threshold: float = 0.5
    barge_in: bool = True  # talk over me to cut me off
    chime: bool = True  # short blip when I've heard you
    # Progress
    progress_voice: bool = True  # say a short "still working, editing app.py" during long quiet stretches
    progress_every: float = 20.0  # seconds of quiet before each of those
    # Devices: part of a device name, or its number from `grokbot devices`. None = Windows default.
    mic: str | None = None
    speaker: str | None = None


# Keys allowed in config.toml (per-session things like `resume` stay on the command line).
FILE_KEYS = frozenset(
    {"model", "effort", "permission_mode", "persona", "voice", "speed", "whisper_model", "whisper_device",
     "stt_hints", "end_silence", "vad_threshold", "barge_in", "chime", "progress_voice", "progress_every",
     "mic", "speaker"}
)


def load_settings(overrides: dict[str, object] | None = None, path: Path | None = None) -> Settings:
    settings = Settings()
    path = path or config_path()
    if path.exists():
        with path.open("rb") as f:
            data = tomllib.load(f)
        unknown = set(data) - FILE_KEYS
        if unknown:
            raise ValueError(f"Unknown setting(s) in {path}: {', '.join(sorted(unknown))}")
        _apply(settings, data)
    _apply(settings, {k: v for k, v in (overrides or {}).items() if v is not None})
    return settings


def _apply(settings: Settings, values: dict[str, object]) -> None:
    known = {f.name for f in fields(Settings)}
    for name, value in values.items():
        if name not in known:
            raise ValueError(f"Unknown setting: {name}")
        if name in ("cwd", "persona"):
            value = Path(str(value)).expanduser().resolve()
        elif name in ("mic", "speaker"):
            value = str(value)
        elif name in ("speed", "end_silence", "vad_threshold", "progress_every"):
            value = float(value)  # type: ignore[arg-type]
        setattr(settings, name, value)
