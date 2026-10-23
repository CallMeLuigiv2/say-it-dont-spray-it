"""Downloads the local speech models on first run."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn

from grokbot.paths import kokoro_dir

KOKORO_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
# SHA-256 of each release file, so a corrupted or swapped download is caught.
KOKORO_FILES = {
    "kokoro-v1.0.onnx": "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
    "voices-v1.0.bin": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
}


def kokoro_files() -> tuple[Path, Path]:
    folder = kokoro_dir()
    return folder / "kokoro-v1.0.onnx", folder / "voices-v1.0.bin"


def kokoro_ready() -> bool:
    return all(path.exists() for path in kokoro_files())


def download_kokoro(console: Console) -> None:
    """Fetches the Kokoro voice model (about 340 MB) unless it's already there."""
    folder = kokoro_dir()
    folder.mkdir(parents=True, exist_ok=True)
    columns = (TextColumn("{task.description}"), BarColumn(), DownloadColumn(), TransferSpeedColumn())
    with Progress(*columns, console=console) as progress:
        for name, sha256 in KOKORO_FILES.items():
            dest = folder / name
            if not dest.exists():
                _download(KOKORO_RELEASE + name, dest, sha256, progress)


def download_whisper(name: str) -> None:
    """Fetches a faster-whisper model into the Hugging Face cache."""
    from faster_whisper import download_model

    download_model(name)


def whisper_ready(name: str) -> bool:
    """Is this speech recognition model already downloaded (or a local folder)?"""
    from faster_whisper import download_model

    if Path(name).is_dir():
        return True
    try:
        download_model(name, local_files_only=True)
    except Exception:
        return False
    return True


def _download(url: str, dest: Path, sha256: str | None, progress: Progress) -> None:
    partial = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "grokbot"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0) or None
        task = progress.add_task(dest.name, total=total)
        while chunk := response.read(1 << 20):
            out.write(chunk)
            digest.update(chunk)
            progress.advance(task, len(chunk))
    if sha256 and digest.hexdigest() != sha256:
        partial.unlink()
        raise RuntimeError(f"{dest.name} didn't match its checksum. Run `grokbot setup` to try again.")
    partial.replace(dest)
