"""Speech to text with faster-whisper, running locally (no API, no plan usage)."""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

# Whisper sometimes invents these from background noise, or from my own voice coming back through
# the speakers. A real "thank you" gets dropped too, which costs nothing.
_PHANTOMS = frozenset({
    "you", "thanks for watching", "thank you for watching", "please subscribe",
    "subtitles by the amara org community", "bye bye",
    "thank you", "thank you very much", "thank you so much", "thanks", "thank you bye",
})
# The CUDA libraries faster-whisper (CTranslate2) loads at runtime on Windows.
_CUDA_DLLS = ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll")


def default_model(device: str) -> str:
    """Big and accurate on a GPU (about 0.2 s per sentence); small and quick on a CPU.

    large-v3-turbo is as fast as distil-large-v3 on a GPU but hears names and file names
    better, and it uses the vocabulary prompt (distil-large-v3 mostly ignores it).
    """
    return "large-v3-turbo" if device == "cuda" else "base.en"


class Transcriber:
    def __init__(self, model: str | None = None, device: str = "auto"):
        from faster_whisper import WhisperModel

        self.device = pick_device(device)
        self.model_name = model or default_model(self.device)
        options = {
            "device": self.device,
            "compute_type": "float16" if self.device == "cuda" else "int8",
            "cpu_threads": min(8, os.cpu_count() or 4),
        }
        try:  # use the cached copy without asking Hugging Face for updates
            self._model = WhisperModel(self.model_name, local_files_only=True, **options)
        except Exception:
            self._model = WhisperModel(self.model_name, **options)

    def warm_up(self) -> None:
        self.transcribe(np.zeros(8000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray, hints: str = "") -> str:
        """`hints` is a prompt listing words you're likely to say (see vocab.py)."""
        text = self._run(audio, hints)
        if hints and repeats_hints(text, hints):
            text = self._run(audio, "")  # Whisper read the word list back instead of listening
        return "" if is_phantom(text) else text

    def _run(self, audio: np.ndarray, prompt: str) -> str:
        segments, _info = self._model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=False,  # the audio is already cut to your speech
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=prompt or None,
        )
        kept = [
            s.text.strip() for s in segments
            if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)
        ]
        return " ".join(t for t in kept if t).strip()


def is_phantom(text: str) -> bool:
    """A stock phrase Whisper makes up when there was no real speech."""
    return _normalize(text) in _PHANTOMS


def repeats_hints(text: str, hints: str) -> bool:
    """True when the transcript is mostly the hint list itself rather than what you said."""
    heard = re.findall(r"[a-z0-9]+", text.lower())
    if len(heard) < 3:
        return False
    listed = set(re.findall(r"[a-z0-9]+", hints.lower()))
    return sum(word in listed for word in heard) / len(heard) >= 0.8


def pick_device(choice: str) -> str:
    """"auto" means the GPU if it's usable, otherwise the CPU."""
    if choice == "cpu":
        return "cpu"
    if cuda_ready():
        return "cuda"
    if choice == "cuda":
        raise RuntimeError(
            "GPU speech recognition needs an NVIDIA GPU and the CUDA libraries. "
            "Install them with `uv sync --extra gpu` in the grokbot folder, or use --whisper-device cpu."
        )
    return "cpu"


def cuda_ready() -> bool:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() == 0:
            return False
    except Exception:
        return False
    if os.name != "nt":
        return True
    _expose_nvidia_dlls()
    import ctypes

    try:
        for dll in _CUDA_DLLS:
            ctypes.WinDLL(dll)
    except OSError:
        return False
    return True


def _expose_nvidia_dlls() -> None:
    """The gpu extra puts NVIDIA's DLLs inside site-packages; make sure Windows can find them."""
    try:
        import nvidia  # namespace package from nvidia-cublas-cu12 / nvidia-cudnn-cu12
    except ImportError:
        return
    for base in nvidia.__path__:
        for bin_dir in Path(base).glob("*/bin"):
            if str(bin_dir) not in os.environ.get("PATH", ""):
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z ]+", "", text.lower()).strip()
