"""Microphone input and speaker output."""

from __future__ import annotations

import queue
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

MIC_RATE = 16_000
FRAME = 512  # samples per mic frame: 32 ms at 16 kHz, the size Silero VAD wants
SPEAKER_RATE = 24_000  # Kokoro's output rate


class DeviceError(RuntimeError):
    pass


def list_devices(kind: str) -> list[tuple[int, str, bool]]:
    """(index, name, is_default) for each input or output on the default host API."""
    channels = "max_input_channels" if kind == "input" else "max_output_channels"
    default = sd.query_devices(kind=kind)["index"]
    return [
        (i, d["name"], i == default)
        for i, d in enumerate(sd.query_devices())
        if d[channels] > 0 and d["hostapi"] == sd.default.hostapi
    ]


def resolve_device(spec: str | None, kind: str) -> int | None:
    """Turns a device number or a piece of its name into a device index (None = default)."""
    if not spec:
        return None
    if spec.isdigit():
        return int(spec)
    channels = "max_input_channels" if kind == "input" else "max_output_channels"
    matches = [
        (i, d) for i, d in enumerate(sd.query_devices())
        if d[channels] > 0 and spec.lower() in d["name"].lower()
    ]
    if not matches:
        raise DeviceError(f"No {kind} device matches {spec!r}. Run `grokbot devices` to see the names.")
    # Prefer the default host API (MME on Windows): it converts sample rates for us.
    matches.sort(key=lambda m: m[1]["hostapi"] != sd.default.hostapi)
    return matches[0][0]


def device_name(index: int | None, kind: str) -> str:
    info = sd.query_devices(kind=kind) if index is None else sd.query_devices(index)
    return str(info["name"])


def rms(frame: np.ndarray) -> float:
    """Loudness of an int16 frame, 0.0 to 1.0."""
    samples = frame.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(samples * samples)))


class Mic:
    """Streams 32 ms frames of 16 kHz int16 audio to `on_frame` (called on PortAudio's thread)."""

    def __init__(self, device: int | None, on_frame: Callable[[np.ndarray], None]):
        self._on_frame = on_frame
        self._pending = np.zeros(0, dtype=np.int16)
        try:
            self._stream = sd.InputStream(
                samplerate=MIC_RATE, channels=1, dtype="int16", blocksize=FRAME,
                device=device, callback=self._callback,
            )
            self._rate = MIC_RATE
        except sd.PortAudioError:
            # Some devices only run at their native rate; resample ourselves.
            self._rate = int(sd.query_devices(device, "input")["default_samplerate"])
            self._stream = sd.InputStream(
                samplerate=self._rate, channels=1, dtype="int16", device=device, callback=self._callback,
            )

    def start(self) -> None:
        self._stream.start()

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()

    def _callback(self, indata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        samples = indata[:, 0]
        if self._rate != MIC_RATE:
            samples = resample(samples, self._rate, MIC_RATE)
        self._pending = np.concatenate([self._pending, samples])
        while len(self._pending) >= FRAME:
            self._on_frame(self._pending[:FRAME].copy())
            self._pending = self._pending[FRAME:]


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    count = int(round(len(samples) * dst_rate / src_rate))
    positions = np.linspace(0, len(samples) - 1, count)
    return np.interp(positions, np.arange(len(samples)), samples.astype(np.float32)).astype(np.int16)


@dataclass(frozen=True)
class Clip:
    audio: np.ndarray
    text: str
    gen: int


class Player:
    """Plays clips in order on one output stream, and can pause, resume, or drop them.

    Every clip carries a generation number. `flush(gen)` moves to a new generation,
    which silently drops whatever older clips are queued or playing.
    """

    BLOCK = 1024  # ~43 ms per write, so pause and flush react quickly

    def __init__(self, device: int | None, on_idle: Callable[[], None]):
        self._stream = sd.OutputStream(
            samplerate=SPEAKER_RATE, channels=1, dtype="float32", device=device, latency="low",
        )
        self._on_idle = on_idle
        self._clips: queue.Queue[Clip | None] = queue.Queue()
        self._lock = threading.Lock()
        self._gen = 0
        self._queued = 0
        self._current: Clip | None = None
        # What was actually played lately (the voice synthesizes ahead, so what it has made is no guide
        # to what the mic could be hearing back).
        self.played: deque[str] = deque(maxlen=3)
        self._paused = threading.Event()
        self._thread = threading.Thread(target=self._run, name="player", daemon=True)

    def start(self) -> None:
        self._stream.start()
        self._thread.start()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._queued > 0

    @property
    def current_text(self) -> str:
        clip = self._current
        return clip.text if clip else ""

    def play(self, audio: np.ndarray, text: str = "", gen: int | None = None) -> None:
        with self._lock:
            gen = self._gen if gen is None else gen
            if gen != self._gen:
                return
            self._queued += 1
        self._clips.put(Clip(np.asarray(audio, dtype=np.float32), text, gen))

    def flush(self, gen: int) -> None:
        with self._lock:
            self._gen = gen
        self._paused.clear()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    def close(self) -> None:
        self._clips.put(None)
        self._thread.join(timeout=1)
        self._stream.abort()
        self._stream.close()

    def _run(self) -> None:
        silence = np.zeros((self.BLOCK, 1), dtype=np.float32)
        while (clip := self._clips.get()) is not None:
            self._current = clip
            if clip.text and clip.gen == self._gen:
                self.played.append(clip.text)
            pos = 0
            while pos < len(clip.audio) and clip.gen == self._gen:
                if self._paused.is_set():
                    self._stream.write(silence)  # keep the stream fed while paused
                    continue
                block = clip.audio[pos : pos + self.BLOCK]
                self._stream.write(block.reshape(-1, 1))
                pos += len(block)
            self._current = None
            with self._lock:
                self._queued -= 1
                idle = self._queued == 0
            if idle:
                self._on_idle()


def chime(kind: str = "heard") -> np.ndarray:
    """A soft two-note blip: rising when I've heard you, falling for an error."""
    notes = {"heard": (660.0, 990.0), "error": (440.0, 330.0)}[kind]
    t = np.arange(int(SPEAKER_RATE * 0.07)) / SPEAKER_RATE
    tones = [np.sin(2 * np.pi * f * t) * np.hanning(len(t)) for f in notes]
    return (np.concatenate(tones) * 0.12).astype(np.float32)
