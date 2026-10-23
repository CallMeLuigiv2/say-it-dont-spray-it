"""Splits the mic stream into utterances with Silero voice activity detection."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from grokbot.audio import FRAME, MIC_RATE

FRAME_SECONDS = FRAME / MIC_RATE

START = "start"  # you started talking
DROPPED = "dropped"  # it was a blip too short to be speech


class Detector(Protocol):
    def __call__(self, audio: bytes) -> float: ...
    def reset(self) -> None: ...


@dataclass(frozen=True)
class Utterance:
    audio: np.ndarray  # float32 at 16 kHz
    seconds: float


class Segmenter:
    """Feed it 32 ms int16 frames; it reports when speech starts and hands back whole utterances."""

    def __init__(
        self,
        threshold: float = 0.5,
        end_silence: float = 0.7,
        start_after: float = 0.1,
        min_speech: float = 0.25,
        preroll: float = 0.3,
        max_length: float = 60.0,
        detector: Detector | None = None,
    ):
        if detector is None:
            from pysilero_vad import SileroVoiceActivityDetector

            detector = SileroVoiceActivityDetector()
        self._detector = detector
        self.threshold = threshold
        self._release = max(0.15, threshold - 0.15)  # hysteresis: stay "in speech" a bit below threshold
        self._start_frames = max(1, round(start_after / FRAME_SECONDS))
        self._end_frames = max(1, round(end_silence / FRAME_SECONDS))
        self._min_voiced = max(1, round(min_speech / FRAME_SECONDS))
        self._keep_tail = round(0.25 / FRAME_SECONDS)
        self._max_frames = round(max_length / FRAME_SECONDS)
        self._preroll: deque[np.ndarray] = deque(maxlen=max(1, round(preroll / FRAME_SECONDS)))
        self.reset()

    def reset(self) -> None:
        self._detector.reset()
        self._preroll.clear()
        self._frames: list[np.ndarray] = []
        self._hot = 0
        self._quiet = 0
        self._voiced = 0
        self.speaking = False
        self.probability = 0.0

    def feed(self, frame: np.ndarray) -> str | Utterance | None:
        """Returns START, an Utterance when you stop talking, DROPPED for a blip, or None."""
        p = self._detector(frame.tobytes())
        self.probability = p
        if not self.speaking:
            self._preroll.append(frame)
            self._hot = self._hot + 1 if p >= self.threshold else 0
            if self._hot < self._start_frames:
                return None
            self.speaking = True
            self._frames = list(self._preroll)
            self._preroll.clear()
            self._voiced = self._hot
            self._quiet = 0
            return START

        self._frames.append(frame)
        if p >= self._release:
            self._quiet = 0
            self._voiced += p >= self.threshold
        else:
            self._quiet += 1
        if self._quiet < self._end_frames and len(self._frames) < self._max_frames:
            return None

        frames, voiced, quiet = self._frames, self._voiced, self._quiet
        self.reset()
        if voiced < self._min_voiced:
            return DROPPED
        if quiet > self._keep_tail:
            frames = frames[: len(frames) - quiet + self._keep_tail]
        audio = np.concatenate(frames).astype(np.float32) / 32768.0
        return Utterance(audio, len(audio) / MIC_RATE)
