"""Text to speech with Kokoro, running locally (no API, no plan usage)."""

from __future__ import annotations

import numpy as np

from grokbot.models import kokoro_files

# Kokoro voice ids start with a two-letter code: accent, then gender.
VOICE_GROUPS = {
    "af": "American, female", "am": "American, male",
    "bf": "British, female", "bm": "British, male",
}


class ModelsMissing(RuntimeError):
    pass


class Voice:
    def __init__(self, name: str = "af_heart", speed: float = 1.05):
        from kokoro_onnx import Kokoro

        model, voices = kokoro_files()
        if not (model.exists() and voices.exists()):
            raise ModelsMissing("The voice model isn't downloaded yet. Run `grokbot setup`.")
        self._kokoro = Kokoro(str(model), str(voices))
        self.speed = speed
        self.name = ""
        self.use(name)

    @property
    def names(self) -> list[str]:
        """English voices only; the model also ships other languages."""
        return sorted(v for v in self._kokoro.get_voices() if v[:2] in VOICE_GROUPS)

    def use(self, name: str) -> None:
        if name not in self._kokoro.get_voices():
            raise ValueError(f"There's no voice called {name!r}. Run `grokbot voices` to see them.")
        self.name = name

    def synth(self, text: str) -> np.ndarray:
        """Float32 audio at 24 kHz. Empty if there's nothing pronounceable."""
        if not any(ch.isalnum() for ch in text):
            return np.zeros(0, dtype=np.float32)
        lang = "en-gb" if self.name.startswith("b") else "en-us"
        audio, _rate = self._kokoro.create(text, voice=self.name, speed=self.speed, lang=lang)
        return np.asarray(audio, dtype=np.float32)
