import numpy as np

from grokbot.vad import DROPPED, START, Segmenter, Utterance


class FakeDetector:
    """Says "speech" for loud frames and "silence" for quiet ones."""

    def __call__(self, audio: bytes) -> float:
        return 0.9 if np.frombuffer(audio, dtype=np.int16).max() > 1000 else 0.05

    def reset(self) -> None:
        pass


def frames(kind: str, seconds: float) -> list[np.ndarray]:
    level = 5000 if kind == "speech" else 10
    return [np.full(512, level, dtype=np.int16) for _ in range(round(seconds / 0.032))]


def run(segmenter: Segmenter, stream: list[np.ndarray]) -> list:
    return [r for f in stream if (r := segmenter.feed(f)) is not None]


def test_utterance_ends_after_the_pause():
    seg = Segmenter(end_silence=0.7, detector=FakeDetector())
    results = run(seg, frames("quiet", 1) + frames("speech", 1.5) + frames("quiet", 1))
    assert results[0] == START
    assert isinstance(results[1], Utterance)
    assert 1.5 < results[1].seconds < 2.4  # speech + preroll + a short tail, not the whole pause
    assert len(results) == 2


def test_short_pause_does_not_end_the_turn():
    seg = Segmenter(end_silence=0.7, detector=FakeDetector())
    results = run(seg, frames("speech", 1) + frames("quiet", 0.4) + frames("speech", 1) + frames("quiet", 1))
    assert sum(isinstance(r, Utterance) for r in results) == 1


def test_blips_are_dropped():
    seg = Segmenter(end_silence=0.3, min_speech=0.25, detector=FakeDetector())
    results = run(seg, frames("speech", 0.13) + frames("quiet", 0.6))
    assert results == [START, DROPPED]
