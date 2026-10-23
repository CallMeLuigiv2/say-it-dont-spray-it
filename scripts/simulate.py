"""Rehearses a whole voice conversation with a fake mic and fake speakers, against real Claude Code.

Spoken lines are synthesized with a second voice and fed in as mic audio, so this exercises
voice detection, speech recognition, Claude Code, permission prompts, echo handling and
barge-in end to end. It uses a little of your Claude plan (Haiku by default).

    uv run python scripts/simulate.py [--model haiku]
"""

from __future__ import annotations

import argparse
import io
import queue
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from rich.console import Console

from grokbot import audio
from grokbot.app import App
from grokbot.config import Settings
from grokbot.tts import Voice
from grokbot.ui import UI

FRAME_SECONDS = audio.FRAME / audio.MIC_RATE


class FakePlayer:
    """Same interface as audio.Player; "plays" in real time and logs what it said."""

    def __init__(self, on_idle):
        self._on_idle = on_idle
        self._clips: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._gen = 0
        self._queued = 0
        self._current = None
        self.played: deque[str] = deque(maxlen=3)
        self._paused = threading.Event()
        self.log: list[tuple[float, str]] = []

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    @property
    def busy(self):
        with self._lock:
            return self._queued > 0

    @property
    def current_text(self):
        return self._current.text if self._current else ""

    def play(self, clip, text="", gen=None):
        with self._lock:
            gen = self._gen if gen is None else gen
            if gen != self._gen:
                return
            self._queued += 1
        self._clips.put(audio.Clip(np.asarray(clip, dtype=np.float32), text, gen))

    def flush(self, gen):
        with self._lock:
            self._gen = gen
        self._paused.clear()

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()

    def close(self):
        self._clips.put(None)

    def _run(self):
        while (clip := self._clips.get()) is not None:
            self._current = clip
            if clip.text and clip.gen == self._gen:
                self.played.append(clip.text)
            if clip.text:
                self.log.append((time.monotonic(), clip.text))
            left = len(clip.audio) / audio.SPEAKER_RATE
            while left > 0 and clip.gen == self._gen:
                time.sleep(0.02)
                if not self._paused.is_set():
                    left -= 0.02
            self._current = None
            with self._lock:
                self._queued -= 1
                idle = self._queued == 0
            if idle:
                self._on_idle()


class NoKeys:
    def start(self): ...
    def stop(self): ...
    def close(self): ...
    def pause(self): ...
    def resume(self): ...


class SimulatedApp(App):
    def open_audio(self):
        self.mic_device = self.speaker_device = None
        self.player = FakePlayer(on_idle=lambda: self.post("idle"))
        self.mic = NoKeys()  # frames are posted by the script instead

    def _on_typed(self, text):
        """Like typing a message with t: for things that are awkward to say, such as code."""
        self.submit(text)


class Script:
    def __init__(self, app: SimulatedApp, user_voice: Voice, screen: io.StringIO):
        self.app = app
        self.user_voice = user_voice
        self.screen = screen
        self.failures: list[str] = []

    # helpers

    def frames(self, clip24k: np.ndarray, rate: int = audio.SPEAKER_RATE) -> list[np.ndarray]:
        pcm = audio.resample((clip24k * 32767).astype(np.int16), rate, audio.MIC_RATE)
        pcm = np.concatenate([pcm, np.zeros(audio.MIC_RATE, dtype=np.int16)])  # then a second of quiet
        return [pcm[i : i + audio.FRAME] for i in range(0, len(pcm) - audio.FRAME + 1, audio.FRAME)]

    def feed(self, frames: list[np.ndarray]) -> None:
        for frame in frames:
            self.app.post("frame", (time.monotonic(), frame))
            time.sleep(FRAME_SECONDS)

    def say(self, text: str) -> None:
        print(f"  [user says] {text}")
        self.feed(self.frames(self.user_voice.synth(text)))

    def silence(self, seconds: float) -> None:
        self.feed([np.zeros(audio.FRAME, dtype=np.int16)] * round(seconds / FRAME_SECONDS))

    def wait(self, label: str, condition, timeout: float = 90) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if condition():
                return True
            self.silence(0.2)  # a real mic keeps sending (quiet) frames
        self.failures.append(f"timed out waiting for: {label}")
        print(f"  !! timed out waiting for: {label}")
        return False

    def idle(self) -> bool:
        a = self.app
        return not a.speaking and not a.outstanding and not a.transcribing and a.permission is None

    def check(self, label: str, ok: bool) -> None:
        print(f"  {'ok ' if ok else 'FAIL'} {label}")
        if not ok:
            self.failures.append(label)

    def said_since(self, since: float) -> str:
        return " ".join(text for at, text in self.app.player.log if at >= since)

    # the conversation

    def run(self, workdir: Path) -> None:
        a = self.app
        self.wait("the greeting", lambda: bool(a.player.log) and self.idle(), 30)

        print("1. a quick question")
        start = time.monotonic()
        self.say("What is two plus two? Answer in one short sentence.")
        self.wait("the answer", lambda: a.player.log and a.player.log[-1][0] > start and self.idle())
        reply = self.said_since(start)
        print(f"     grokbot said: {reply!r}")
        self.check("answered with four", "four" in reply.lower() or "4" in reply)

        print("2. a permission prompt answered by voice")
        start = time.monotonic()
        self.say("Use the Write tool to create a file named hello.txt that contains the word hi.")
        self.wait("the permission question", lambda: a.permission is not None and not a.speaking)
        print(f"     grokbot asked: {self.said_since(start)!r}")
        self.say("Yes, go ahead.")
        self.wait("the write to finish", self.idle)
        self.check("hello.txt was written", (workdir / "hello.txt").exists())

        print("3. echo of my own voice is ignored, then you cut in")
        start = time.monotonic()
        self.say("Tell me a story about a lighthouse keeper in about ten sentences.")
        self.wait("the story to start", lambda: bool(a.player.current_text), 60)
        time.sleep(1.0)
        echo = a.player.current_text
        print(f"     feeding my own voice back: {echo!r}")
        self.feed(self.frames(a.voice.synth(echo)))  # what an open speaker would leak into the mic
        self.wait("the echo check", lambda: not a.transcribing and not a.paused_for_you, 20)
        self.check("kept talking after hearing its own echo", a.echo_strikes == 1 and a.speaking)
        self.say("Stop.")
        self.wait("the story to stop", self.idle, 30)
        told = self.said_since(start)
        self.check("story was cut short", told.count(".") < 9)

        print("4. still remembers the conversation")
        start = time.monotonic()
        self.say("What was the very first math question I asked you?")
        self.wait("the answer", lambda: a.player.log and a.player.log[-1][0] > start and self.idle())
        reply = self.said_since(start)
        print(f"     grokbot said: {reply!r}")
        self.check("remembered two plus two", "two" in reply.lower() or "2" in reply)

        print("5. talking while I'm still thinking adds to the conversation")
        self.say("Look at the files in this folder and tell me what you find.")
        self.wait("thinking", lambda: a.outstanding > 0 and not a.speaking, 30)
        self.say("Also, keep it to one sentence.")
        self.check("second message queued while thinking", a.outstanding == 2 or self.idle() is False)
        self.wait("both answers", self.idle, 120)
        self.check("back to listening after both", self.idle())

        print("6. interrupting with a message still queued doesn't get stuck")
        self.say("Write me a limerick about debugging.")
        self.wait("thinking", lambda: a.outstanding > 0, 30)
        self.say("And after that, a haiku about coffee.")
        queued = a.outstanding
        a.post("key", " ")  # the space bar
        print(f"     pressed space with {queued} message(s) outstanding")
        self.wait("everything to settle", self.idle, 120)
        self.check("not stuck after interrupting", self.idle())

        print("7. auto mode by voice: no more yes or no questions")
        start = time.monotonic()
        asked = self.screen.getvalue().count("allow?")
        self.say("Auto mode on.")
        self.wait("the switch", lambda: a.mode == "auto" and self.idle(), 30)
        print(f"     grokbot said: {self.said_since(start)!r}")
        self.check("switched to auto mode", a.mode == "auto")
        self.check("handled it without asking Claude", "stop asking" in self.said_since(start).lower())
        self.say("Use the Write tool to create a file named auto.txt that contains the word ok.")
        self.wait("the write", lambda: (workdir / "auto.txt").exists() and self.idle(), 120)
        self.check("auto.txt was written", (workdir / "auto.txt").exists())
        self.check("no permission question in auto mode", self.screen.getvalue().count("allow?") == asked)
        if a.permission is not None:
            a.post("key", "n")  # don't leave a question hanging over the next scenario
            self.wait("the refusal", self.idle, 60)

        print("8. spoken check-ins while Claude works quietly")
        a.settings.progress_every = 4
        start = time.monotonic()
        a.post("typed", "Run this exact command and tell me what it printed, in one short sentence: "
                        "python -c \"import time; time.sleep(14); print('ok')\"")
        self.wait("the command", lambda: a.player.log and a.player.log[-1][0] > start + 10 and self.idle(), 120)
        said = self.said_since(start)
        print(f"     grokbot said: {said!r}")
        self.check("checked in while working", "still" in said.lower())
        self.check("summed up the turn on screen", "done in" in self.screen.getvalue())


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="haiku")
    args = parser.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="grokbot-sim-"))
    out = io.StringIO()
    settings = Settings(cwd=workdir, model=args.model, chime=False)
    app = SimulatedApp(settings, UI(Console(file=out, width=110)))
    print("loading models…")
    app.load()
    app.keys = NoKeys()
    script = Script(app, Voice("am_michael", 1.0), out)

    runner = threading.Thread(target=app.run, daemon=True)
    runner.start()
    try:
        script.run(workdir)
    finally:
        app.running = False
        runner.join(timeout=10)
    print("\n--- screen transcript ---")
    print(out.getvalue())
    print(f"session: {app.session_id}  folder: {workdir}")
    if script.failures:
        print("FAILED:", *script.failures, sep="\n  ")
        return 1
    print("all scenarios passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
