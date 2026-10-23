"""The conversation loop: mic → voice detection → Whisper → Claude Code → Kokoro → speakers.

All conversation state lives on one thread (`run`). The mic, the transcriber, Claude Code,
the voice and the keyboard each post events to a queue, and `run` handles them in order,
so the turn-taking rules are all in this file.
"""

from __future__ import annotations

import difflib
import queue
import random
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from importlib import resources
from pathlib import Path

from grokbot import audio, claude, intent
from grokbot.config import Settings
from grokbot.keys import KeyReader, Typed
from grokbot.progress import Progress
from grokbot.speech_text import SentenceSplitter
from grokbot.stt import Transcriber
from grokbot.tts import Voice
from grokbot.ui import UI
from grokbot.vad import DROPPED, START, Segmenter, Utterance
from grokbot.vocab import Vocabulary

GREETINGS = (
    "Hey, I'm listening.",
    "I'm here. What are we working on?",
    "Ready when you are.",
    "Go ahead, I'm all ears.",
)
MIC_TAIL = 0.35  # keep the mic shut this long after I stop talking, so I don't hear my own echo

# Permission modes, by Claude Code's names. "manual" is the command line's name for "default".
MODE_CYCLE = ("default", "acceptEdits", "auto")  # what shift+tab steps through
COMMAND_MODES = {"ask": "default", "edits": "acceptEdits", "auto": "auto"}
MODE_SPOKEN = {
    "default": "Okay, I'll ask before changing things.",
    "acceptEdits": "Okay, I'll edit files without asking, and still check before commands.",
    "auto": "Auto mode on. I'll stop asking.",
}
MODE_ALREADY = {
    "default": "I'm already asking first.",
    "acceptEdits": "Auto-edit is already on.",
    "auto": "Auto mode is already on.",
}
EDIT_TOOLS = frozenset({"Edit", "MultiEdit", "Write", "NotebookEdit"})
AUTO_TIP_AFTER = 3  # approvals before I mention auto mode
SPEED_RANGE = (0.7, 1.6)
SLASH_COMMANDS = ("/mute", "/repeat", "/faster", "/slower", "/quit", "/help")
HELP = ("Type and press Enter to send; a paste or Ctrl+Enter adds new lines. Esc stops me, or clears what you typed. "
        "Shift+Tab switches modes. While a question waits, y, n or a answers it. "
        "Commands: /mute /repeat /faster /slower /quit")


def default_persona() -> Path:
    return Path(str(resources.files("grokbot").joinpath("persona.md")))


class App:
    def __init__(self, settings: Settings, ui: UI):
        self.settings = settings
        self.ui = ui
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.running = False
        self.session_id: str | None = None

        # Speaking
        self.gen = 0  # bumping this drops everything queued to be said
        self.splitter = SentenceSplitter()
        self.say_queue: queue.Queue[tuple[int, str] | None] = queue.Queue()
        self._unsaid_lock = threading.Lock()
        self._unsaid = 0  # sentences queued but not yet handed to the player
        self.reply: list[str] = []  # sentences of the reply in progress, for "say that again"
        self.last_reply: list[str] = []
        self.check_ins: deque[str] = deque(maxlen=4)  # progress lines I said, so talking over one isn't a cancel
        self.quiet_since = 0.0  # when I last stopped talking
        self.preview = ""  # the reply text streamed so far, shown on screen

        # Turn-taking
        self.outstanding = 0  # messages Claude hasn't finished answering
        self.discarding = False  # ignoring the rest of a reply you interrupted
        self.held: list[str] = []  # what you said while that reply was winding down
        self.permission: claude.PermissionRequest | None = None
        self.explained_approval = False
        self.approvals = 0
        self.transcribing = 0
        self.paused_for_you = False  # I paused mid-sentence because you started talking
        self.barge_in = settings.barge_in
        self.echo_strikes = 0
        self.muted = False
        self.draft = ""  # what you're typing on the input line; the mic waits while there's any
        self.mic_open_at = 0.0
        self.mic_was_open = False
        self.warned_usage = False

        # What Claude is doing
        mode = settings.permission_mode or "default"
        self.mode = "default" if mode == "manual" else mode  # as Claude Code last reported it
        self.wanted_mode = self.mode  # what we last asked for
        self.progress = Progress()
        self.compacting = False
        self.vocab = Vocabulary()  # filled with the project's names by load()

    # ---- setup and teardown ---------------------------------------------------------------

    def load(self) -> None:
        """Loads the models and opens the devices. Slow (a few seconds), so it's separate from run().

        Claude Code starts first and boots while the speech models load side by side.
        """
        s = self.settings
        self.claude = claude.ClaudeSession(
            cwd=s.cwd, persona=s.persona or default_persona(), on_event=lambda e: self.post("claude", e),
            model=s.model, effort=s.effort, permission_mode=s.permission_mode,
            resume=s.resume, continue_session=s.continue_session,
        )
        self.claude.start()
        try:
            with ThreadPoolExecutor(3, thread_name_prefix="load") as pool:
                hearing = pool.submit(self._load_hearing)
                voice = pool.submit(self._load_voice)
                vocab = pool.submit(Vocabulary, s.cwd) if s.stt_hints else None
                self.stt, self.voice = hearing.result(), voice.result()
                if vocab:
                    self.vocab = vocab.result()
            self.segmenter = Segmenter(threshold=s.vad_threshold, end_silence=s.end_silence)
            self.open_audio()
        except BaseException:
            self.claude.close()
            raise
        self.keys = KeyReader(lambda ch: self.post("key", ch))
        self.stt_pool = ThreadPoolExecutor(1, thread_name_prefix="stt")

    def _load_hearing(self) -> Transcriber:
        stt = Transcriber(self.settings.whisper_model, self.settings.whisper_device)
        stt.warm_up()
        return stt

    def _load_voice(self) -> Voice:
        voice = Voice(self.settings.voice, self.settings.speed)
        voice.synth("Warming up.")
        return voice

    def open_audio(self) -> None:
        self.mic_device = audio.resolve_device(self.settings.mic, "input")
        self.speaker_device = audio.resolve_device(self.settings.speaker, "output")
        self.player = audio.Player(self.speaker_device, on_idle=lambda: self.post("idle"))
        self.mic = audio.Mic(self.mic_device, on_frame=lambda f: self.post("frame", (time.monotonic(), f)))

    def post(self, kind: str, payload: object = None) -> None:
        self.events.put((kind, payload))

    def run(self) -> str | None:
        """Talks until you quit. Returns the Claude Code session id, so the conversation can be resumed."""
        self.running = True
        self.player.start()
        threading.Thread(target=self._speak_loop, name="tts", daemon=True).start()
        self.ui.mode(self.mode)
        self.say(random.choice(GREETINGS))
        self.mic.start()
        self.keys.start()
        self.ui.start()
        try:
            while self.running:
                try:
                    kind, payload = self.events.get(timeout=0.1)
                    getattr(self, f"_on_{kind}")(payload)
                except queue.Empty:
                    pass
                now = time.monotonic()
                self._maybe_check_in(now)
                self._refresh(now)
        finally:
            self.shutdown()
        return self.session_id

    def shutdown(self) -> None:
        self.running = False
        steps = (
            self.ui.stop, self.keys.stop, self.mic.close, self.claude.close,
            lambda: self.say_queue.put(None), self.player.close,
            lambda: self.stt_pool.shutdown(wait=False, cancel_futures=True),
        )
        for step in steps:
            try:
                step()
            except Exception:
                pass

    # ---- speaking -------------------------------------------------------------------------

    @property
    def speaking(self) -> bool:
        with self._unsaid_lock:
            unsaid = self._unsaid
        return unsaid > 0 or self.player.busy

    def say(self, text: str) -> None:
        with self._unsaid_lock:
            self._unsaid += 1
        self.say_queue.put((self.gen, text))

    def _say_reply(self, sentence: str) -> None:
        self.reply.append(sentence)
        self.say(sentence)

    def hush(self) -> None:
        """Stop talking now. The reply that's still streaming in carries on from where it is."""
        self.gen += 1
        self.player.flush(self.gen)
        self.paused_for_you = False

    def stop_speaking(self) -> None:
        """Stop talking and forget the rest of the reply."""
        self.hush()
        self.splitter.reset()

    def _speak_loop(self) -> None:
        """Runs on its own thread: turns queued sentences into audio for the player."""
        while (item := self.say_queue.get()) is not None:
            gen, text = item
            try:
                if gen == self.gen:
                    clip = self.voice.synth(text)
                    if clip.size and gen == self.gen:
                        self.player.play(clip, text, gen)
            except Exception as exc:
                self.post("error", f"The voice couldn't say {text!r}: {exc}")
            finally:
                with self._unsaid_lock:
                    self._unsaid -= 1
                self.post("idle")

    # ---- listening ------------------------------------------------------------------------

    def _mic_mode(self, now: float) -> str | None:
        """"turn" when it's your turn, "interject" while I'm busy, None when the mic is ignored."""
        if self.muted or self.draft or now < self.mic_open_at:
            return None
        if self.speaking:
            return "interject" if (self.barge_in or self.paused_for_you) else None
        if self.permission is not None:
            return "turn"
        if self.outstanding or self.transcribing:
            return "interject"  # I'm thinking, not talking, so there's no echo to worry about
        return "turn"

    def _on_frame(self, payload: object) -> None:
        stamp, frame = payload  # type: ignore[misc]
        mode = self._mic_mode(stamp)
        self.ui.level(audio.rms(frame) if mode else 0.0)
        if mode is None:
            if self.mic_was_open:
                self.segmenter.reset()
                self.mic_was_open = False
            return
        self.mic_was_open = True
        result = self.segmenter.feed(frame)
        if result is None:
            return
        if result == START:
            if self.speaking and self.player.current_text and self.barge_in:
                self.player.pause()  # you're talking over me: hold my sentence while I check it's really you
                self.paused_for_you = True
            return
        if result == DROPPED:
            self._resume_if_paused()
            return
        assert isinstance(result, Utterance)
        context = "over" if self.paused_for_you else mode
        if context == "turn" and self.permission is None and self.settings.chime:
            self.player.play(audio.chime("heard"), "", self.gen)
        self.transcribing += 1
        hints = self.vocab.prompt() if self.settings.stt_hints else ""
        self.stt_pool.submit(self._transcribe, result, context, hints)

    def _transcribe(self, utterance: Utterance, context: str, hints: str) -> None:
        try:
            text = self.stt.transcribe(utterance.audio, hints)
        except Exception as exc:
            self.post("error", f"Speech recognition failed: {exc}")
            text = ""
        self.post("heard", (context, text, utterance.seconds))

    def _resume_if_paused(self) -> None:
        if self.paused_for_you:
            self.paused_for_you = False
            self.player.resume()

    def _on_heard(self, payload: object) -> None:
        context, text, seconds = payload  # type: ignore[misc]
        self.transcribing -= 1
        command = self._command(text) if text else None
        if context == "over":
            if intent.looks_like_echo(text, " ".join([self.player.current_text, *self.player.played])):
                self._resume_if_paused()
                # No words at all counts too: something that sounded like speech while I talked, with
                # nothing of yours in it, is almost always my voice coming back through the speakers.
                self._count_echo()
                return
            if command:
                pass  # a command doesn't cancel what you talked over
            elif self.player.current_text in self.check_ins:
                self.hush()  # you talked over a progress check-in, not a reply, so Claude keeps working
                context = "interject"
            elif self.permission is None:
                self.interrupt()
                if intent.is_stop(text):
                    self.ui.you(text)
                    return
            else:
                self.stop_speaking()
        if command:
            self.ui.you(text)
            self._run_command(command)
        elif text:
            self._handle(text, context)
        elif context == "turn" and seconds >= 1.0:
            self.ui.notice("(I heard you but couldn't make out the words. Try again?)", style="dim")

    def _handle(self, text: str, context: str) -> None:
        if self.permission is not None:
            self._answer(text)
        elif context == "interject" and intent.is_stop(text):
            self.ui.you(text)
            self.interrupt()
        else:
            self.submit(text)

    def _count_echo(self) -> None:
        self.echo_strikes += 1
        if self.echo_strikes >= 2 and self.barge_in:
            self.barge_in = False
            self.ui.notice(
                "Your mic is picking up my voice, so I'll stop listening while I talk. "
                "Use headphones to talk over me, or press Esc to cut me off."
            )

    # ---- my own commands ------------------------------------------------------------------

    def _command(self, text: str) -> intent.Command | None:
        """A command for me. "Ask me first" when I already do is more likely an answer meant for Claude."""
        command = intent.command(text)
        unchanged = command in COMMAND_MODES and COMMAND_MODES[command] == self.wanted_mode
        if unchanged and self.permission is None and not intent.names_a_mode(text):
            return None
        return command

    def _run_command(self, command: intent.Command) -> None:
        if command == "repeat":
            self._repeat()
        else:
            self._set_mode(COMMAND_MODES[command], spoken=True)

    def _set_mode(self, mode: str, spoken: bool = False) -> None:
        """Switches the permission mode. A switch that covers the question I'm waiting on answers it yes."""
        request = self.permission
        if request is not None and (mode == "auto" or (mode == "acceptEdits" and request.tool in EDIT_TOOLS)):
            self.hush()  # the question is moot
            self.permission = None
            self.claude.answer(request.request_id, True, tool_input=request.tool_input)
        else:
            self._resume_if_paused()
            if mode == self.wanted_mode and spoken and request is None:
                self.say(MODE_ALREADY.get(mode, "Okay."))
        if mode == self.wanted_mode:
            return
        self.wanted_mode = mode
        self.claude.set_mode(mode)  # Claude Code confirms with ModeChanged
        if spoken:
            self.say(MODE_SPOKEN.get(mode, "Okay."))

    def _cycle_mode(self) -> None:
        if self.wanted_mode in MODE_CYCLE:
            mode = MODE_CYCLE[(MODE_CYCLE.index(self.wanted_mode) + 1) % len(MODE_CYCLE)]
        else:
            mode = "default"
        self._set_mode(mode)

    def _mode_confirmed(self, mode: str, announce: bool = True) -> None:
        if self.wanted_mode == self.mode:
            self.wanted_mode = mode  # Claude Code switched by itself (an "always" answer, or leaving plan mode)
        if mode == self.mode:
            return
        self.mode = mode
        self.ui.mode(mode)
        if announce:
            self.ui.mode_notice(mode)

    def _repeat(self) -> None:
        """Says the last reply again, or the question I'm waiting on. All local: no plan usage."""
        self.hush()
        if self.permission is not None:
            self.say(intent.permission_question(self.permission.tool, self.permission.tool_input))
            return
        sentences = self.reply or self.last_reply
        for sentence in sentences or ["I haven't said anything yet."]:
            self.say(sentence)

    def _nudge_speed(self, step: float) -> None:
        low, high = SPEED_RANGE
        self.voice.speed = round(min(high, max(low, self.voice.speed + step)), 2)
        self.ui.notice(f"voice speed {self.voice.speed:.2f}", style="dim")

    # ---- talking to Claude ----------------------------------------------------------------

    def submit(self, text: str, show: bool = True) -> None:
        if show:
            self.ui.you(text)
        if self.discarding:
            self.held.append(text)  # sent once the interrupted reply has wound down
            return
        self._send(text)

    def _send(self, text: str) -> None:
        self.claude.send(text)
        self.outstanding += 1
        self.progress.begin(time.monotonic())

    def interrupt(self) -> None:
        """Stop talking and cancel the reply in progress."""
        self.stop_speaking()
        self.preview = ""
        self.ui.preview("")
        if self.permission is not None:
            self.claude.answer(self.permission.request_id, False, message="The user interrupted.")
            self.permission = None
        if self.outstanding and not self.discarding:
            self.claude.interrupt()
            self.discarding = True
            self.ui.notice("stopped", style="dim")

    def _on_claude(self, event: claude.Event) -> None:
        match event:
            case claude.Ready():
                if event.api_key_source not in (None, "none"):
                    self.ui.error(
                        f"Claude Code is using an API key ({event.api_key_source}) instead of your plan. "
                        "Stopping so you aren't billed for API usage."
                    )
                    self.running = False
                    return
                if self.session_id is None:
                    self.ui.notice(f"Claude Code is up · {event.model} · on your plan", style="dim")
                self.session_id = event.session_id
                if event.permission_mode:
                    self._mode_confirmed(event.permission_mode, announce=False)
            case claude.Thinking(tokens=tokens, text=text):
                if not self.discarding:
                    self.progress.thinking(tokens, text)
            case claude.TextDelta(text=text):
                if not self.discarding:
                    self.progress.writing()
                    self.preview += text
                    self.ui.preview(self.preview)
                    for sentence in self.splitter.feed(text):
                        self._say_reply(sentence)
            case claude.TextDone(text=text):
                if not self.discarding:
                    self.preview = ""
                    self.ui.bot(text)
                    self.vocab.add_text(text)
                    for sentence in self.splitter.flush():
                        self._say_reply(sentence)
            case claude.ToolCall():
                if not self.discarding:
                    self.ui.tool(event.name, event.detail, event.nested)
                    self.progress.tool(event)
                    self.vocab.add_tool(event.input)
            case claude.ToolResult():
                failed = self.progress.result(event)
                if failed and not self.discarding:
                    self.ui.tool_failed(failed, event.text)
            case claude.PermissionRequest():
                self._ask(event)
            case claude.TurnDone():
                self._turn_done(event)
            case claude.Usage():
                self._check_usage(event)
            case claude.ModeChanged(mode=mode):
                self._mode_confirmed(mode)
            case claude.Status(status=status):
                self.compacting = status == "compacting"
            case claude.ControlError(message=message, request=request):
                if request == "set_permission_mode":
                    self.wanted_mode = self.mode
                    self.ui.error(f"Couldn't switch modes: {message}")
            case claude.Exited(code=code, stderr=stderr):
                if self.running:
                    self.ui.error(f"Claude Code stopped (exit code {code}).\n{stderr.strip()[-800:]}")
                    self.running = False

    def _turn_done(self, event: claude.TurnDone) -> None:
        now = time.monotonic()
        self.outstanding = max(0, self.outstanding - 1)
        if self.discarding:
            self.splitter.reset()
        else:
            for sentence in self.splitter.end_turn():
                self._say_reply(sentence)
            if not event.ok:
                self.ui.error(f"Claude Code reported an error ({event.subtype}).")
                self.say("Something went wrong on my end. The details are on screen.")
            if self.progress.steps or self.progress.elapsed(now) >= 8:
                self.ui.turn_done(self.progress.summary(now))
        self.discarding = False
        self.preview = ""
        self.ui.preview("")
        if self.reply:
            self.last_reply, self.reply = self.reply, []
        self.progress.end()
        if self.outstanding:
            self.progress.begin(now)  # Claude moves straight on to your next message
        if event.session_id:
            self.session_id = event.session_id
        if self.held:
            text = " ".join(self.held)
            self.held.clear()
            self._send(text)

    def _check_usage(self, usage: claude.Usage) -> None:
        if self.warned_usage:
            return
        if usage.status not in ("allowed", "") or (usage.utilization or 0) >= 0.9:
            self.warned_usage = True
            used = f" ({usage.utilization:.0%} used)" if usage.utilization is not None else ""
            self.ui.notice(f"Heads up: you're near your plan's {usage.window or 'usage'} limit{used}.")

    def _maybe_check_in(self, now: float) -> None:
        """After a long quiet stretch of work, says briefly what Claude is doing."""
        s = self.settings
        if not (s.progress_voice and s.progress_every > 0 and self.outstanding and self.progress.started):
            return
        busy = (self.discarding or self.speaking or self.transcribing or self.paused_for_you
                or self.permission is not None or self.segmenter.speaking)
        if busy or now - max(self.progress.started, self.quiet_since) < s.progress_every:
            return
        line = "Still tidying up our conversation." if self.compacting else self.progress.check_in()
        self.check_ins.append(line)
        self.quiet_since = now
        self.say(line)

    # ---- permission prompts ---------------------------------------------------------------

    def _ask(self, request: claude.PermissionRequest) -> None:
        if self.discarding:
            self.claude.answer(request.request_id, False, message="The user interrupted.")
            return
        for sentence in self.splitter.flush():
            self._say_reply(sentence)
        self.permission = request
        self.ui.permission(request.tool, request.detail)
        question = intent.permission_question(request.tool, request.tool_input)
        if self.wanted_mode == "auto":
            question = "Auto mode wants you to check this one. " + question
        elif not self.explained_approval:
            question += " Say yes, no, or always. Or say auto mode, and I'll stop asking."
            self.explained_approval = True
        self.say(question)

    def _answer(self, text: str, reply: intent.Reply | None = None) -> None:
        request = self.permission
        if request is None:
            return
        reply = reply or intent.classify_reply(text)
        if text:
            self.ui.you(text)
        if reply == "unclear" and len(intent.words(text)) <= 2:
            self.say("Sorry, was that a yes or a no?")
            return
        if reply == "auto":
            self._set_mode("auto", spoken=bool(text))
            return
        self.permission = None
        if reply in ("yes", "always"):
            remember = request.suggestions if reply == "always" else None
            self.claude.answer(request.request_id, True, tool_input=request.tool_input, remember=remember)
            if reply == "always":
                self.ui.notice("Okay, I won't ask about that again this session.", style="dim")
            self.approvals += 1
            if self.approvals == AUTO_TIP_AFTER and self.wanted_mode == "default":
                self.ui.notice('Tip: say "auto mode" or press shift+tab, and I\'ll stop asking.', style="dim")
        elif reply == "no":
            said = f' They said: "{text}"' if text else ""
            self.claude.answer(request.request_id, False, message=f"The user said no.{said}")
        else:
            self.claude.answer(request.request_id, False,
                               message=f'The user didn\'t approve this. They said: "{text}". Do what they asked.')

    # ---- keyboard and housekeeping --------------------------------------------------------

    def _on_key(self, event: object) -> None:
        busy = self.speaking or self.outstanding or self.permission is not None
        if isinstance(event, Typed):
            self._type(event.text, busy)
        elif event == "enter":
            self._send_draft()
        elif event == "backspace":
            self._set_draft(self.draft[:-1])
        elif event == "esc":
            if busy:
                self.interrupt()
            else:
                self._set_draft("")
        elif event == "shift-tab":
            self._cycle_mode()
        elif event == "ctrl-c":
            self.running = False

    def _type(self, text: str, busy: bool) -> None:
        if not self.draft:
            if text == " ":
                if busy:
                    self.interrupt()  # space on an empty line cuts me off, as it always has
                return
            if self.permission is not None and text.lower() in ("y", "n", "a"):
                self._answer("", {"y": "yes", "n": "no", "a": "always"}[text.lower()])  # type: ignore[arg-type]
                return
        self._set_draft(self.draft + text)

    def _set_draft(self, text: str) -> None:
        self.draft = text
        self.ui.draft(text)

    def _send_draft(self) -> None:
        text = self.draft.strip()
        self._set_draft("")
        self.mic_open_at = time.monotonic() + 0.2  # skip the sound of the Enter key
        if not text:
            return
        if text.lower() in SLASH_COMMANDS:
            self._slash(text.lower())
        elif re.fullmatch(r"/[a-z]+", text.lower()):
            close = difflib.get_close_matches(text.lower(), SLASH_COMMANDS, n=1)
            hint = f"Did you mean {close[0]}?" if close else "Commands: " + " ".join(SLASH_COMMANDS)
            self.ui.notice(f"There's no {text} command. {hint}", style="dim")  # a typo shouldn't go to Claude
        elif command := self._command(text):
            self.ui.you(text)
            self._run_command(command)
        elif self.permission is not None:
            self._answer(text)
        else:
            self.submit(text)

    def _slash(self, name: str) -> None:
        if name == "/mute":
            self.muted = not self.muted
            self.segmenter.reset()
            self.mic_was_open = False
            self._resume_if_paused()
            self.ui.notice("mic muted. /mute again to unmute." if self.muted else "mic on", style="dim")
        elif name == "/repeat":
            self._repeat()
        elif name == "/faster":
            self._nudge_speed(0.05)
        elif name == "/slower":
            self._nudge_speed(-0.05)
        elif name == "/quit":
            self.running = False
        else:
            self.ui.notice(HELP, style="dim")

    def _on_idle(self, _payload: object) -> None:
        if not self.speaking:
            now = time.monotonic()
            self.mic_open_at = max(self.mic_open_at, now + MIC_TAIL)
            self.quiet_since = now

    def _on_error(self, message: object) -> None:
        self.ui.error(str(message))

    def _refresh(self, now: float) -> None:
        if self.muted:
            state = "muted"
        elif self.paused_for_you or self.segmenter.speaking:
            state = "hearing"
        elif self.transcribing:
            state = "transcribing"
        elif self.permission is not None and not self.speaking:
            state = "approval"
        elif self.speaking:
            state = "speaking"
        elif self.discarding:
            state = "stopping"
        elif self.outstanding:
            state = "compacting" if self.compacting else self.progress.phase
        else:
            state = "typing" if self.draft else "listening"
        detail = self.progress.detail(now) if self.outstanding and state != "approval" else ""
        self.ui.status(state, detail)
