"""The turn-taking rules in app.py, with Claude Code, the speakers and the mic replaced by fakes."""

import io
import time

from rich.console import Console

from grokbot import claude
from grokbot.app import App
from grokbot.config import Settings
from grokbot.keys import Typed, parse
from grokbot.ui import UI
from grokbot.vad import Segmenter


class FakeClaude:
    def __init__(self):
        self.sent: list[str] = []
        self.answers: list[tuple[str, bool]] = []
        self.modes: list[str] = []
        self.interrupts = 0

    def send(self, text):
        self.sent.append(text)

    def answer(self, request_id, allow, **_kwargs):
        self.answers.append((request_id, allow))

    def set_mode(self, mode):
        self.modes.append(mode)

    def interrupt(self):
        self.interrupts += 1


class FakePlayer:
    busy = False
    current_text = ""
    played = ()

    def play(self, *_args): ...
    def flush(self, _gen): ...
    def pause(self): ...
    def resume(self): ...


class Silence:
    def __call__(self, _audio):
        return 0.0

    def reset(self): ...


def make_app(**settings) -> tuple[App, io.StringIO]:
    out = io.StringIO()
    app = App(Settings(**settings), UI(Console(file=out, width=120)))
    app.claude = FakeClaude()
    app.player = FakePlayer()
    app.segmenter = Segmenter(detector=Silence())
    return app, out


def said(app: App) -> list[str]:
    """What's been queued to say since last time (and pretends it's been said)."""
    lines = []
    while not app.say_queue.empty():
        lines.append(app.say_queue.get()[1])
        app._unsaid -= 1
    return lines


def hear(app: App, text: str, context: str = "turn", seconds: float = 1.5) -> None:
    app.transcribing += 1
    app._on_heard((context, text, seconds))


def ask(app: App, tool: str = "Edit", tool_input: dict | None = None) -> None:
    app._on_claude(claude.PermissionRequest("r1", tool, "", tool_input or {"file_path": "app.py"}, []))
    said(app)


# ---- auto mode ------------------------------------------------------------------------------


def test_saying_auto_mode_switches_without_bothering_claude():
    app, out = make_app()
    hear(app, "Okay, auto mode on.")
    assert app.claude.modes == ["auto"]
    assert app.claude.sent == []
    assert said(app) == ["Auto mode on. I'll stop asking."]
    app._on_claude(claude.ModeChanged("auto"))
    assert app.mode == "auto"
    assert "AUTO" in out.getvalue()


def test_mode_commands_that_change_nothing():
    app, _ = make_app(permission_mode="auto")
    hear(app, "auto mode")
    assert app.claude.modes == [] and app.claude.sent == []
    assert said(app) == ["Auto mode is already on."]

    app, _ = make_app()
    hear(app, "Ask me first.")  # probably an answer to something Claude asked
    assert app.claude.sent == ["Ask me first."]


def test_answering_auto_approves_and_stops_the_questions():
    app, _ = make_app()
    ask(app, "Bash", {"command": "npm test", "description": "Run tests"})
    hear(app, "Yes, go auto.")
    assert app.claude.answers == [("r1", True)]
    assert app.claude.modes == ["auto"]
    assert app.permission is None


def test_switching_modes_answers_the_question_it_covers():
    app, _ = make_app()
    ask(app, "Edit")
    app._on_key("shift-tab")  # ask -> auto-edit, which covers edits
    assert app.claude.modes == ["acceptEdits"]
    assert app.claude.answers == [("r1", True)]

    app, _ = make_app()
    ask(app, "Bash", {"command": "rm -r build", "description": "Remove the build folder"})
    app._on_key("shift-tab")  # auto-edit doesn't cover commands, so the question stands
    assert app.claude.answers == []
    assert app.permission is not None


def test_shift_tab_cycles_through_the_modes():
    app, _ = make_app()
    for mode in ("acceptEdits", "auto", "default"):
        app._on_key("shift-tab")
        app._on_claude(claude.ModeChanged(mode))
    assert app.claude.modes == ["acceptEdits", "auto", "default"]
    assert app.mode == "default"


def test_mode_changes_claude_makes_itself_are_shown():
    app, out = make_app()
    app._on_claude(claude.ModeChanged("acceptEdits"))  # e.g. after answering "always" to an edit
    assert app.mode == app.wanted_mode == "acceptEdits"
    assert "auto-edit" in out.getvalue()


def test_in_auto_mode_a_question_says_why_it_still_asks():
    app, _ = make_app(permission_mode="auto")
    app._on_claude(claude.PermissionRequest("r1", "Bash", "", {"description": "Force push to main"}, []))
    assert said(app) == ["Auto mode wants you to check this one. Can I run a command to force push to main?"]


# ---- progress -------------------------------------------------------------------------------


def test_a_check_in_after_a_quiet_stretch_of_work():
    app, _ = make_app(progress_every=20)
    app.submit("Refactor the loader.")
    app._on_claude(claude.ToolCall("Edit", "loader.py", "t1", input={"file_path": "src/loader.py"}))
    start = app.progress.started
    app._maybe_check_in(start + 5)
    assert said(app) == []
    app._maybe_check_in(start + 21)
    assert said(app) == ["Still working. Editing loader.py."]
    app._maybe_check_in(start + 30)
    assert said(app) == []  # not again so soon


def test_no_check_ins_when_turned_off():
    app, _ = make_app(progress_voice=False)
    app.submit("Refactor the loader.")
    app._maybe_check_in(app.progress.started + 600)
    assert said(app) == []


def test_talking_over_a_check_in_adds_to_the_work_instead_of_cancelling_it():
    app, _ = make_app()
    app.submit("Refactor the loader.")
    app.check_ins.append("Still working. Editing loader.py.")
    app.player.current_text = "Still working. Editing loader.py."
    app.paused_for_you = True
    hear(app, "Also add a test for it.", context="over")
    assert app.claude.interrupts == 0
    assert app.claude.sent == ["Refactor the loader.", "Also add a test for it."]


def test_an_echo_of_an_earlier_sentence_is_still_recognized():
    app, _ = make_app()
    app.submit("Tell me a story.")
    app.player.played = ["Every night for thirty years,", "Elias climbed the stairs of the old lighthouse."]
    app.player.current_text = "His wife had left him for a sailor from the mainland."
    app.paused_for_you = True
    hear(app, "Every night for 30 years.", context="over")
    assert app.claude.interrupts == 0
    assert app.claude.sent == ["Tell me a story."]
    assert app.echo_strikes == 1


def test_wordless_noise_while_i_talk_counts_as_echo():
    app, out = make_app()
    app.submit("Tell me a story.")
    app.player.current_text = "Once upon a time there was a lighthouse keeper."
    for _ in range(2):
        app.paused_for_you = True
        hear(app, "", context="over")  # e.g. Whisper heard "Thank you." and stt.py dropped it
        assert not app.paused_for_you
    assert app.claude.interrupts == 0
    assert app.claude.sent == ["Tell me a story."]
    assert not app.barge_in
    assert "picking up my voice" in out.getvalue()


def test_talking_over_a_reply_still_cuts_it_off():
    app, _ = make_app()
    app.submit("Tell me a story.")
    app.player.current_text = "Once upon a time there was a lighthouse keeper."
    app.paused_for_you = True
    hear(app, "Actually, fix the failing test instead.", context="over")
    assert app.claude.interrupts == 1


def test_status_line_follows_the_work():
    app, _ = make_app()
    app.submit("Fix it.")
    app._on_claude(claude.Thinking(tokens=240))
    app._refresh(time.monotonic())
    assert app.ui._state == "thinking" and "240 tokens" in app.ui._detail
    app._on_claude(claude.ToolCall("Read", "app.py", "t1", input={"file_path": "app.py"}))
    app._refresh(time.monotonic())
    assert app.ui._state == "working" and "1 step · reading app.py" in app.ui._detail


def test_failed_steps_and_the_turn_summary_are_on_screen():
    app, out = make_app()
    app.submit("Run the tests.")
    app._on_claude(claude.ToolCall("Bash", "Run tests", "t1", input={"description": "Run tests"}))
    app._on_claude(claude.ToolResult("t1", False, "\nError: 2 tests failed\nmore detail"))
    app._on_claude(claude.TurnDone(True, "success", "s1", 0))
    screen = out.getvalue()
    assert "Bash failed" in screen and "Error: 2 tests failed" in screen
    assert "done in" in screen and "1 step · 1 failed" in screen
    assert not app.progress.active


# ---- other conveniences ---------------------------------------------------------------------


def test_say_that_again_repeats_the_last_reply_locally():
    app, _ = make_app()
    app.submit("Hi.")
    app._on_claude(claude.TextDelta("Hello there. "))
    app._on_claude(claude.TextDelta("How are you?"))
    app._on_claude(claude.TextDone("Hello there. How are you?"))
    app._on_claude(claude.TurnDone(True, "success", "s1", 0))
    said(app)
    hear(app, "Sorry, say that again?")
    assert said(app) == ["Hello there.", "How are you?"]
    assert app.claude.sent == ["Hi."]


def test_unclear_speech_gets_a_note_instead_of_silence():
    app, out = make_app()
    hear(app, "", seconds=2.0)
    assert "couldn't make out" in out.getvalue()
    hear(app, "", seconds=0.4)  # a cough or a click: say nothing
    assert out.getvalue().count("couldn't make out") == 1


def test_voice_speed_commands():
    app, _ = make_app()

    class Voice:
        speed = 1.05

    app.voice = Voice()
    type_in(app, "/faster")
    assert app.voice.speed == 1.1
    for _ in range(30):
        type_in(app, "/slower")
    assert app.voice.speed == 0.7
    assert app.claude.sent == []


# ---- typing ---------------------------------------------------------------------------------


def type_in(app: App, text: str, send: bool = True) -> None:
    app._on_key(Typed(text))
    if send:
        app._on_key("enter")


def test_a_multi_line_paste_goes_out_whole_as_one_message():
    app, out = make_app()
    paste = "1. Userspace TCP/IP Stack\nwhat you master: the deepest mechanics\n" + "\n".join(f"line {i}" for i in range(30))
    for event in parse(list(paste.replace("\n", "\r\n"))):
        app._on_key(event)
    assert app.draft == paste
    assert app._mic_mode(time.monotonic()) is None  # the mic waits while you type
    app._on_key("enter")
    assert app.claude.sent == [paste]
    assert app.draft == ""
    assert "more lines" in out.getvalue()  # the screen keeps the start of a long paste, Claude gets all of it


def test_typed_letters_are_text_not_shortcuts():
    app, _ = make_app()
    app.running = True
    app.submit("Tell me a story.")
    app._unsaid = 1  # I'm talking
    type_in(app, "make the mute button quieter", send=False)
    assert app.claude.interrupts == 0 and not app.muted and app.running
    app._on_key("backspace")
    assert app.draft == "make the mute button quiete"


def test_esc_and_space_on_an_empty_line_cut_me_off():
    app, _ = make_app()
    app.submit("Tell me a story.")
    app._on_key("esc")
    assert app.claude.interrupts == 1
    app, _ = make_app()
    app.submit("Tell me a story.")
    app._on_key(Typed(" "))
    assert app.claude.interrupts == 1 and app.draft == ""


def test_esc_clears_the_line_when_there_is_nothing_to_stop():
    app, _ = make_app()
    type_in(app, "half a thought", send=False)
    app._on_key("esc")
    assert app.draft == "" and app.claude.sent == []


def test_y_on_an_empty_line_answers_a_question():
    app, _ = make_app()
    ask(app)
    app._on_key(Typed("y"))
    assert app.claude.answers == [("r1", True)]
    assert app.draft == ""


def test_typed_answer_to_a_question_goes_to_the_question():
    app, _ = make_app()
    ask(app)
    type_in(app, "no, put it in utils instead")
    assert app.claude.answers == [("r1", False)]
    assert app.claude.sent == []


def test_slash_mute_toggles_the_mic():
    app, out = make_app()
    type_in(app, "/mute")
    assert app.muted and app._mic_mode(time.monotonic()) is None
    type_in(app, "/mute")
    assert not app.muted
    assert app.claude.sent == []
    type_in(app, "/quit")
    assert not app.running


def test_a_mistyped_command_stays_local():
    app, out = make_app()
    type_in(app, "/mue")
    assert app.claude.sent == [] and not app.muted
    assert "Did you mean /mute?" in out.getvalue()
    type_in(app, "/etc/hosts is wrong, fix it")  # a sentence that starts with a path still goes to Claude
    assert app.claude.sent == ["/etc/hosts is wrong, fix it"]


def test_the_input_line_shows_the_end_of_a_long_paste():
    app, _ = make_app()
    app._set_draft("\n".join(f"line {i}" for i in range(10)))
    screen = io.StringIO()
    Console(file=screen, width=80).print(app.ui._render())
    shown = screen.getvalue()
    assert "10 lines" in shown and "line 9" in shown and "line 2" not in shown
