import pytest

from grokbot.intent import classify_reply, command, is_stop, looks_like_echo, names_a_mode, permission_question


@pytest.mark.parametrize(
    "said, expected",
    [
        ("Yes.", "yes"),
        ("yeah go ahead", "yes"),
        ("Sure, do it.", "yes"),
        ("okay", "yes"),
        ("No.", "no"),
        ("Nope, don't.", "no"),
        ("Wait, hold on.", "no"),
        ("No, go ahead actually", "no"),
        ("Yes, always.", "always"),
        ("Go ahead and don't ask me again.", "always"),
        ("don't ask again", "always"),
        ("No, always ask me first", "no"),
        ("Yes but call it main.py instead", "unclear"),
        ("What does it do?", "unclear"),
        ("", "unclear"),
        ("Auto mode.", "auto"),
        ("Yes, auto mode.", "auto"),
        ("Go Otto.", "auto"),
        ("auto", "auto"),
        ("Yes to everything.", "auto"),
        ("Automodon.", "auto"),
        ("No, not auto mode.", "no"),
        ("Auto mode, but use main.py instead", "unclear"),
    ],
)
def test_classify_reply(said, expected):
    assert classify_reply(said) == expected


@pytest.mark.parametrize(
    "said, expected",
    [
        ("Auto mode.", "auto"),
        ("Okay, turn on auto mode please.", "auto"),
        ("Go auto.", "auto"),
        ("Otto mode on", "auto"),
        ("Autopilot.", "auto"),
        ("Switch to auto-mode.", "auto"),
        ("Turn off auto mode.", "ask"),
        ("Auto mode off.", "ask"),
        ("Stop auto mode", "ask"),
        ("Ask me first again.", "ask"),
        ("Manual mode.", "ask"),
        ("Ask me before editing anything.", "ask"),
        ("Accept edits.", "edits"),
        ("Auto edit mode", "edits"),
        ("Atomodon.", "auto"),  # what Whisper makes of a quick "auto mode on"
        ("Otomodon", "auto"),
        ("Automode.", "auto"),
        ("AutoModaf.", "ask"),
        ("Automotive.", None),
        ("Say that again?", "repeat"),
        ("Can you repeat that?", "repeat"),
        ("What did you say?", "repeat"),
        ("Auto", None),  # might be the answer to a question Claude asked
        ("Use the auto formatter on app.py", None),
        ("What is auto mode?", None),
        ("Turn on auto mode and then fix the failing tests in app.py", None),
        ("Repeat the test three times.", None),
        ("", None),
    ],
)
def test_command(said, expected):
    assert command(said) == expected


def test_names_a_mode():
    assert names_a_mode("auto mode") and names_a_mode("Autopilot") and names_a_mode("manual mode")
    assert not names_a_mode("ask me first")


def test_is_stop_only_for_short_commands():
    assert is_stop("Stop.")
    assert is_stop("Wait, hold on.")
    assert is_stop("never mind")
    assert not is_stop("Stop using tabs in the config file and switch to spaces")
    assert not is_stop("What time is it?")


def test_looks_like_echo():
    spoken = "I added a retry loop to the fetch function and the tests pass."
    assert looks_like_echo("added a retry loop to the fetch", spoken)
    assert looks_like_echo("", spoken)
    assert not looks_like_echo("actually make it three retries", spoken)
    assert not looks_like_echo("stop", spoken)
    assert not looks_like_echo("hey grokbot the tests", spoken)


def test_permission_question():
    ask = permission_question
    assert ask("Bash", {"command": "ls", "description": "List files in the folder"}) == (
        "Can I run a command to list files in the folder?"
    )
    assert ask("Bash", {"command": "npm test", "description": "NPM test run"}) == "Can I run a command to NPM test run?"
    assert ask("Write", {"file_path": r"C:\x\notes.md"}) == "Can I write notes.md?"
    assert ask("Edit", {"file_path": "/a/b/app.py"}) == "Can I edit app.py?"
    assert ask("WebFetch", {"url": "https://www.example.com/x"}) == "Can I open example.com?"
    assert ask("mcp__github__create_issue", {}) == "Can I use create issue?"
