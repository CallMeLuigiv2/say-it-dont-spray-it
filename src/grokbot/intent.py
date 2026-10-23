"""Small judgments about what you said: yes or no, "stop", a command for me, or just my own voice echoing back."""

from __future__ import annotations

import re
from pathlib import PurePath
from typing import Any, Literal
from urllib.parse import urlparse

Reply = Literal["yes", "no", "always", "auto", "unclear"]
Command = Literal["auto", "edits", "ask", "repeat"]

_WORD = re.compile(r"[a-z0-9']+")
_YES = re.compile(
    r"\b(yes|yeah|yea|yep|yup|ya|sure|ok|okay|alright|approve[d]?|allow(ed)?|affirmative|absolutely|"
    r"definitely|proceed|fine|correct|granted|go ahead|do it|sounds good|go for it|let'?s go|ship it|why not)\b"
)
_NO = re.compile(
    r"\b(no|nope|nah|don'?t|do not|stop|cancel|deny|denied|wait|hold on|hold up|negative|never|abort|skip|not yet)\b"
)
_ALWAYS = re.compile(r"\b(always|from now on|don'?t ask( me)?( again)?|stop asking|for (this|the) session)\b")
_ALWAYS_ASK = re.compile(r"\b(always ask|ask (me )?every time)\b")
# A "yes" that comes with changes is really "no, do it differently".
_CONDITIONS = re.compile(r"\b(but|instead|except|rather|first)\b")
_STOP = re.compile(r"\b(stop|wait|hold on|hold up|pause|quiet|shut up|enough|cancel|never ?mind|forget it)\b")
# "Approve this and stop asking about anything." Whisper often hears "auto" as "Otto".
_AUTO_REPLY = re.compile(
    r"\b(auto|otto) ?(mode|pilot)\b|\bgo (auto|otto)\b|^(auto|otto)$|\byes to (everything|all)\b|"
    r"\b(approve|allow) (everything|all)\b"
)

# Commands I handle myself instead of passing to Claude. Each must be the whole utterance
# (give or take "okay", "please" and the like), so a sentence that merely mentions them is left alone.
_LEAD = r"(?:(?:ok|okay|hey|grokbot|please|alright|so|and|yeah|now|just|sorry|let's|lets|can you|could you)\s+)*"
_TAIL = r"(?:\s+(?:please|now|thanks|thank you|grokbot))*"
_AUTO = r"(?:auto|otto)"
_COMMANDS: tuple[tuple[Command, re.Pattern[str]], ...] = tuple(
    (name, re.compile(rf"^{_LEAD}(?:{pattern}){_TAIL}$"))
    for name, pattern in (
        ("edits", rf"(?:(?:turn on|switch to|go to|use|enable)\s+)?(?:accept|{_AUTO})\s*edits?(?:\s+mode)?(?:\s+on)?"),
        ("ask", (
            rf"(?:turn off|switch off|stop|exit|leave|disable|end|cancel|quit|no more|get out of)\s+"
            rf"(?:the\s+)?{_AUTO}(?:\s*(?:mode|pilot))?"
            rf"|{_AUTO}\s*(?:mode|pilot)?\s+off"
            rf"|(?:(?:turn on|switch to|go to|go back to|back to|use|enable)\s+)?(?:manual|normal|ask|safe)\s+mode"
            rf"|(?:start\s+)?ask(?:ing)?\s+(?:me\s+)?(?:first|again|before\s+\w+(?:\s+\w+)?)(?:\s+again)?"
            rf"|start asking(?:\s+me)?(?:\s+again)?"
        )),
        ("auto", (
            rf"(?:turn on|switch to|switch on|go|go to|go into|get into|enable|use|start|enter|back to|go back to)"
            rf"\s+{_AUTO}(?:\s*(?:mode|pilot))?(?:\s+on)?"
            rf"|{_AUTO}\s*(?:mode|pilot)(?:\s+on)?|{_AUTO}\s+on"
        )),
        ("repeat", (
            r"(?:repeat|say)\s+(?:that|it|this)(?:\s+again)?|repeat(?:\s+yourself)?|say (?:again|it again)"
            r"|come again|what did you (?:just )?say|one more time|pardon(?: me)?|sorry what|i missed that"
        )),
    )
)


# Whisper often runs "auto mode on" together into one made-up word: "Atomodon", "Automodon", "AutoModaf".
_RUN_TOGETHER = re.compile(r"\b(?:au|a|o)tt?o?mo?d[aeo]?(on|off|of|af)?\b")
_SWITCH = {"on": " on", "off": " off", "of": " off", "af": " off"}


def words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _said(text: str) -> str:
    """The words, lowercased, with run-together "auto mode" pulled apart."""
    said = " ".join(words(text))
    return _RUN_TOGETHER.sub(lambda m: "auto mode" + _SWITCH.get(m.group(1) or "", ""), said)


def command(text: str) -> Command | None:
    """ "Auto mode on", "ask me first", "say that again": things I do myself rather than ask Claude."""
    said = _said(text)
    if not said or len(said.split()) > 8:
        return None
    for name, pattern in _COMMANDS:
        if pattern.match(said):
            return name
    return None


def names_a_mode(text: str) -> bool:
    """Says "mode" or "pilot" outright ("auto mode", "manual mode"), unlike "ask me first"."""
    return bool(re.search(r"(mode|pilot)\b", _said(text)))


def classify_reply(text: str) -> Reply:
    """How you answered "can I do X?"."""
    said = _said(text)
    always = bool(_ALWAYS.search(said)) and not _ALWAYS_ASK.search(said)
    said = _ALWAYS.sub(" ", said).strip()  # so "don't ask again" doesn't count as a "don't"
    yes, no, auto = _YES.search(said), _NO.search(said), _AUTO_REPLY.search(said)
    if auto and not (no and no.start() < auto.start()):
        return "unclear" if _CONDITIONS.search(said[auto.end():]) else "auto"
    if no and (not yes or no.start() < yes.start()):
        return "no"
    if yes and _CONDITIONS.search(said[yes.end():]):
        return "unclear"
    if yes or always:
        return "always" if always else "yes"
    return "unclear"


def is_stop(text: str) -> bool:
    """A short "stop" / "wait" / "never mind" with nothing else to it."""
    return bool(_STOP.search(text.lower())) and len(words(text)) <= 4


def looks_like_echo(heard: str, spoken: str) -> bool:
    """Did the mic just pick up my own voice? True when most of what was heard is words I was saying."""
    heard_words = words(heard)
    if not heard_words:
        return True
    if _STOP.search(heard.lower()) or "grokbot" in heard_words:
        return False
    spoken_words = set(words(spoken))
    if not spoken_words:
        return False
    overlap = sum(w in spoken_words for w in heard_words) / len(heard_words)
    return overlap >= 0.6


def permission_question(tool: str, tool_input: dict[str, Any]) -> str:
    """How to ask out loud for permission to use a tool."""
    if tool in ("Bash", "PowerShell"):
        what = str(tool_input.get("description") or "").strip().rstrip(".")
        return f"Can I run a command to {_lower_first(what)}?" if what else "Can I run a command?"
    file = tool_input.get("file_path") or tool_input.get("notebook_path")
    if file:
        name = PurePath(str(file)).name
        verb = "write" if tool == "Write" else "edit"
        return f"Can I {verb} {name}?"
    if tool == "WebFetch" and tool_input.get("url"):
        host = urlparse(str(tool_input["url"])).netloc.removeprefix("www.")
        return f"Can I open {host or 'that page'}?"
    if tool == "WebSearch" and tool_input.get("query"):
        return f"Can I search the web for {tool_input['query']}?"
    if tool.startswith("mcp__"):
        tool = tool.split("__")[-1].replace("_", " ")
    return f"Can I use {tool}?"


def _lower_first(text: str) -> str:
    # "List files in the folder" -> "list files in the folder"; acronyms like "NPM" stay as they are.
    return text[:1].lower() + text[1:] if text[1:2].islower() else text
