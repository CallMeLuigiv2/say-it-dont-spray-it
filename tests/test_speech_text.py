from grokbot.speech_text import CODE_NOTICE, SentenceSplitter, to_speakable


def stream(text: str, step: int = 3) -> list[str]:
    """Feeds text in small chunks, like Claude's streamed deltas."""
    splitter = SentenceSplitter()
    out: list[str] = []
    for i in range(0, len(text), step):
        out += splitter.feed(text[i : i + step])
    return out + splitter.end_turn()


def test_splits_sentences_as_they_complete():
    splitter = SentenceSplitter()
    assert splitter.feed("Sure thing. I fixed") == ["Sure thing."]
    assert splitter.feed(" the bug! Anything") == ["I fixed the bug!"]
    assert splitter.flush() == ["Anything"]


def test_keeps_numbers_filenames_and_abbreviations_together():
    assert stream("Version 3.5 is in main.py now. It handles e.g. retries fine. Done.") == [
        "Version 3.5 is in main.py now.",
        "It handles for example retries fine.",
        "Done.",
    ]


def test_skips_code_blocks_and_mentions_them_once():
    text = "Here's the fix.\n```python\nprint('hi')\n```\nAnd a second one.\n```js\nx()\n```\nThat's it."
    assert stream(text) == ["Here's the fix.", CODE_NOTICE, "And a second one.", "That's it."]


def test_code_fence_split_across_deltas():
    splitter = SentenceSplitter()
    out = splitter.feed("Look. `") + splitter.feed("``py\nsecret()\n`") + splitter.feed("``\nDone.")
    assert out + splitter.flush() == ["Look.", CODE_NOTICE, "Done."]


def test_long_first_sentence_is_cut_early_at_a_comma():
    first, *rest = stream(
        "Okay so the thing about this function, which is honestly pretty gnarly, "
        "is that it never closes the file handle it opens."
    )
    assert first.endswith(",") and len(first) <= 90
    assert rest


def test_markdown_lists_become_separate_sentences():
    assert stream("Two things:\n- **First** item\n- Second `thing`\n") == ["Two things:", "First item", "Second thing"]


def test_to_speakable_shortens_links_and_paths():
    linked = "See [the docs](https://x.ai/docs) or https://www.github.com/a/b"
    assert to_speakable(linked) == "See the docs or github.com"
    assert to_speakable(r"I edited C:\Users\Owner\projects\grokbot\src\app.py") == "I edited app.py"
    assert to_speakable("Open src/grokbot/tts.py and/or the README") == "Open tts.py and/or the README"
    assert to_speakable("Set last_assistant_message -> done ✅") == "Set last assistant message to done"
    assert to_speakable("## Heading") == "Heading"


def test_nothing_speakable_is_dropped():
    assert stream("---\n\n```\nonly code\n```") == [CODE_NOTICE]
