from pathlib import Path

from grokbot import vocab
from grokbot.stt import repeats_hints
from grokbot.vocab import MAX_CHARS, PREFIX, Vocabulary, project_terms


def test_prompt_puts_recent_names_last():
    words = Vocabulary()
    words.add_tool({"file_path": r"C:\proj\src\speech_text.py"})
    words.add_text("I renamed `get_user` to `fetch_user` in `src/app.py`, then ran `x`.")
    prompt = words.prompt()
    assert prompt.startswith(PREFIX)
    assert prompt.endswith("grokbot, Claude Code, speech_text.py, get_user, fetch_user, app.py.")


def test_prompt_stays_short_and_keeps_the_newest():
    words = Vocabulary()
    for i in range(200):
        words.add(f"name_number_{i}")
    prompt = words.prompt()
    assert len(prompt) <= MAX_CHARS
    assert prompt.endswith("name_number_199.")
    assert "name_number_100," not in prompt


def test_project_terms_are_shallow_first_and_skip_clutter(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(vocab, "_git_files", lambda cwd: [])  # use the folder walk
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "deep_module.py").write_text("")
    (tmp_path / "README.md").write_text("")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.js").write_text("")
    (tmp_path / ".env").write_text("")
    terms = project_terms(tmp_path)
    assert terms[0] == tmp_path.name
    assert terms.index("README.md") < terms.index("deep_module.py")
    assert not {"junk.js", "node_modules", ".env"} & set(terms)


def test_whisper_reading_the_hints_back_is_caught():
    hints = "Words that may come up: grokbot, Claude Code, app.py, config.toml."
    assert repeats_hints("app.py, config.toml, grokbot.", hints)
    assert not repeats_hints("Open app.py and fix the loader.", hints)
    assert not repeats_hints("app.py", hints)
