from pathlib import Path

import pytest

from grokbot.config import load_settings


def test_file_then_flags(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text('voice = "am_michael"\nspeed = 1\nmic = 3\nbarge_in = false\n', encoding="utf-8")
    settings = load_settings({"speed": 1.2, "model": None}, path=path)
    assert settings.voice == "am_michael"
    assert settings.speed == 1.2
    assert settings.mic == "3"
    assert settings.barge_in is False
    assert settings.model is None


def test_progress_and_hint_settings(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text("progress_voice = false\nprogress_every = 30\nstt_hints = false\n", encoding="utf-8")
    settings = load_settings({"progress_voice": True}, path=path)
    assert settings.progress_voice is True  # the command line wins
    assert settings.progress_every == 30.0
    assert settings.stt_hints is False


def test_unknown_keys_are_reported(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text('voise = "oops"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="voise"):
        load_settings(path=path)
