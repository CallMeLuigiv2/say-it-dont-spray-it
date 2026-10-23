from grokbot.stt import is_phantom


def test_stock_phrases_whisper_invents_are_phantoms():
    for text in ("Thank you.", "Thank you very much.", " thanks!", "Thanks for watching!", "you"):
        assert is_phantom(text)
    for text in ("Thank you, now run the tests.", "Fix the bug.", ""):
        assert not is_phantom(text)
