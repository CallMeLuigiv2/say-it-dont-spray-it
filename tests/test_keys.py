from grokbot.keys import Typed, parse


def test_a_lone_enter_sends():
    assert parse(["\r"]) == ["enter"]


def test_enter_inside_a_paste_is_a_line_break():
    assert parse(list("one\r\ntwo\rthree\r")) == [Typed("one\ntwo\nthree\n")]


def test_ctrl_enter_is_a_line_break():
    assert parse(["\n"]) == [Typed("\n")]


def test_named_keys_and_control_characters():
    assert parse(["\x1b"]) == ["esc"]
    assert parse(["\x08"]) == ["backspace"]
    assert parse(["shift-tab"]) == ["shift-tab"]
    assert parse(list("ab\x08c")) == [Typed("ab"), "backspace", Typed("c")]
    assert parse(["", "\x07", "x"]) == [Typed("x")]  # an arrow key and a stray control character drop out


def test_tabs_in_a_paste_are_kept():
    assert parse(list("def f():\r\n\treturn 1")) == [Typed("def f():\n\treturn 1")]
