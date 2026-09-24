import io
import time

import pytest

from livetext.controls import Command, StdinReader, TextController

ENTER, ESC, BACKSPACE = 13, 27, 8


def test_plain_line_replaces_text():
    c = TextController("a")
    c.handle_line("Nouveau texte\n")
    assert c.text == "Nouveau texte"


def test_empty_line_is_ignored():
    c = TextController("garde-moi")
    c.handle_line("\n")
    assert c.text == "garde-moi"


def test_slash_lines_are_commands_and_double_slash_is_literal():
    c = TextController("x")
    c.handle_line("/erase inpaint")
    c.handle_line("/RESET")
    c.handle_line("//pas une commande")
    assert c.pop_commands() == [Command("erase", "inpaint"), Command("reset")]
    assert c.text == "/pas une commande"


def test_clear_command_empties_text():
    c = TextController("x")
    c.handle_line("/clear")
    assert c.text == ""


def test_window_banner_updates_display_text_live():
    c = TextController("avant")
    c.handle_key(ord("t"))
    assert c.editing and c.buffer == "avant"
    for _ in range(5):
        c.handle_key(BACKSPACE)
    for ch in "Été":
        c.handle_key(ord(ch))
    # Mise à jour immédiate : le texte incrusté suit la frappe.
    assert c.display_text == "Été"
    assert c.text == "avant"
    c.handle_key(ENTER)
    assert not c.editing and c.text == "Été"


def test_escape_cancels_editing_and_does_not_quit():
    c = TextController("avant")
    c.handle_key(ENTER)
    c.handle_key(ord("z"))
    assert c.handle_key(ESC) is None
    assert c.text == "avant" and c.display_text == "avant"
    # Échap hors saisie ne coupe pas le direct.
    assert c.handle_key(ESC) is None


def test_hotkeys_outside_editing():
    c = TextController("")
    assert c.handle_key(ord("q")) == Command("quit")
    assert c.handle_key(ord("r")) == Command("reset")
    assert c.handle_key(ord("s")) == Command("select")
    assert c.handle_key(ord("d")) == Command("debug")
    assert c.handle_key(-1) is None


def test_hotkey_letters_are_typed_while_editing():
    c = TextController("")
    c.handle_key(ENTER)
    for ch in "qrsd":
        assert c.handle_key(ord(ch)) is None
    assert c.buffer == "qrsd"


def test_stdin_reader_runs_in_background_thread():
    c = TextController("init")
    StdinReader(c, io.StringIO("premier\n/debug\nsecond\n")).start()
    deadline = time.time() + 2
    while c.text != "second" and time.time() < deadline:
        time.sleep(0.01)
    assert c.text == "second"
    assert c.pop_commands() == [Command("debug")]


# -- Réglages fins en direct (flèches, + / -, 0) ------------------------------

GTK_LEFT, GTK_UP, GTK_RIGHT, GTK_DOWN = 65361, 65362, 65363, 65364


@pytest.mark.parametrize("code,expected", [
    (GTK_RIGHT, "0.5 0.0"), (GTK_LEFT, "-0.5 0.0"), (GTK_UP, "0.0 -0.5"), (GTK_DOWN, "0.0 0.5"),
    (2555904, "0.5 0.0"),    # Windows
    (63235, "0.5 0.0"),      # macOS
    (16777236, "0.5 0.0"),   # Qt
])
def test_arrows_nudge_text_by_half_a_pixel(code, expected):
    c = TextController("")
    command = c.handle_key(code)
    assert command.name == "nudge"
    assert [float(v) for v in command.arg.split()] == [float(v) for v in expected.split()]


def test_arrows_still_nudge_while_typing_without_touching_the_buffer():
    c = TextController("abc")
    c.handle_key(ord("t"))
    assert c.handle_key(GTK_RIGHT) == Command("nudge", "0.5 0.0")
    assert c.buffer == "abc" and c.editing


@pytest.mark.parametrize("key,arg", [("+", "+1"), ("=", "+1"), ("-", "-1")])
def test_plus_minus_change_font_size(key, arg):
    assert TextController("").handle_key(ord(key)) == Command("fontsize", arg)


def test_zero_recenters():
    assert TextController("").handle_key(ord("0")) == Command("recenter")


def test_plus_minus_and_zero_are_typed_while_editing():
    c = TextController("")
    c.handle_key(ord("t"))
    for ch in "+-0":
        assert c.handle_key(ord(ch)) is None
    assert c.buffer == "+-0"


def test_terminal_layout_commands():
    c = TextController("")
    for line in ("/tracking 1.5", "/weight -0.5", "/nudge 0.5 -1", "/recenter"):
        c.handle_line(line)
    assert c.pop_commands() == [Command("tracking", "1.5"), Command("weight", "-0.5"),
                                Command("nudge", "0.5 -1"), Command("recenter")]


@pytest.mark.parametrize("key,arg", [("[", "-0.25"), ("]", "+0.25")])
def test_brackets_adjust_tracking(key, arg):
    assert TextController("").handle_key(ord(key)) == Command("tracking_delta", arg)


def test_brackets_are_typed_while_editing():
    c = TextController("")
    c.handle_key(ord("t"))
    for ch in "[]":
        assert c.handle_key(ord(ch)) is None
    assert c.buffer == "[]"
