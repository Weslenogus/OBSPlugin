import io
import time

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
