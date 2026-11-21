import json

import pytest

from askkit.terminal import printable

# ESC [ 2 J clears the screen; ESC [ 1 A moves the cursor up a line.
CLEAR_SCREEN = "\x1b[2J"
LINE_UP = "\x1b[1A"
# ESC ] 8 ;; URL ESC \ starts a link that shows other words than its URL.
LINK = "\x1b]8;;https://example.com/?notes=4417\x1b\\docs.python.org"


def test_plain_text_is_printed_as_it_is() -> None:
    text = "Order 4417: a replacement mug, 12 € and 👍."
    assert printable(text) == text


def test_line_breaks_and_tabs_keep_the_text_in_shape() -> None:
    assert printable("Model:\tHaiku\nCost:\t$0.01") == (
        "Model:\tHaiku\nCost:\t$0.01"
    )


def test_esc_is_shown_as_its_escape() -> None:
    assert printable(f"Done.{CLEAR_SCREEN}") == "Done.\\u001b[2J"


def test_no_esc_is_left_for_the_terminal() -> None:
    shown = printable(f"{LINE_UP}Cost: $0.00{LINK}")
    assert "\x1b" not in shown
    assert shown.count("\\u001b") == 3


@pytest.mark.parametrize(
    ("character", "escape"),
    [
        ("\r", "\\r"),
        ("\b", "\\b"),
        ("\x07", "\\u0007"),
        ("\x7f", "\\u007f"),
        ("\x9b", "\\u009b"),
        ("‮", "\\u202e"),
    ],
    ids=["return", "backspace", "bell", "delete", "csi", "right-to-left"],
)
def test_every_character_a_terminal_obeys_is_shown(
    character: str, escape: str
) -> None:
    assert printable(f"a{character}b") == f"a{escape}b"


def test_a_hidden_tag_character_is_shown() -> None:
    # U+E0041, an invisible "tag" letter A, which can hide words.
    assert "\U000e0041" not in printable("Hi\U000e0041")


def test_printed_json_stays_json() -> None:
    # Every escape is one JSON reads, so a printed ticket still parses.
    text = json.dumps({"summary": "Fine\x9b‮."}, ensure_ascii=False)
    assert json.loads(printable(text)) == {"summary": "Fine\x9b‮."}
