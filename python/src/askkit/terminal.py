import json

# A line break and a tab only move to a new line or the next column, so
# the text keeps its shape.
KEPT = ("\n", "\t")


def printable(text: str) -> str:
    # Text a model wrote, ready for the terminal. A terminal obeys some
    # characters instead of printing them: ESC starts a sequence that can
    # move the cursor, clear the screen or turn words into a link, and \r
    # goes back to the start of the line to write over it. Each character
    # Python counts as not printable is shown as the escape JSON writes
    # for it, such as \u001b for ESC, so you see it and nothing obeys it.
    shown: list[str] = []
    for character in text:
        if character.isprintable() or character in KEPT:
            shown.append(character)
        else:
            shown.append(json.dumps(character)[1:-1])
    return "".join(shown)
