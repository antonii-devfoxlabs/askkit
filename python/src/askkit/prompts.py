from pathlib import Path
from string import Template

# Every prompt AskKit sends is a file in templates/, named
# <name>.v<version>.txt. A version that has been sent is never edited: a
# change is a new file with the next number, and templates/CHANGES.md
# says what changed and why. So an answer can be traced to the exact
# words that made it, and two versions can be run on the same questions.
TEMPLATES = Path(__file__).parent / "templates"

# The version of each prompt AskKit sends now.
IN_USE = {
    "ask_system": 2,
    "quote_request": 2,
    "extract_ticket": 2,
    "retry_ticket": 1,
    "classify_email": 1,
    "summarise_turns": 1,
    "fold_turns": 1,
    "earlier_turns": 1,
    "remembered": 1,
    "read_invoice": 1,
}


def template_path(name: str, version: int) -> Path:
    return TEMPLATES / f"{name}.v{version}.txt"


def render(
    name: str, values: dict[str, str], version: int | None = None
) -> str:
    if version is None:
        version = IN_USE[name]
    text = template_path(name, version).read_text(encoding="utf-8")
    template = Template(text.strip())
    # substitute() fails on a placeholder with no value, but drops a value
    # with no placeholder without a word: compare the names both ways.
    needed = sorted(template.get_identifiers())
    given = sorted(values)
    if needed != given:
        raise ValueError(
            f"{name} v{version} needs {needed}, but was given {given}"
        )
    return template.substitute(values)


ASK_SYSTEM = render("ask_system", {})
