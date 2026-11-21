import json
from datetime import UTC, datetime
from typing import Any

from askkit.label import LABEL_KEY, ai_label, labelled_json

WRITTEN = datetime(2026, 11, 21, 9, 30, tzinfo=UTC)
TICKET = {"order": 4417, "problem": "mug broken", "wish": "replacement"}


def printed_json(out: str) -> tuple[dict[str, Any], list[str]]:
    # What extract and read print: a JSON object, its last line a lone
    # "}", then the lines about the call. The object, and those lines.
    lines = out.splitlines()
    end = lines.index("}")
    data = json.loads("\n".join(lines[: end + 1]))
    assert isinstance(data, dict)
    return data, lines[end + 1 :]


def test_a_label_says_what_wrote_it_when_and_what_was_checked() -> None:
    label = ai_label("claude-haiku-5-5", "extract", WRITTEN, ["a check"])
    assert label == {
        "written_by": "claude-haiku-5-5",
        "via": "askkit extract",
        "written": "2026-11-21T09:30:00Z",
        "checked_by_code": ["a check"],
        "read_by_a_person": False,
    }


def test_the_label_comes_after_the_fields_inside_the_json() -> None:
    label = ai_label("claude-haiku-5-5", "extract", WRITTEN, [])
    text = labelled_json(TICKET, label)
    data = json.loads(text)
    assert list(data) == ["order", "problem", "wish", LABEL_KEY]
    assert data[LABEL_KEY] == label
    assert text.startswith('{\n  "order": 4417,\n')


def test_the_fields_given_get_no_label() -> None:
    labelled_json(TICKET, ai_label("claude-haiku-5-5", "read", WRITTEN, []))
    assert LABEL_KEY not in TICKET


def test_text_beyond_ascii_stays_as_written() -> None:
    fields = {"seller": "Café Mira"}
    label = ai_label("claude-haiku-5-5", "read", WRITTEN, [])
    assert '"seller": "Café Mira"' in labelled_json(fields, label)
