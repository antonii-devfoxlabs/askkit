import json
from datetime import datetime
from typing import Any

# The key AskKit adds to the JSON it writes for other people to act on: a
# ticket for the warehouse, an invoice for the books. Inside the JSON, so
# the label goes wherever the JSON goes; the lines AskKit prints under it
# stay in the terminal.
LABEL_KEY = "ai_label"


def ai_label(
    model: str, command: str, written: datetime, checked: list[str]
) -> dict[str, Any]:
    # What is true when AskKit writes the JSON, and nothing more: a model
    # wrote it, which one, through which command, when, and what AskKit's
    # code checked. No person has read it yet, so the label says so; a
    # tool that lets a person edit it changes that field, never the model.
    return {
        "model": model,
        "via": f"askkit {command}",
        "written": written.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checked_by_code": checked,
        "read_by_a_person": False,
    }


def labelled_json(fields: dict[str, Any], label: dict[str, Any]) -> str:
    # The fields as they were, with the label after them, as AskKit prints
    # JSON. A copy: the dict the caller holds gets no label.
    data = dict(fields)
    data[LABEL_KEY] = label
    return json.dumps(data, indent=2, ensure_ascii=False)
