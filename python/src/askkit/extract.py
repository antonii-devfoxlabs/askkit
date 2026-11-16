from typing import Any

from anthropic import Anthropic
from anthropic.types import Message

from askkit.prompts import render

# The shape of a ticket in JSON Schema. The API holds the answer to it
# while the model writes, so the JSON always parses and has these three
# fields. Each field has a way out ("null", "not stated"), because a
# schema with no way out makes the model invent a value to fill it.
TICKET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "order": {"type": ["integer", "null"]},
        "problem": {"type": "string"},
        "wish": {"enum": ["replacement", "refund", "not stated"]},
    },
    "required": ["order", "problem", "wish"],
    "additionalProperties": False,
}


def extract_ticket(
    client: Anthropic, email: str, model: str, max_tokens: int
) -> Message:
    return client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=render("extract_ticket", {}),
        messages=[{"role": "user", "content": email}],
        output_config={
            "format": {"type": "json_schema", "schema": TICKET_SCHEMA}
        },
    )
