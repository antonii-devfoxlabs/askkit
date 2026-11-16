from typing import Any, Literal

from anthropic import Anthropic, transform_schema
from anthropic.types import Message
from pydantic import BaseModel, ConfigDict, ValidationError

from askkit.prompts import render


# A support ticket, as the warehouse team reads it. One class is the
# contract on both sides: its JSON schema goes with the request, and
# every answer is checked against it again in our own code. Each field
# has a way out (None, "not stated"), because a schema with no way out
# makes the model invent a value to fill it.
class Ticket(BaseModel):
    # Nothing extra: an answer with a field the class does not name is
    # rejected, and the schema says "additionalProperties": false.
    model_config = ConfigDict(extra="forbid")

    order: int | None
    problem: str
    wish: Literal["replacement", "refund", "not stated"]


# The schema the API holds the answer to, made from Ticket by the SDK's
# own helper, the one messages.parse() uses.
TICKET_SCHEMA: dict[str, Any] = transform_schema(Ticket)


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


def ticket_error(error: ValidationError) -> str:
    # One line per field that broke the contract, so the message says
    # what came back as well as what Ticket allows. Ticket has no fields
    # inside fields, so a problem's place ("loc") is one field name, or
    # empty when the whole answer is not a ticket.
    problems: list[str] = []
    for problem in error.errors():
        field = "answer"
        if problem["loc"]:
            field = str(problem["loc"][0])
        line = f"{field}: {problem['msg']}"
        if problem["type"] != "missing":
            line += f", not {problem['input']!r}"
        problems.append(line)
    return "the ticket does not match its model: " + "; ".join(problems)
