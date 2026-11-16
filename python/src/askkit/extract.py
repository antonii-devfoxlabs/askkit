import json
from typing import Any, Literal, get_args

from anthropic import Anthropic, transform_schema
from anthropic.types import Message
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from askkit.prompts import render

# The three wishes the warehouse acts on. One list, read by Ticket and by
# repair_ticket, so the two cannot disagree on what is allowed.
Wish = Literal["replacement", "refund", "not stated"]
WISHES: tuple[str, ...] = get_args(Wish)


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
    # "Two or three words" in the prompt, held here as a length the code
    # can check. The API cannot hold a length: transform_schema moves it
    # into the field's description, so only this class enforces it.
    problem: str = Field(max_length=40)
    wish: Wish


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


def retry_ticket(
    client: Anthropic,
    email: str,
    answer: str,
    faults: str,
    model: str,
    max_tokens: int,
) -> Message:
    # The same request, plus the answer that failed and what was wrong
    # with it, so the model knows what to change. The conversation ends
    # with a user message: an assistant message last is a prefill, which
    # the API refuses.
    return client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=render("extract_ticket", {}),
        messages=[
            {"role": "user", "content": email},
            {"role": "assistant", "content": answer},
            {
                "role": "user",
                "content": render("retry_ticket", {"faults": faults}),
            },
        ],
        output_config={
            "format": {"type": "json_schema", "schema": TICKET_SCHEMA}
        },
    )


def repair_ticket(text: str) -> tuple[str, str] | None:
    # Repairs only what has one right reading: an allowed wish written with
    # other capitals, which Anthropic's documentation says the API does
    # not hold ("not Stated" for "not stated"). Anything else is a
    # question about the email, and only the model can answer it again.
    # Returns the repaired answer and what changed, or None.
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    wish = data.get("wish")
    if not isinstance(wish, str) or wish in WISHES:
        return None
    if wish.lower() not in WISHES:
        return None
    data["wish"] = wish.lower()
    return json.dumps(data), f"wish {wish!r} to {wish.lower()!r}"


def check_ticket(text: str) -> tuple[Ticket, str | None]:
    # The answer as a Ticket, and what repair_ticket changed, if anything.
    # A repaired answer is checked again in full: the repair touches one
    # field, and the others still have to pass.
    try:
        return Ticket.model_validate_json(text), None
    except ValidationError:
        repaired = repair_ticket(text)
        if repaired is None:
            raise
    repaired_text, change = repaired
    return Ticket.model_validate_json(repaired_text), change


def ticket_faults(error: ValidationError) -> str:
    # One part per field that broke the contract, so the message says
    # what came back as well as what Ticket allows. Ticket has no fields
    # inside fields, so a problem's place ("loc") is one field name, or
    # empty when the whole answer is not a ticket.
    faults: list[str] = []
    for problem in error.errors():
        field = "answer"
        if problem["loc"]:
            field = str(problem["loc"][0])
        line = f"{field}: {problem['msg']}"
        if problem["type"] != "missing":
            line += f", not {problem['input']!r}"
        faults.append(line)
    return "; ".join(faults)


def ticket_error(error: ValidationError) -> str:
    return "the ticket does not match its model: " + ticket_faults(error)
