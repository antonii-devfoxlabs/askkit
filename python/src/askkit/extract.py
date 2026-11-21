import json
from typing import Any, Literal, get_args

from anthropic import Anthropic, transform_schema
from anthropic.types import Message, MessageParam
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from askkit.prompts import render
from askkit.redact import redacted

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
    # The prompt's "two or three words" as a length code can check. Only
    # this class holds it: the API cannot hold a length.
    problem: str = Field(max_length=40)
    wish: Wish


# The schema the API holds the answer to, made from Ticket by the SDK's
# own helper, the one messages.parse() uses.
TICKET_SCHEMA: dict[str, Any] = transform_schema(Ticket)


def ticket_messages(
    email: str, answer: str | None = None, faults: str | None = None
) -> list[MessageParam]:
    # The email alone, or, to ask again, the email, the failed answer and
    # its faults, so the model knows what to change. A retry ends on the
    # user turn: an assistant turn last is a prefill, which the API refuses.
    # A ticket holds no card, address or phone, so they are taken out of
    # the email before it is sent.
    messages: list[MessageParam] = [
        {"role": "user", "content": redacted(email)}
    ]
    if answer is not None and faults is not None:
        messages.append({"role": "assistant", "content": answer})
        messages.append(
            {
                "role": "user",
                "content": render("retry_ticket", {"faults": faults}),
            }
        )
    return messages


def extract_ticket(
    client: Anthropic,
    email: str,
    model: str,
    max_tokens: int,
    version: int | None = None,
    answer: str | None = None,
    faults: str | None = None,
) -> Message:
    # The eval's call, not streamed: nobody watches it. version picks the
    # prompt file; None sends the one in IN_USE. The eval passes another
    # one to score a version before it is put in use. With a failed
    # answer and its faults, it is the retry, as extract asks it.
    return client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=render("extract_ticket", {}, version),
        messages=ticket_messages(email, answer, faults),
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
    # inside fields, so a fault's place ("loc") is one field name, or
    # empty when the whole answer is not a ticket.
    faults: list[str] = []
    for fault in error.errors():
        field = "answer"
        if fault["loc"]:
            field = str(fault["loc"][0])
        line = f"{field}: {fault['msg']}"
        if fault["type"] != "missing":
            line += f", not {fault['input']!r}"
        faults.append(line)
    return "; ".join(faults)


def ticket_error(error: ValidationError) -> str:
    return "the ticket does not match its model: " + ticket_faults(error)


def not_stated(ticket: Ticket) -> list[str]:
    # The fields that hold a way out: the email does not give them. The
    # ticket is right as it is, but whoever acts on it has to ask the
    # customer for these, so AskKit names them instead of leaving a null
    # for someone to notice.
    fields: list[str] = []
    if ticket.order is None:
        fields.append("order")
    if ticket.wish == "not stated":
        fields.append("wish")
    return fields
