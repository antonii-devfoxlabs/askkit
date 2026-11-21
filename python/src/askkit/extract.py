from typing import Any, Literal, get_args

from anthropic import Anthropic, transform_schema
from anthropic.types import Message, MessageParam
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from askkit.prompts import render
from askkit.redact import redacted

# The three wishes the warehouse acts on. One list, read by the type and
# by the repair, so the two cannot disagree on what is allowed.
Wish = Literal["replacement", "refund", "not stated"]
WISHES: tuple[str, ...] = get_args(Wish)

# Where a repair writes what it changed, in the validation context: a
# dict the caller passes to model_validate_json(text, context=...), and
# a structured-output library passes on to the class (Instructor's
# context=, Pydantic AI's validation_context=).
REPAIRED = "repaired"


def in_lower_case(
    field: str, value: Any, allowed: tuple[str, ...], info: ValidationInfo
) -> Any:
    # Repairs only what has one right reading: an allowed value written
    # with other capitals, which Anthropic's documentation says the API
    # does not hold ("not Stated" for "not stated"). Anything else is
    # left for the type check to refuse: a question about the email, for
    # the model to answer again. It runs inside the class, before the
    # type check, so every loop that checks the class gets it, AskKit's
    # own or a library's, and no call is paid for a fault code can fix.
    if not isinstance(value, str) or value in allowed:
        return value
    if value.lower() not in allowed:
        return value
    # No context: validated by code that keeps no notes, such as a test
    # or a library called without one. The repair still happens.
    if isinstance(info.context, dict):
        changes = info.context.setdefault(REPAIRED, [])
        changes.append(f"{field} {value!r} to {value.lower()!r}")
    return value.lower()


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

    # mode="before": the value as the answer gave it, before the type
    # check, so "Refund" can still become "refund" and pass.
    @field_validator("wish", mode="before")
    @classmethod
    def wish_in_lower_case(cls, wish: Any, info: ValidationInfo) -> Any:
        return in_lower_case("wish", wish, WISHES, info)


# What Ticket checked, for a ticket's label (label.py).
TICKET_CHECK = "types and allowed values"

# The schema the API holds the answer to, made from Ticket by the SDK's
# own helper, the one messages.parse() uses.
TICKET_SCHEMA: dict[str, Any] = transform_schema(Ticket)


def ticket_messages(
    email: str, answer: str | None = None, faults: str | None = None
) -> list[MessageParam]:
    # The email alone, or, to ask again, the email, the failed answer and
    # its faults, so the model knows what to change. A retry ends on the
    # user turn: an assistant turn last is a prefill, which the API refuses.
    # A ticket holds no card, email address or phone, so they are taken
    # out of the email before it is sent.
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


def check_ticket(text: str) -> tuple[Ticket, str | None]:
    # The answer as a Ticket, and what the class repaired, if anything.
    # The repair is one of Ticket's own checks, so the rest of the answer
    # is checked with it in the same pass.
    notes: dict[str, list[str]] = {}
    ticket = Ticket.model_validate_json(text, context=notes)
    return ticket, repairs(notes)


def repairs(notes: dict[str, list[str]]) -> str | None:
    # The changes the repairs wrote into the context, as one note.
    changes = notes.get(REPAIRED, [])
    if not changes:
        return None
    return "; ".join(changes)


def fault_place(loc: tuple[str | int, ...]) -> str:
    # Where a fault is: its parts joined with dots, so the amount of the
    # second row of an invoice is "lines.1.amount" (rows count from 0).
    # An empty place means the whole answer, such as the JSON list [].
    if len(loc) == 0:
        return "answer"
    parts: list[str] = []
    for part in loc:
        parts.append(str(part))
    return ".".join(parts)


def ticket_faults(error: ValidationError) -> str:
    # One part per fault, so the message says what came back as well as
    # what the model class allows. Any ValidationError reads the same
    # way, a Ticket's, a Classification's or an Invoice's.
    faults: list[str] = []
    for fault in error.errors():
        place = fault_place(fault["loc"])
        if fault["type"] == "value_error":
            # A validator of ours raised ValueError: its own words name
            # the values, so they are shown as written, without
            # Pydantic's "Value error, " or the whole input after them.
            faults.append(f"{place}: {fault['ctx']['error']}")
            continue
        line = f"{place}: {fault['msg']}"
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
