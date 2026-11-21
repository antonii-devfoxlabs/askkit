import json
from typing import Any, Literal, get_args

from anthropic import transform_schema
from anthropic.types import MessageParam
from pydantic import BaseModel, ConfigDict, ValidationError

from askkit.extract import ticket_faults
from askkit.untrusted import marked

# The queues of the shop's support inbox, one per team. "other" is the
# queue for an email that meets none of the first three conditions:
# without it, the schema would make the model put such an email in a
# queue it does not belong to. One list, read by Classification and by
# repair_queue, and the template defines each queue in the same words.
Queue = Literal["refund", "damaged", "delivery", "other"]
QUEUES: tuple[str, ...] = get_args(Queue)


# An email's queue. An object with one field, as Ticket is, so the same
# checks apply to it and a field can be added later without a new shape.
class Classification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queue: Queue


CLASSIFICATION_SCHEMA: dict[str, Any] = transform_schema(Classification)

# What the JSON object around an email says it is: the classify_email
# prompt tells the model what that means.
EMAIL_SOURCE = "customer email"


def sorting_messages(email: str) -> list[MessageParam]:
    # The one message of every sorting call, from classify, inbox and
    # batch alike: the cleaned email, marked as text a stranger wrote, so
    # a closing tag or a "new rule" typed into it stays inside it.
    return [{"role": "user", "content": marked(EMAIL_SOURCE, email)}]


def repair_queue(text: str) -> tuple[str, str] | None:
    # The capitals of an allowed queue, which the API does not hold
    # ("Other" for "other"): the one fault with one right reading.
    # Returns the repaired answer and what changed, or None.
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    queue = data.get("queue")
    if not isinstance(queue, str) or queue in QUEUES:
        return None
    if queue.lower() not in QUEUES:
        return None
    data["queue"] = queue.lower()
    return json.dumps(data), f"queue {queue!r} to {queue.lower()!r}"


def check_classification(text: str) -> tuple[Classification, str | None]:
    # The answer as a Classification, and what repair_queue changed, if
    # anything. With the schema held, the queue is one of four words and
    # only its capitals can slip, so AskKit does not ask again: an answer
    # that still fails goes to a person.
    try:
        return Classification.model_validate_json(text), None
    except ValidationError:
        repaired = repair_queue(text)
        if repaired is None:
            raise
    repaired_text, change = repaired
    return Classification.model_validate_json(repaired_text), change


def classification_error(error: ValidationError) -> str:
    # ticket_faults reads any ValidationError, one part per field.
    return "the queue does not match its model: " + ticket_faults(error)
