from typing import Any, Literal, get_args

from anthropic import transform_schema
from anthropic.types import MessageParam
from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidationInfo,
    field_validator,
)

from askkit.extract import in_lower_case, repairs, ticket_faults
from askkit.redact import redacted
from askkit.untrusted import marked

# The queues of the shop's support inbox, one per team. "other" is the
# queue for an email that meets none of the first three conditions:
# without it, the schema would make the model put such an email in a
# queue it does not belong to. One list, read by the type and by the
# repair, and the template defines each queue in the same words.
Queue = Literal["refund", "damaged", "delivery", "other"]
QUEUES: tuple[str, ...] = get_args(Queue)


# An email's queue. An object with one field, as Ticket is, so the same
# checks apply to it and a field can be added later without a new shape.
class Classification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    queue: Queue

    # The capitals of an allowed queue ("Other" for "other"), repaired
    # inside the class, as Ticket repairs its wish.
    @field_validator("queue", mode="before")
    @classmethod
    def queue_in_lower_case(cls, queue: Any, info: ValidationInfo) -> Any:
        return in_lower_case("queue", queue, QUEUES, info)


CLASSIFICATION_SCHEMA: dict[str, Any] = transform_schema(Classification)

# What the JSON object around an email says it is: the classify_email
# prompt tells the model what that means.
EMAIL_SOURCE = "customer email"


def sorting_messages(email: str) -> list[MessageParam]:
    # The one message of every sorting call, from classify, inbox and
    # batch alike: the cleaned email, marked as text a stranger wrote, so
    # a closing tag or a "new rule" typed into it stays inside it. Its
    # card numbers, email addresses and phone numbers are taken out
    # first: a queue never depends on them, so they need not leave.
    content = marked(EMAIL_SOURCE, redacted(email))
    return [{"role": "user", "content": content}]


def check_classification(text: str) -> tuple[Classification, str | None]:
    # The answer as a Classification, and what the class repaired, if
    # anything. With the schema held, the queue is one of four words and
    # only its capitals can slip, so AskKit does not ask again: an answer
    # that still fails goes to a person.
    notes: dict[str, list[str]] = {}
    classification = Classification.model_validate_json(text, context=notes)
    return classification, repairs(notes)


def classification_error(error: ValidationError) -> str:
    # ticket_faults reads any ValidationError, one part per field.
    return "the queue does not match its model: " + ticket_faults(error)
