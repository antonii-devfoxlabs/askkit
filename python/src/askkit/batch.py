import re
from typing import cast

from anthropic.types import Message
from anthropic.types.message_create_params import (
    MessageCreateParamsNonStreaming,
)
from anthropic.types.messages import MessageBatch
from anthropic.types.messages.batch_create_params import Request

from askkit import calllog
from askkit.classify import CLASSIFICATION_SCHEMA, sorting_messages
from askkit.clean import clean_email
from askkit.cost import usage_cost
from askkit.inbox import InboxEmail
from askkit.prices import BATCH_SHARE
from askkit.structured import json_request

# What one Message Batch may hold, from Anthropic's batch processing
# page, checked on 2026-11-20: 100,000 requests (or 256 MB, which the
# API checks itself, with a 413).
MOST_REQUESTS = 100_000

# A request's custom_id names it in the results, which come back in any
# order. The API takes 1 to 64 letters, digits, hyphens and underscores
# (^[a-zA-Z0-9_-]{1,64}$ on the same page). An email's id is used as it
# is, so a result names its email without a lookup.
CUSTOM_ID = re.compile(r"[a-zA-Z0-9_-]{1,64}")

# How often askkit batch --wait asks how the batch is doing. Most
# batches end within an hour and a batch may take 24, so once a minute
# is plenty; the Batches API has a limit of requests a minute of its own
# (1,000 on the Start tier, checked on 2026-11-20).
POLL_SECONDS = 60

# The batches whose answers are already in the call log, one ID a line,
# in the log's folder: collecting a batch twice logs it once.
COLLECTED = "batches.txt"


def id_fault(emails: list[InboxEmail]) -> str | None:
    # Checked before anything is sent: the API would reject the whole
    # batch for one id it cannot take, or one used twice.
    if len(emails) > MOST_REQUESTS:
        return (
            f"{len(emails):,} emails are more than the {MOST_REQUESTS:,}"
            " one batch may hold; send them with --first in parts"
        )
    seen: set[str] = set()
    for email in emails:
        if not CUSTOM_ID.fullmatch(email.id):
            return (
                f"the id {email.id!r} cannot name a batch request; an id"
                " is 1 to 64 letters, digits, - or _"
            )
        if email.id in seen:
            return f"the id {email.id} is in the file twice"
        seen.add(email.id)
    return None


def email_request(
    email: InboxEmail, model: str, max_tokens: int, system: str
) -> Request:
    # The request askkit inbox sends for one email, as a batch request:
    # the same params, named by the email's id.
    cleaned, _ = clean_email(email.email)
    params = json_request(
        model,
        max_tokens,
        system,
        sorting_messages(cleaned),
        CLASSIFICATION_SCHEMA,
    )
    # json_request returns a plain dict, the one stream_json and ask_json
    # send; cast tells mypy it has the shape of the SDK's params.
    return Request(
        custom_id=email.id,
        params=cast(MessageCreateParamsNonStreaming, params),
    )


def batch_cost(model: str, response: Message) -> float:
    # One batch request's answer, at the batch price.
    return usage_cost(model, response.usage) * BATCH_SHARE


def status_text(batch: MessageBatch) -> str:
    # in_progress, canceling or ended; until it ends, how many requests
    # are still waiting to be answered.
    if batch.processing_status == "ended":
        return "ended"
    counts = batch.request_counts
    total = (
        counts.processing
        + counts.succeeded
        + counts.errored
        + counts.canceled
        + counts.expired
    )
    return (
        f"{batch.processing_status}, {counts.processing:,} of {total:,}"
        " still processing"
    )


def batch_seconds(batch: MessageBatch) -> float:
    # From sending the batch to its last answer: how long each answer
    # in it took to come back.
    if batch.ended_at is None:
        raise ValueError(f"{batch.id} has not ended")
    return (batch.ended_at - batch.created_at).total_seconds()


def collected_before(batch_id: str) -> bool:
    if calllog.current is None:
        return False
    path = calllog.current.folder / COLLECTED
    if not path.exists():
        return False
    return batch_id in path.read_text(encoding="utf-8").split()


def log_batch(
    batch: MessageBatch, answers: list[tuple[str, Message]]
) -> None:
    # Each answer is one call in the log, priced at the batch price, its
    # seconds the batch's own. A batch request has no request ID of its
    # own; its custom_id and the batch's ID name it instead. A batch
    # already collected is not logged again: it was paid for once.
    if calllog.current is None or collected_before(batch.id):
        return
    seconds = batch_seconds(batch)
    for custom_id, response in answers:
        call = calllog.logged_call(
            calllog.current.command,
            response,
            response.model,
            seconds,
            0,
            None,
            calllog.clock(),
        )
        cost = round(batch_cost(response.model, response), 10)
        call = call.model_copy(update={"cost": cost})
        calllog.write_call(
            call,
            f"[request {custom_id} of batch {batch.id}]",
            calllog.text_of(response.content),
        )
    calllog.append_line(calllog.current.folder / COLLECTED, batch.id)
