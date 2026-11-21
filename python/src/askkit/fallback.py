import time
from collections.abc import Callable
from dataclasses import dataclass

import anthropic
import openai
from anthropic.types import (
    ContentBlock,
    Message,
    StopReason,
    TextBlock,
    Usage,
)
from openai import OpenAI
from openai.types.responses import Response

from askkit.retry import RETRY_STATUSES, worth_retrying

# The models AskKit can fall back to at another provider: OpenAI's, sent
# through its Responses API with OpenAI's own SDK.
OPENAI_MODELS: tuple[str, ...] = ("gpt-6-luna",)

# Seconds a model that used up its tries is passed over in a run, so the
# questions after it do not each wait up to two minutes for a model that
# has just failed. After that it is asked again first.
COOL_DOWN = 60.0

# Refusal categories another model is not meant to answer. Of
# reasoning_extraction Anthropic's "Refusals and fallback" page says:
# "This category has no recommended fallback model, so change the prompt
# rather than retry the request" (checked on 2026-11-20).
NO_FALLBACK_CATEGORIES = ("reasoning_extraction",)

ProviderError = anthropic.APIError | openai.APIError


@dataclass(frozen=True)
class Handover:
    # One model a question left for the next one: which, and why ("529",
    # "timeout", "no connection", "refusal", or "cooling down").
    model: str
    reason: str


@dataclass
class FirstAnswer:
    # The response that answers, the model of the chain that gave it,
    # every model the question left on the way, and the declined
    # responses before it, which may be billed too.
    response: Message
    model: str
    handovers: list[Handover]
    declined: list[tuple[Message, str]]


def is_claude(model: str) -> bool:
    return model.startswith("claude-")


def falls_back(error: ProviderError) -> bool:
    # Only what another model can fix: too busy, too much too fast, the
    # provider's side failed, or no answer at all. A request the API
    # rejected (400, 413) or a key it refused (401, 403) fails the same
    # way on the next model, and falling back would only hide it.
    if isinstance(error, anthropic.APIError):
        return worth_retrying(error)
    if isinstance(error, openai.APIStatusError):
        status = error.status_code
        return status in RETRY_STATUSES or status >= 500
    return isinstance(error, openai.APIConnectionError)


def failure(error: ProviderError) -> str:
    # The reason a model failed, in the words of the Retries line.
    if isinstance(error, anthropic.APIStatusError | openai.APIStatusError):
        return str(error.status_code)
    if isinstance(error, anthropic.APITimeoutError | openai.APITimeoutError):
        return "timeout"
    return "no connection"


def declined_onward(response: Message, next_model: str) -> bool:
    # A declined answer goes on only to another Claude model, as
    # Anthropic's page advises: "You can usually still get an answer by
    # sending the same request to another Claude model." A decline is a
    # judgement about the request, not an outage, and sending it to
    # another company to get round it fixes nothing.
    if response.stop_reason != "refusal" or not is_claude(next_model):
        return False
    details = response.stop_details
    if details is not None and details.category in NO_FALLBACK_CATEGORIES:
        return False
    return True


def first_answer(
    chain: list[str],
    send: Callable[[str], Message],
    down_until: dict[str, float],
    clock: Callable[[], float] = time.monotonic,
    cool_down: float = COOL_DOWN,
) -> FirstAnswer:
    # Sends the question to the models of the chain in order until one
    # answers. A model hands the question on when send() raises an error
    # another model can fix, which send() raises only once its own tries
    # are used up, or when it declines and the next model is Claude. A
    # model that failed is passed over until down_until says it may be
    # asked again; the last model is always asked. The last model's
    # error is raised as it is.
    handovers: list[Handover] = []
    declined: list[tuple[Message, str]] = []
    last = len(chain) - 1
    for position, model in enumerate(chain):
        if position < last and clock() < down_until.get(model, 0.0):
            handovers.append(Handover(model, "cooling down"))
            continue
        try:
            response = send(model)
        except (anthropic.APIError, openai.APIError) as error:
            if position == last or not falls_back(error):
                raise
            down_until[model] = clock() + cool_down
            handovers.append(Handover(model, failure(error)))
            continue
        if position < last and declined_onward(response, chain[position + 1]):
            declined.append((response, model))
            handovers.append(Handover(model, "refusal"))
            continue
        return FirstAnswer(response, model, handovers, declined)
    # Not reached: the last model either returns or raises.
    raise AssertionError("the chain is empty")


def stop_reason_of(response: Response) -> StopReason:
    # OpenAI's ways an answer ends, in Claude's words. A refusal is a
    # part of the answer's message; a cut-off answer is "incomplete"
    # with the reason max_output_tokens; content_filter means OpenAI's
    # filter stopped the answer, which AskKit treats as declined.
    for item in response.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "refusal":
                    return "refusal"
    if response.status == "completed":
        return "end_turn"
    details = response.incomplete_details
    if response.status == "incomplete" and details is not None:
        if details.reason == "max_output_tokens":
            return "max_tokens"
        if details.reason == "content_filter":
            return "refusal"
    raise ValueError(
        f"{response.model} stopped as {response.status}, which AskKit"
        " cannot read as an answer"
    )


def as_message(response: Response) -> Message:
    # OpenAI's response in the shape of Claude's, so every check and
    # every price in AskKit reads it as it reads Claude's. OpenAI's
    # input_tokens counts the whole prompt, cache reads and writes
    # included; Claude's counts only what was neither.
    stop_reason = stop_reason_of(response)
    usage = response.usage
    if usage is None:
        raise ValueError(
            f"{response.model} sent no token counts, so AskKit cannot"
            " price its answer"
        )
    read = usage.input_tokens_details.cached_tokens
    written = usage.input_tokens_details.cache_write_tokens
    content: list[ContentBlock] = []
    # A declined answer keeps no text, as Claude's does.
    if stop_reason != "refusal" and response.output_text:
        content.append(TextBlock(type="text", text=response.output_text))
    return Message(
        id=response.id,
        type="message",
        role="assistant",
        model=response.model,
        content=content,
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(
            input_tokens=usage.input_tokens - read - written,
            cache_read_input_tokens=read,
            cache_creation_input_tokens=written,
            output_tokens=usage.output_tokens,
        ),
    )


def ask_openai(
    client: OpenAI, question: str, model: str, max_tokens: int, system: str
) -> Message:
    # The same question as ask() sends Claude: the system prompt as
    # instructions, the question as the input.
    response = client.responses.create(
        model=model,
        instructions=system,
        input=question,
        max_output_tokens=max_tokens,
        # OpenAI keeps every response for 30 days unless told not to;
        # AskKit saves the answers it needs itself.
        store=False,
    )
    message = as_message(response)
    # The ID OpenAI gave the request, which its support asks for: the
    # call log keeps it, as it keeps Claude's (lesson 10.8).
    message._request_id = response._request_id
    return message


def fallbacks_line(answered: list[tuple[str, FirstAnswer]]) -> str | None:
    # One line for the run: how many questions another model answered,
    # and why the models before it handed them on.
    moved: list[str] = []
    for case, outcome in answered:
        if outcome.handovers:
            reasons = ", ".join(
                handover.reason for handover in outcome.handovers
            )
            moved.append(f"{case}: {reasons}")
    if not moved:
        return None
    return f"Fallbacks:     {len(moved)} ({'; '.join(moved)})"
