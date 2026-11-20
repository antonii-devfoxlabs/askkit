from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal

from anthropic import Anthropic, Omit, omit
from anthropic.types import (
    ContentBlockParam,
    Message,
    MessageParam,
    OutputConfigParam,
    ToolChoiceParam,
    ToolUnionParam,
)

from askkit.caching import CACHE, cached_system
from askkit.calllog import log_call
from askkit.prices import PRICES
from askkit.server_tools import server_line

# Anthropic's SDK reaches only Anthropic's models.
MODELS: list[str] = []
for name in sorted(PRICES):
    if name.startswith("claude-"):
        MODELS.append(name)

# What a user message holds: plain text, or a list of blocks, such as
# images with a question, or the results of tool requests.
Content = str | Sequence[ContentBlockParam]

# The effort levels the API takes, from the least work to the most.
Effort = Literal["low", "medium", "high", "xhigh", "max"]
EFFORTS: list[str] = ["low", "medium", "high", "xhigh", "max"]


def ask(
    client: Anthropic,
    question: str,
    model: str,
    max_tokens: int,
    system: str,
    effort: Effort | None = None,
    history: list[MessageParam] | None = None,
) -> Message:
    # Not logged here: askkit answer calls this through its retries, and
    # only its caller, send_to, knows how many came before the answer.
    return client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages_for(question, history),
        output_config=effort_config(effort),
    )


@dataclass
class Timing:
    # Seconds from sending the request: to the first piece of text (None
    # when no text came at all) and to the end of the stream.
    first_token: float | None
    total: float


def stream_answer(
    client: Anthropic,
    question: Content | None,
    model: str,
    max_tokens: int,
    system: str,
    effort: Effort | None = None,
    history: list[MessageParam] | None = None,
    tools: Sequence[ToolUnionParam] | None = None,
    tool_choice: ToolChoiceParam | None = None,
    cache_conversation: bool = False,
) -> tuple[Message | None, Timing]:
    # The same request as ask(), printed piece by piece as the model
    # writes it, and timed from the moment it is sent. No Message when
    # you cancel it with Ctrl-C: the stream never reached its end. With
    # tools, the model may ask for one instead of answering; the question
    # of the next call is then the results. Without a tool_choice, the
    # model decides for itself whether to ask. The system prompt goes with
    # a cache breakpoint; cache_conversation adds one the API moves to the
    # end of the messages on every call, for a chat.
    first_token: float | None = None
    # mid_line: text was printed and its line not ended yet.
    mid_line = False
    # perf_counter: a clock for durations, which never goes back.
    sent = messages_for(question, history)
    start = perf_counter()
    try:
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=cached_system(system),
            messages=sent,
            output_config=effort_config(effort),
            cache_control=CACHE if cache_conversation else omit,
            tools=omit if tools is None else tools,
            tool_choice=omit if tool_choice is None else tool_choice,
        ) as stream:
            for event in stream:
                if event.type == "text":
                    if first_token is None and event.text:
                        first_token = perf_counter() - start
                    # flush: show the piece now, not when the line ends.
                    print(event.text, end="", flush=True)
                    mid_line = mid_line or bool(event.text)
                elif event.type == "content_block_stop":
                    # A search the API ran inside this reply gets a line
                    # of its own, between the text before it and after it.
                    line = server_line(event.content_block)
                    if line is not None:
                        if mid_line:
                            print()
                        print(line, flush=True)
                        mid_line = False
            response = stream.get_final_message()
            total = perf_counter() - start
            log_call(response, model, total, system, sent, stream.request_id)
            return response, Timing(first_token, total)
    except KeyboardInterrupt:
        # Ctrl-C while the answer streams. Leaving the with block closed
        # the connection, so no more of the answer comes.
        return None, Timing(first_token, perf_counter() - start)
    finally:
        # Ends the answer's last line, also when the stream breaks off,
        # so whatever AskKit prints next starts on a line of its own.
        if mid_line:
            print()


def timing_line(timing: Timing, cancelled: bool = False) -> str:
    total = f"total {timing.total:.2f} s"
    if cancelled:
        total = f"cancelled at {timing.total:.2f} s"
    if timing.first_token is None:
        return f"no text, {total}"
    return f"first token {timing.first_token:.2f} s, {total}"


def effort_config(effort: Effort | None) -> OutputConfigParam | Omit:
    # Without an effort, the field stays out and the model's default holds.
    if effort is None:
        return omit
    return {"effort": effort}


def messages_for(
    question: Content | None,
    history: list[MessageParam] | None = None,
) -> list[MessageParam]:
    # The API remembers nothing, so every earlier message goes again,
    # oldest first, before the new question. No question: the history
    # ends with a turn the model was in the middle of, and goes as it is.
    messages: list[MessageParam] = []
    if history is not None:
        messages = list(history)
    if question is not None:
        messages.append({"role": "user", "content": question})
    return messages


def answer_text(response: Message) -> str:
    # The model may think first, so the text is not always block 0.
    parts: list[str] = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
    return "".join(parts)


def thinking_tokens(response: Message) -> int | None:
    # How many output tokens were thinking, when the API sends the split.
    details = response.usage.output_tokens_details
    if details is None:
        return None
    return details.thinking_tokens


# The stop reasons of an answer that is whole.
FINISHED = ("end_turn", "stop_sequence")


def stop_warning(
    response: Message, task: str = "answer this question"
) -> str | None:
    # task says what the model was asked to do, so a declined email reads
    # as one: "declined to turn this email into a ticket".
    reason = response.stop_reason
    if reason in FINISHED:
        return None
    if reason == "max_tokens":
        return (
            f"the answer was cut off at {response.usage.output_tokens:,}"
            " tokens; give it a larger --max-tokens"
        )
    if reason == "model_context_window_exceeded":
        return "the answer was cut off at the end of the context window"
    if reason == "refusal":
        # No part of a declined answer is used. The person who picks it up
        # gets the category and the API's explanation, shown as they came:
        # the explanation's words may change, so code never reads them.
        warning = f"the model declined to {task}"
        details = response.stop_details
        if details is not None and details.category is not None:
            warning += f" (category: {details.category})"
        if details is not None and details.explanation is not None:
            warning += f". {details.explanation}"
        return warning
    return f"the answer stopped early: {reason}"


# Streamed, a declined answer may have shown some text before the model
# stopped. Anthropic advises discarding such a part, so AskKit says so.
DECLINED_PART = (
    "the text above stopped where the model declined; do not use it"
)


# Printed after an answer cancelled with Ctrl-C: its text stops midway,
# with no stop reason and no usage, so nothing may use it as an answer.
CANCELLED = "cancelled; the text above is not a whole answer"
