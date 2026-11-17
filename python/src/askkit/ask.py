from typing import Literal

from anthropic import Anthropic, Omit, omit
from anthropic.types import Message, MessageParam, OutputConfigParam

from askkit.prices import PRICES

# Anthropic's SDK reaches only Anthropic's models.
MODELS: list[str] = []
for name in sorted(PRICES):
    if name.startswith("claude-"):
        MODELS.append(name)

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
    return client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages_for(question, history),
        output_config=effort_config(effort),
    )


def stream_answer(
    client: Anthropic,
    question: str,
    model: str,
    max_tokens: int,
    system: str,
    effort: Effort | None = None,
    history: list[MessageParam] | None = None,
) -> Message:
    # The same request as ask(), printed piece by piece as the model
    # writes it.
    shown = False
    try:
        with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages_for(question, history),
            output_config=effort_config(effort),
        ) as stream:
            for text in stream.text_stream:
                # flush: show the piece now, not when the line ends.
                print(text, end="", flush=True)
                shown = True
            return stream.get_final_message()
    finally:
        # Ends the answer's last line, also when the stream breaks off,
        # so whatever AskKit prints next starts on a line of its own.
        if shown:
            print()


def effort_config(effort: Effort | None) -> OutputConfigParam | Omit:
    # Without an effort, the field stays out and the model's default holds.
    if effort is None:
        return omit
    return {"effort": effort}


def messages_for(
    question: str, history: list[MessageParam] | None = None
) -> list[MessageParam]:
    # The API remembers nothing, so every earlier message goes again,
    # oldest first, before the new question.
    messages: list[MessageParam] = []
    if history is not None:
        messages = list(history)
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
