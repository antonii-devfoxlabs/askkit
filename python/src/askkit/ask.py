from anthropic import Anthropic
from anthropic.types import Message

from askkit.prices import PRICES

# Anthropic's SDK reaches only Anthropic's models.
MODELS: list[str] = []
for name in sorted(PRICES):
    if name.startswith("claude-"):
        MODELS.append(name)


def ask(
    client: Anthropic,
    question: str,
    model: str,
    max_tokens: int,
    system: str,
) -> Message:
    return client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": question}],
    )


def answer_text(response: Message) -> str:
    # The model may think first, so the text is not always block 0.
    parts: list[str] = []
    for block in response.content:
        if block.type == "text":
            parts.append(block.text)
    return "".join(parts)


# The stop reasons of an answer that is whole.
FINISHED = ("end_turn", "stop_sequence")


def stop_warning(response: Message) -> str | None:
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
        return "the model declined to answer this question"
    return f"the answer stopped early: {reason}"
