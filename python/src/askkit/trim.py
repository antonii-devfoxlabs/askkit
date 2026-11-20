from time import perf_counter
from typing import Literal

from anthropic import Anthropic
from anthropic.types import Message, MessageParam

from askkit.calllog import log_call, request_id_of
from askkit.history import SavedMessage
from askkit.prompts import render

# What askkit chat does with its oldest turns when a turn would not fit
# its budget: drop them, or fold them into a summary first.
OldTurns = Literal["drop", "summarise"]
OLD_TURNS: list[str] = ["drop", "summarise"]

# A summary leaves the last turns word for word: the next question
# usually points at them ("and the second one?").
KEEP_TURNS = 2

# The summary call may think before it writes, and max_tokens covers the
# thinking too, so it gets room for several thousand tokens (Anthropic's
# page on compaction on demand, checked on 2026-11-17).
SUMMARY_MAX_TOKENS = 4_000


def turns_label(first: int, last: int) -> str:
    # Turns are counted from 1, as the Turn line counts them.
    if first == last:
        return f"turn {first}"
    return f"turns {first} to {last}"


def transcript(saved: list[SavedMessage]) -> str:
    lines: list[str] = []
    for message in saved:
        if message.role == "user":
            lines.append(f"User: {message.content}")
        else:
            lines.append(f"Assistant: {message.content}")
    return "\n".join(lines)


def summarise(
    client: Anthropic, model: str, summary: str, folded: list[SavedMessage]
) -> Message:
    # One call folds the turns into the summary so far. Low effort is
    # enough to write a summary, and keeps its thinking short.
    if not summary:
        summary = "No summary yet."
    request = render(
        "fold_turns", {"summary": summary, "turns": transcript(folded)}
    )
    system = render("summarise_turns", {})
    messages: list[MessageParam] = [{"role": "user", "content": request}]
    start = perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=SUMMARY_MAX_TOKENS,
        system=system,
        messages=messages,
        output_config={"effort": "low"},
    )
    seconds = perf_counter() - start
    log_call(
        response, model, seconds, system, messages, request_id_of(response)
    )
    return response


def with_summary(system: str, summary: str) -> str:
    # The summary holds for the rest of the conversation, as the chat's
    # rules do, so it goes in the system prompt, after them.
    if not summary:
        return system
    return render("earlier_turns", {"system": system, "summary": summary})
