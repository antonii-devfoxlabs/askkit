from collections.abc import Callable
from dataclasses import dataclass

from anthropic import Anthropic
from anthropic.types import (
    Message,
    MessageParam,
    ToolParam,
    ToolResultBlockParam,
)

from askkit.ask import Effort, Timing, messages_for, stream_answer
from askkit.permissions import Permission, run_permitted
from askkit.tools import tool_calls

# The most rounds of tool requests one question may run. Every round is
# one more call that sends the whole conversation again, so a model that
# keeps asking would cost more with every round and never end.
MAX_ROUNDS = 5

TOO_MANY_ROUNDS = (
    f"the model still asked for a tool after {MAX_ROUNDS} rounds;"
    " nothing more ran"
)


@dataclass
class Answered:
    # Every reply the question took, in order, and the time of every
    # call. When you cancel a call with Ctrl-C, it has a time but no
    # reply, and cancelled is True.
    replies: list[Message]
    timings: list[Timing]
    cancelled: bool


def tool_loop(
    client: Anthropic,
    question: str,
    model: str,
    max_tokens: int,
    system: str,
    effort: Effort | None,
    history: list[MessageParam],
    tools: list[ToolParam] | None,
    permissions: dict[str, Permission],
    confirm: Callable[[str], bool],
) -> Answered:
    # Call, run every request, send the results back, and call again,
    # until the model answers instead of asking. Every request of every
    # round passes your permission before anything runs.
    replies: list[Message] = []
    timings: list[Timing] = []
    sent = history
    content: str | list[ToolResultBlockParam] = question
    rounds = 0
    while True:
        reply, timing = stream_answer(
            client,
            content,
            model,
            max_tokens,
            system,
            effort,
            sent,
            tools,
        )
        timings.append(timing)
        if reply is None:
            return Answered(replies, timings, cancelled=True)
        replies.append(reply)
        if reply.stop_reason != "tool_use" or rounds == MAX_ROUNDS:
            return Answered(replies, timings, cancelled=False)
        rounds += 1
        results: list[ToolResultBlockParam] = []
        for call in tool_calls(reply):
            result = run_permitted(call, permissions, confirm)
            print(f"Tool:          {call.name}: {result['content']}")
            results.append(result)
        # The next call carries everything so far: what was sent, the
        # model's turn as it came, then the results as the next message.
        sent = messages_for(content, sent)
        sent.append({"role": "assistant", "content": reply.content})
        content = results


def requests_line(replies: list[Message]) -> str | None:
    # How many tool requests a question ran, and in how many rounds. Every
    # reply but the last asked for tools that ran, one round each; the
    # last one answered, or asked once too often and ran nothing.
    rounds = replies[:-1]
    if not rounds:
        return None
    requests = 0
    for reply in rounds:
        requests += len(tool_calls(reply))
    if len(rounds) == 1:
        return f"{requests} in 1 round"
    return f"{requests} in {len(rounds)} rounds"
