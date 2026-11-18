from collections.abc import Callable
from dataclasses import dataclass

from anthropic import Anthropic
from anthropic.types import (
    Message,
    MessageParam,
    ToolChoiceParam,
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

# A result longer than this is shown as its size: a page would fill the
# terminal.
LONG_RESULT = 200

TOO_MANY_ROUNDS = (
    f"the model still asked for a tool after {MAX_ROUNDS} rounds;"
    " nothing more ran"
)


TOOL_LIMIT = f"{MAX_ROUNDS} rounds ran; the next call may not ask for a tool"


def tool_choice_for(
    rounds: int, use_tool: str | None
) -> ToolChoiceParam | None:
    # What the next call lets the model do with the tools. None leaves the
    # field out, and the model decides for itself ("auto").
    if rounds == MAX_ROUNDS:
        # The last call AskKit allows: the model answers with what it has.
        return {"type": "none"}
    if rounds == 0 and use_tool is not None:
        # Forced only on the first call. A call forced every round could
        # never end with an answer, only with the round limit.
        return {"type": "tool", "name": use_tool}
    return None


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
    sites: tuple[str, ...] = (),
    use_tool: str | None = None,
) -> Answered:
    # Call, run every request, send the results back, and call again,
    # until the model answers instead of asking. Every request of every
    # round passes your permission before anything runs. With use_tool,
    # the first call must ask for that tool.
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
            tool_choice_for(rounds, use_tool) if tools else None,
        )
        timings.append(timing)
        if reply is None:
            return Answered(replies, timings, cancelled=True)
        replies.append(reply)
        # The call after the last round is sent with tool_choice "none",
        # so a request for a tool can only come back from it if the API
        # did not hold to that; nothing more runs then either.
        if reply.stop_reason != "tool_use" or rounds == MAX_ROUNDS:
            return Answered(replies, timings, cancelled=False)
        rounds += 1
        results: list[ToolResultBlockParam] = []
        for call in tool_calls(reply):
            result = run_permitted(call, permissions, confirm, sites)
            if result.get("is_error"):
                # So you can tell a refusal or a failure from a result.
                print(f"Tool error:    {call.name}: {result['content']}")
            else:
                text = str(result["content"])
                if len(text) > LONG_RESULT:
                    text = f"{len(text):,} characters"
                print(f"Tool:          {call.name}: {text}")
            results.append(result)
        # The next call carries everything so far: what was sent, the
        # model's turn as it came, then the results as the next message.
        sent = messages_for(content, sent)
        sent.append({"role": "assistant", "content": reply.content})
        content = results
        if rounds == MAX_ROUNDS:
            print(f"Tool limit:    {TOOL_LIMIT}")


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
