from collections.abc import Callable, Sequence
from dataclasses import dataclass

from anthropic import Anthropic
from anthropic.types import (
    Message,
    MessageParam,
    ToolChoiceParam,
    ToolResultBlockParam,
    ToolUnionParam,
)

from askkit.ask import (
    Content,
    Effort,
    Timing,
    answer_text,
    messages_for,
    stream_answer,
)
from askkit.caching import cached
from askkit.permissions import Permission, run_permitted
from askkit.terminal import printable
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

# The most times one question continues a turn the API paused. A tool
# that runs on Anthropic's servers runs in a loop there too, and the API
# may pause that loop and hand the turn back unfinished (pause_turn).
MAX_PAUSES = 3


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
    question: Content,
    model: str,
    max_tokens: int,
    system: str,
    effort: Effort | None,
    history: list[MessageParam],
    tools: Sequence[ToolUnionParam] | None,
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
    # The question carries a breakpoint, so every later call reads the
    # first call's input from the cache. The tool results do not: a
    # result such as a page is written to the cache at 1.25 times the
    # price, and the call that would read it is often never made.
    content: Content | None = cached(question)
    rounds = 0
    pauses = 0
    while True:
        # A paused turn goes on unforced: its reply has already begun.
        forced = use_tool if content is not None else None
        reply, timing = stream_answer(
            client,
            content,
            model,
            max_tokens,
            system,
            effort,
            sent,
            tools,
            tool_choice_for(rounds, forced) if tools else None,
        )
        timings.append(timing)
        if reply is None:
            return Answered(replies, timings, cancelled=True)
        replies.append(reply)
        if reply.stop_reason == "pause_turn" and pauses < MAX_PAUSES:
            # The API paused its own loop. The paused turn goes back as
            # it came, with no new message, and the API carries on where
            # it stopped. The same tools go too: a search may still wait.
            pauses += 1
            sent = messages_for(content, sent)
            sent.append({"role": "assistant", "content": reply.content})
            content = None
            continue
        # The call after the last round is sent with tool_choice "none",
        # so a request for a tool can only come back from it if the API
        # did not hold to that; nothing more runs then either.
        if reply.stop_reason != "tool_use" or rounds == MAX_ROUNDS:
            return Answered(replies, timings, cancelled=False)
        rounds += 1
        results: list[ToolResultBlockParam] = []
        for call in tool_calls(reply):
            result = run_permitted(call, permissions, confirm, sites)
            # A result can hold a site's own words, such as the address
            # a page moved to, so it is printed as printable text.
            if result.get("is_error"):
                # So you can tell a refusal or a failure from a result.
                error = printable(str(result["content"]))
                print(f"Tool error:    {call.name}: {error}")
            else:
                text = str(result["content"])
                if len(text) > LONG_RESULT:
                    text = f"{len(text):,} characters"
                else:
                    text = printable(text)
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
    rounds: list[Message] = []
    for reply in replies[:-1]:
        # A paused turn ran no tool of yours, so it is not a round.
        if reply.stop_reason == "tool_use":
            rounds.append(reply)
    if not rounds:
        return None
    requests = 0
    for reply in rounds:
        requests += len(tool_calls(reply))
    if len(rounds) == 1:
        return f"{requests} in 1 round"
    return f"{requests} in {len(rounds)} rounds"


def final_text(replies: list[Message]) -> str:
    # The answer's words. A turn the API paused and then continued is
    # one answer in several replies, so their text is joined.
    parts: list[str] = []
    for reply in reversed(replies[:-1]):
        if reply.stop_reason != "pause_turn":
            break
        parts.insert(0, answer_text(reply))
    parts.append(answer_text(replies[-1]))
    return "".join(parts)
