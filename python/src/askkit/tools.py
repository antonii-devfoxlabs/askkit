from datetime import datetime

from anthropic.types import (
    Message,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
)

# What the model is told about the clock: a name to ask for it by, what
# it does and when to ask, and the shape of its input (here, none).
CLOCK: ToolParam = {
    "name": "current_time",
    "description": (
        "The current date and time on the user's computer, with its offset"
        " from UTC. Call it when an answer depends on today's date or the"
        " time now."
    ),
    "input_schema": {"type": "object", "properties": {}},
}

TOOLS: list[ToolParam] = [CLOCK]


def current_time() -> str:
    # The computer's clock in its own time zone, such as
    # 2026-11-17T18:40:12+02:00.
    return datetime.now().astimezone().isoformat(timespec="seconds")


def tool_calls(response: Message) -> list[ToolUseBlock]:
    # Every tool the model asked for in this answer, in its order.
    calls: list[ToolUseBlock] = []
    for block in response.content:
        if block.type == "tool_use":
            calls.append(block)
    return calls


def run_tool(call: ToolUseBlock) -> ToolResultBlockParam:
    # The model only asks. Nothing runs until the request is checked: a
    # tool AskKit offered, with the input it takes. Anything else is
    # answered with an error result, and nothing runs.
    if call.name != CLOCK["name"]:
        return {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": f"there is no tool named {call.name}",
            "is_error": True,
        }
    if call.input != {}:
        return {
            "type": "tool_result",
            "tool_use_id": call.id,
            "content": "current_time takes no input",
            "is_error": True,
        }
    return {
        "type": "tool_result",
        "tool_use_id": call.id,
        "content": current_time(),
    }
