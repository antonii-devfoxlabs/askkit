from datetime import datetime

from anthropic.types import (
    Message,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
)

# A definition is all the model ever learns about a tool: the name it
# asks for it by, a description of what it does and when to call it,
# and the schema of its input. Every AskKit tool is strict, so the API
# holds the model's input to that schema; AskKit still checks it.
CLOCK: ToolParam = {
    "name": "current_time",
    "description": (
        "Returns the current date and time on the user's computer, as text"
        " with its offset from UTC, such as 2026-11-17T18:40:12+02:00. Call"
        " it when an answer depends on today's date or the time now, such"
        " as the number of days left until a date. Do not call it for"
        " facts that are the same every day, such as how many days a week"
        " has. It takes no input."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    "strict": True,
}

# The operations the calculator knows, shared by its schema and its check.
OPERATIONS = ["add", "subtract", "multiply", "divide"]

CALCULATOR: ToolParam = {
    "name": "calculate",
    "description": (
        "Does one arithmetic operation on two numbers and returns the"
        " result as text, such as 0.128. Call it whenever an answer needs"
        " a sum, a difference, a product or a quotient, such as the cost"
        " of 1,000 calls, instead of working the number out yourself. It"
        " does one operation per call. Write a percentage as a decimal:"
        " 15% is 0.15. If the question leaves out a number you need, ask"
        " the user for it; never guess one."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": OPERATIONS,
                "description": "a + b, a - b, a * b or a / b, in that order.",
            },
            "a": {
                "type": "number",
                "description": "The first number, such as 1000.",
            },
            "b": {
                "type": "number",
                "description": "The second number, such as 0.000128.",
            },
        },
        "required": ["operation", "a", "b"],
        "additionalProperties": False,
    },
    "strict": True,
}

TOOLS: list[ToolParam] = [CLOCK, CALCULATOR]


def current_time() -> str:
    # The computer's clock in its own time zone, such as
    # 2026-11-17T18:40:12+02:00.
    return datetime.now().astimezone().isoformat(timespec="seconds")


def calculate(tool_input: dict[str, object]) -> str:
    # Raises ValueError, saying what is wrong, for an input it cannot use.
    if set(tool_input) != {"operation", "a", "b"}:
        raise ValueError("calculate takes exactly operation, a and b")
    operation = tool_input["operation"]
    a = tool_input["a"]
    b = tool_input["b"]
    # Capitals in an enum value are not guaranteed, so compare in lower
    # case.
    if not isinstance(operation, str) or operation.lower() not in OPERATIONS:
        raise ValueError(
            "operation must be add, subtract, multiply or divide"
        )
    # Booleans are refused by name, before the check for numbers.
    if (
        isinstance(a, bool)
        or isinstance(b, bool)
        or not isinstance(a, (int, float))
        or not isinstance(b, (int, float))
    ):
        raise ValueError("a and b must be numbers, such as 1000 or 0.000128")
    operation = operation.lower()
    # The schema cannot say that b is not zero, so the check does.
    if operation == "divide" and b == 0:
        raise ValueError("cannot divide by zero")
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    else:
        result = a / b
    # At most 15 significant digits, the most a float always keeps, so
    # 0.1 + 0.2 comes out as 0.3, not 0.30000000000000004.
    return format(result, ".15g")


def tool_calls(response: Message) -> list[ToolUseBlock]:
    # Every tool the model asked for in this answer, in its order.
    calls: list[ToolUseBlock] = []
    for block in response.content:
        if block.type == "tool_use":
            calls.append(block)
    return calls


def error_result(call: ToolUseBlock, message: str) -> ToolResultBlockParam:
    return {
        "type": "tool_result",
        "tool_use_id": call.id,
        "content": message,
        "is_error": True,
    }


def run_tool(call: ToolUseBlock) -> ToolResultBlockParam:
    # The model only asks. Nothing runs until the request is checked: a
    # tool AskKit offered, with an input it can use. Anything else is
    # answered with an error result, and nothing runs.
    if call.name == CLOCK["name"]:
        if call.input != {}:
            return error_result(call, "current_time takes no input")
        content = current_time()
    elif call.name == CALCULATOR["name"]:
        try:
            content = calculate(call.input)
        except ValueError as error:
            return error_result(call, str(error))
    else:
        return error_result(call, f"there is no tool named {call.name}")
    return {"type": "tool_result", "tool_use_id": call.id, "content": content}
