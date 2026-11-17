from datetime import datetime

from anthropic.types import (
    Message,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlock,
)

from askkit.fetch import FETCH, fetch_page

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

TOOLS: list[ToolParam] = [CLOCK, CALCULATOR, FETCH]

# An error result is all the model learns about a failure, so each one
# says what went wrong and what to do next: call again with a fixed
# input, or tell the user, and never call again with the same one.
TOO_LARGE = (
    "the result is too large for a number; tell the user it is too large"
    " to work out, instead of calling again"
)


def current_time() -> str:
    # The computer's clock in its own time zone, such as
    # 2026-11-17T18:40:12+02:00.
    return datetime.now().astimezone().isoformat(timespec="seconds")


def calculate(tool_input: dict[str, object]) -> str:
    # Raises ValueError, saying what is wrong and what to do instead, for
    # an input it cannot use.
    if set(tool_input) != {"operation", "a", "b"}:
        raise ValueError(
            "calculate takes exactly operation, a and b; call it again with"
            " all three and nothing else"
        )
    operation = tool_input["operation"]
    a = tool_input["a"]
    b = tool_input["b"]
    # Capitals in an enum value are not guaranteed, so compare in lower
    # case.
    if not isinstance(operation, str) or operation.lower() not in OPERATIONS:
        raise ValueError(
            "operation must be add, subtract, multiply or divide; call it"
            " again with one of them"
        )
    # Booleans are refused by name, before the check for numbers.
    if (
        isinstance(a, bool)
        or isinstance(b, bool)
        or not isinstance(a, (int, float))
        or not isinstance(b, (int, float))
    ):
        raise ValueError(
            "a and b must be numbers, such as 1000 or 0.000128; call it"
            " again with both written as numbers"
        )
    operation = operation.lower()
    # The schema cannot say that b is not zero, so the check does.
    if operation == "divide" and b == 0:
        raise ValueError(
            "cannot divide by zero; a number divided by zero has no answer,"
            " so tell the user that instead of calling again"
        )
    try:
        if operation == "add":
            result = a + b
        elif operation == "subtract":
            result = a - b
        elif operation == "multiply":
            result = a * b
        else:
            result = a / b
        # At most 15 significant digits, the most a float always keeps,
        # so 0.1 + 0.2 comes out as 0.3, not 0.30000000000000004.
        text = format(result, ".15g")
    except OverflowError:
        # An integer too large to turn into a float.
        text = "inf"
    # A float too large to hold is inf, and inf - inf is nan: neither is
    # a number the user can use.
    if text in ("inf", "-inf", "nan"):
        raise ValueError(TOO_LARGE)
    return text


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


def tool_output(call: ToolUseBlock, sites: tuple[str, ...]) -> str:
    # Runs the tool the model asked for, after checking the request.
    # Raises ValueError, saying what is wrong and what to do instead, for
    # a request it cannot run.
    if call.name == CLOCK["name"]:
        if call.input != {}:
            raise ValueError(
                "current_time takes no input; call it again with an empty"
                " input, {}"
            )
        return current_time()
    if call.name == CALCULATOR["name"]:
        return calculate(call.input)
    if call.name == FETCH["name"]:
        return fetch_page(call.input, sites)
    names = []
    for tool in TOOLS:
        names.append(tool["name"])
    raise ValueError(
        f"there is no tool named {call.name}; the tools are"
        f" {', '.join(names)}"
    )


def run_tool(
    call: ToolUseBlock, sites: tuple[str, ...] = ()
) -> ToolResultBlockParam:
    # The model only asks. Whatever happens while a request runs, it gets
    # a result: the tool's output, or an error result the model can act
    # on. Nothing a tool raises stops AskKit.
    try:
        content = tool_output(call, sites)
    except ValueError as error:
        return error_result(call, str(error))
    except Exception as error:
        # A fault in AskKit itself, not in the request, so the model is
        # told not to try again. Only the error's kind goes back: its own
        # text may hold paths or values meant for you alone.
        return error_result(
            call,
            f"{call.name} failed because of a fault in AskKit"
            f" ({type(error).__name__}); do not call it again for this"
            " request: tell the user it could not be done",
        )
    return {"type": "tool_result", "tool_use_id": call.id, "content": content}
