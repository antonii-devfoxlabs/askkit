import json
from collections.abc import Callable
from typing import Literal

from anthropic.types import ToolParam, ToolResultBlockParam, ToolUseBlock

from askkit.tools import error_result, run_tool

# What AskKit may do with a tool the model asks for: run it, ask you
# first, or never offer it at all.
Permission = Literal["allow", "ask", "deny"]

# Reading the clock and doing a sum change nothing and send nothing
# anywhere, so both run unless you say otherwise.
DEFAULT_PERMISSIONS: dict[str, Permission] = {
    "current_time": "allow",
    "calculate": "allow",
}


def permissions_for(
    confirm: list[str], deny: list[str]
) -> dict[str, Permission]:
    # The defaults, changed by what you asked for on the command line. A
    # tool named in both lists is denied: no stays no.
    permissions = dict(DEFAULT_PERMISSIONS)
    for name in confirm:
        permissions[name] = "ask"
    for name in deny:
        permissions[name] = "deny"
    return permissions


def offered(
    tools: list[ToolParam], permissions: dict[str, Permission]
) -> list[ToolParam]:
    # A tool you denied is never offered, so the model cannot ask for it
    # and its definition costs no tokens.
    kept: list[ToolParam] = []
    for tool in tools:
        if permissions.get(tool["name"], "deny") != "deny":
            kept.append(tool)
    return kept


def request_line(call: ToolUseBlock) -> str:
    # The request exactly as it would run: the tool and the model's input,
    # not the model's own words about it.
    return f"{call.name} {json.dumps(call.input)}"


def run_permitted(
    call: ToolUseBlock,
    permissions: dict[str, Permission],
    confirm: Callable[[str], bool],
) -> ToolResultBlockParam:
    # Checked by AskKit before anything runs, whatever the model was told
    # or wrote: the model proposes, you decide.
    permission = permissions.get(call.name)
    if permission is None:
        # Not one of AskKit's tools: run_tool answers that there is none.
        return run_tool(call)
    if permission == "deny":
        return error_result(call, f"the user has not allowed {call.name}")
    if permission == "ask" and not confirm(request_line(call)):
        return error_result(call, f"the user declined to run {call.name}")
    return run_tool(call)
