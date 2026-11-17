import json
from typing import Any

import pytest
from test_tools import ANSWERED, NOW, asked_for, call, use_answers

from askkit.cli import main
from askkit.tools import TOO_LARGE, calculate, run_tool

# A number with 401 digits: Python keeps it whole as an integer, but no
# float can hold it.
HUGE = 10**400

# How the model would write that number into its request: as JSON
# digits, which Python reads back as an integer.
HUGE_REQUEST = '{"operation": "multiply", "a": 1' + "0" * 400 + ', "b": 0.5}'


def test_the_request_holds_the_number_as_an_integer() -> None:
    assert json.loads(HUGE_REQUEST)["a"] == HUGE


@pytest.mark.parametrize(
    "tool_input",
    [
        json.loads(HUGE_REQUEST),
        {"operation": "add", "a": HUGE, "b": HUGE},
        {"operation": "multiply", "a": 1e308, "b": 10},
        {"operation": "divide", "a": 1e308, "b": 1e-10},
        json.loads('{"operation": "add", "a": 1e400, "b": 1}'),
        json.loads('{"operation": "subtract", "a": 1e400, "b": 1e400}'),
    ],
    ids=[
        "integer-times-float",
        "integer-sum",
        "float-product",
        "float-quotient",
        "infinite-input",
        "no-number",
    ],
)
def test_a_result_too_large_for_a_number_is_refused(
    tool_input: dict[str, object],
) -> None:
    # Before this lesson the first case crashed AskKit with OverflowError
    # and the others came back as the text inf or nan.
    with pytest.raises(ValueError) as refused:
        calculate(tool_input)
    assert str(refused.value) == TOO_LARGE


def test_a_large_result_that_fits_still_comes_back() -> None:
    assert calculate({"operation": "multiply", "a": 1e300, "b": 10}) == (
        "1e+301"
    )


def broken_clock() -> str:
    # A fault in AskKit's own code, with a detail meant for you alone.
    raise OSError("[Errno 2] No such file: '/Users/ana/.askkit/zone'")


def test_a_fault_in_a_tool_becomes_an_error_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("askkit.tools.current_time", broken_clock)
    assert run_tool(call("current_time", {})) == {
        "type": "tool_result",
        "tool_use_id": "toolu_01AskKitClock",
        "content": (
            "current_time failed because of a fault in AskKit (OSError);"
            " do not call it again for this request: tell the user it"
            " could not be done"
        ),
        "is_error": True,
    }


def test_the_faults_own_text_never_reaches_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("askkit.tools.current_time", broken_clock)
    content = run_tool(call("current_time", {}))["content"]
    assert isinstance(content, str)
    assert "/Users/ana" not in content
    assert "Errno" not in content


def pressed_ctrl_c() -> str:
    raise KeyboardInterrupt


def test_ctrl_c_inside_a_tool_is_not_turned_into_a_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ctrl-C is yours, not a fault: it still stops AskKit.
    monkeypatch.setattr("askkit.tools.current_time", pressed_ctrl_c)
    with pytest.raises(KeyboardInterrupt):
        run_tool(call("current_time", {}))


@pytest.mark.parametrize(
    ("name", "tool_input", "message"),
    [
        (
            "current_time",
            {"zone": "Europe/Kyiv"},
            "current_time takes no input; call it again with an empty"
            " input, {}",
        ),
        (
            "calculate",
            {"operation": "divide", "a": 5, "b": 0},
            "cannot divide by zero; a number divided by zero has no"
            " answer, so tell the user that instead of calling again",
        ),
        (
            "calculate",
            {"operation": "power", "a": 2, "b": 3},
            "operation must be add, subtract, multiply or divide; call it"
            " again with one of them",
        ),
        (
            "cancel_order",
            {"order": 4417},
            "there is no tool named cancel_order; the tools are"
            " current_time, calculate",
        ),
    ],
    ids=["clock-input", "zero", "operation", "unknown-tool"],
)
def test_every_error_result_says_what_to_do_next(
    name: str, tool_input: dict[str, object], message: str
) -> None:
    assert run_tool(call(name, tool_input)) == {
        "type": "tool_result",
        "tool_use_id": "toolu_01AskKitClock",
        "content": message,
        "is_error": True,
    }


def test_askkit_ask_survives_a_number_too_large(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The cold open of the lesson: before it, this run ended in a
    # traceback and the model never heard what went wrong.
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    request = asked_for("calculate", json.loads(HUGE_REQUEST))
    use_answers(monkeypatch, sent, [request, ANSWERED])
    main(["ask", "What is 10 to the power 400, times 0.5?"])
    assert len(sent) == 2
    result = sent[1]["messages"][2]["content"][0]
    assert result == {
        "type": "tool_result",
        "tool_use_id": "toolu_01AskKitClock",
        "content": TOO_LARGE,
        "is_error": True,
    }
    assert capsys.readouterr().out.splitlines()[0] == (
        f"Tool error:    calculate: {TOO_LARGE}"
    )


def test_a_result_that_worked_is_still_a_plain_tool_line(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [asked_for("current_time", {}), ANSWERED])
    main(["ask", "What is today's date?"])
    assert capsys.readouterr().out.splitlines()[0] == (
        f"Tool:          current_time: {NOW}"
    )
