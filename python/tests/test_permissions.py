from copy import deepcopy
from typing import Any

import pytest
from test_tools import ANSWERED, NOW, asked_for, call, use_answers

from askkit.cli import main
from askkit.fetch import FETCH
from askkit.permissions import (
    DEFAULT_PERMISSIONS,
    offered,
    permissions_for,
    request_line,
    run_permitted,
)
from askkit.tools import CALCULATOR, CLOCK, TOOLS

PRODUCT = {"operation": "multiply", "a": 1284, "b": 0.000137}


def never_asked(request: str) -> bool:
    raise AssertionError(f"asked about {request}")


def answer_with(reply: bool, asked: list[str]) -> Any:
    # A person at the keyboard who gives the same reply every time, and
    # remembers what they were asked.
    def confirm(request: str) -> bool:
        asked.append(request)
        return reply

    return confirm


def test_every_tool_has_a_default_permission() -> None:
    assert sorted(DEFAULT_PERMISSIONS) == sorted(
        tool["name"] for tool in TOOLS
    )


def test_every_tool_runs_without_asking_by_default() -> None:
    assert permissions_for([], []) == {
        "current_time": "allow",
        "calculate": "allow",
        "fetch_page": "allow",
    }


def test_confirm_and_deny_change_one_tool_each() -> None:
    permissions = permissions_for(["calculate"], ["current_time"])
    assert permissions == {
        "current_time": "deny",
        "calculate": "ask",
        "fetch_page": "allow",
    }


def test_a_tool_both_confirmed_and_denied_is_denied() -> None:
    permissions = permissions_for(["calculate"], ["calculate"])
    assert permissions["calculate"] == "deny"


def test_the_defaults_are_never_changed() -> None:
    permissions_for(["calculate"], ["current_time"])
    assert DEFAULT_PERMISSIONS["calculate"] == "allow"
    assert DEFAULT_PERMISSIONS["current_time"] == "allow"


def test_a_denied_tool_is_not_offered() -> None:
    permissions = permissions_for([], ["current_time"])
    assert offered(TOOLS, permissions) == [CALCULATOR, FETCH]


def test_a_tool_to_confirm_is_still_offered() -> None:
    permissions = permissions_for(["calculate"], [])
    assert offered(TOOLS, permissions) == [CLOCK, CALCULATOR, FETCH]


def test_a_tool_with_no_permission_is_not_offered() -> None:
    assert offered(TOOLS, {"calculate": "allow"}) == [CALCULATOR]


def test_the_request_line_shows_the_exact_input() -> None:
    assert request_line(call("calculate", PRODUCT)) == (
        'calculate {"operation": "multiply", "a": 1284, "b": 0.000137}'
    )


def test_an_allowed_tool_runs_without_asking() -> None:
    result = run_permitted(
        call("calculate", PRODUCT), {"calculate": "allow"}, never_asked
    )
    assert result["content"] == "0.175908"
    assert "is_error" not in result


def test_a_denied_tool_runs_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    def never_run(tool_input: dict[str, object]) -> str:
        raise AssertionError("the calculator ran")

    monkeypatch.setattr("askkit.tools.calculate", never_run)
    result = run_permitted(
        call("calculate", PRODUCT), {"calculate": "deny"}, never_asked
    )
    assert result == {
        "type": "tool_result",
        "tool_use_id": "toolu_01AskKitClock",
        "content": (
            "the user has not allowed calculate; do not ask for it again:"
            " answer"
            " without it, or tell the user it is turned off"
        ),
        "is_error": True,
    }


def test_a_tool_to_confirm_runs_after_a_yes() -> None:
    asked: list[str] = []
    result = run_permitted(
        call("calculate", PRODUCT),
        {"calculate": "ask"},
        answer_with(True, asked),
    )
    assert result["content"] == "0.175908"
    assert asked == [request_line(call("calculate", PRODUCT))]


def test_a_tool_to_confirm_runs_nothing_after_a_no(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def never_run(tool_input: dict[str, object]) -> str:
        raise AssertionError("the calculator ran")

    monkeypatch.setattr("askkit.tools.calculate", never_run)
    result = run_permitted(
        call("calculate", PRODUCT),
        {"calculate": "ask"},
        answer_with(False, []),
    )
    assert result["content"] == (
        "the user declined to run calculate; do not ask for it again with the"
        " same input: answer without it, or ask the user what to do instead"
    )
    assert result.get("is_error") is True


def test_a_tool_askkit_does_not_have_is_answered_as_unknown() -> None:
    result = run_permitted(
        call("cancel_order", {"order": 4417}),
        DEFAULT_PERMISSIONS,
        never_asked,
    )
    assert result["content"] == (
        "there is no tool named cancel_order; the tools are current_time,"
        " calculate, fetch_page"
    )
    assert result.get("is_error") is True


def calculator_asked_and_answered() -> list[dict[str, Any]]:
    answered = deepcopy(ANSWERED)
    answered["content"][0]["text"] = "About 18 cents."
    return [asked_for("calculate", PRODUCT), answered]


def test_askkit_ask_offers_only_what_you_allow(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [ANSWERED])
    main(["ask", "What is 2 + 2?", "--deny", "current_time"])
    assert sent[0]["tools"] == [CALCULATOR]


def test_askkit_ask_with_every_tool_denied_offers_none(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [ANSWERED])
    main(
        [
            "ask",
            "What is 2 + 2?",
            "--deny",
            "current_time",
            "--deny",
            "calculate",
        ]
    )
    assert "tools" not in sent[0]


def test_askkit_ask_refuses_an_unknown_tool_name(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["ask", "Hi", "--deny", "send_email"])
    assert stopped.value.code == 2
    assert "invalid choice: 'send_email'" in capsys.readouterr().err


def test_askkit_ask_asks_before_it_runs_a_tool_to_confirm(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda: "y")
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, calculator_asked_and_answered())
    main(["ask", "What do 1,284 calls cost?", "--confirm", "calculate"])
    printed = capsys.readouterr()
    assert printed.err == (
        'Run calculate {"operation": "multiply", "a": 1284,'
        ' "b": 0.000137}? [y/N] '
    )
    assert printed.out.splitlines()[0] == "Tool:          calculate: 0.175908"
    assert sent[1]["messages"][2]["content"][0]["content"] == "0.175908"


@pytest.mark.parametrize("reply", ["", "n", "no", "Yes please"])
def test_askkit_ask_sends_a_refusal_unless_you_say_yes(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    reply: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda: reply)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, calculator_asked_and_answered())
    main(["ask", "What do 1,284 calls cost?", "--confirm", "calculate"])
    assert sent[1]["messages"][2]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_01AskKitClock",
            "content": (
                "the user declined to run calculate; do not ask for it again"
                " with the"
                " same input: answer without it, or ask the user what to do"
                " instead"
            ),
            "is_error": True,
        }
    ]
    assert capsys.readouterr().out.splitlines()[0] == (
        "Tool error:    calculate: the user declined to run calculate; do not"
        " ask for it again with the same input: answer without it, or ask the"
        " user what to do instead"
    )


@pytest.mark.parametrize("stop", [EOFError, KeyboardInterrupt])
def test_askkit_ask_takes_no_answer_as_a_no(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stop: type[BaseException],
) -> None:
    def no_one_answers() -> str:
        raise stop

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", no_one_answers)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, calculator_asked_and_answered())
    main(["ask", "What do 1,284 calls cost?", "--confirm", "calculate"])
    assert sent[1]["messages"][2]["content"][0]["is_error"] is True
    # The prompt's line is ended, so the next output starts on its own.
    assert capsys.readouterr().err.endswith("[y/N] \n")


def test_askkit_ask_never_asks_about_an_allowed_tool(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never_called() -> str:
        raise AssertionError("AskKit asked")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", never_called)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [asked_for("current_time", {}), ANSWERED])
    main(["ask", "What is today's date?"])
    assert sent[1]["messages"][2]["content"][0]["content"] == NOW


def test_a_tool_with_no_permission_never_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def never_run(tool_input: dict[str, object]) -> str:
        raise AssertionError("the calculator ran")

    monkeypatch.setattr("askkit.tools.calculate", never_run)
    result = run_permitted(
        call("calculate", PRODUCT), {"current_time": "allow"}, never_asked
    )
    assert result["content"] == (
        "the user has not allowed calculate; do not ask for it again: answer"
        " without it, or tell the user it is turned off"
    )
    assert result.get("is_error") is True
