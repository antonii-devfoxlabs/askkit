from copy import deepcopy
from typing import Any

import pytest
from anthropic.types import Message
from test_loop import PRODUCT, client_for, counted_runs, loop_with, streamed
from test_tools import ANSWERED, NOW, use_answers

from askkit.cli import main
from askkit.loop import requests_line

QUESTION = (
    "What is the date today, and what do 1,284 calls at $0.000137 each cost?"
)


def both_at_once() -> dict[str, Any]:
    # The two halves of the question do not depend on each other, so the
    # model asks for the clock and the calculator in one turn, with a line
    # of text before them.
    return {
        "id": "msg_01AskKitBoth",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "test"},
            {"type": "text", "text": "I'll check both."},
            {
                "type": "tool_use",
                "id": "toolu_01AskKitClock",
                "name": "current_time",
                "input": {},
            },
            {
                "type": "tool_use",
                "id": "toolu_01AskKitSum",
                "name": "calculate",
                "input": PRODUCT,
            },
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 618, "output_tokens": 86},
    }


def one_round() -> list[dict[str, Any]]:
    answered = deepcopy(ANSWERED)
    answered["content"][0]["text"] = (
        "Today is 17 November 2026, and 1,284 calls cost about $0.18."
    )
    answered["usage"] = {"input_tokens": 741, "output_tokens": 34}
    return [both_at_once(), answered]


def test_both_requests_of_one_turn_run_in_one_round(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, one_round())
    main(["ask", QUESTION])
    # One round, so two calls, not three.
    assert len(sent) == 2
    printed = capsys.readouterr().out.splitlines()
    # The text the model wrote before its requests is shown as it
    # streams, then each request's result, in the order asked.
    assert printed[:3] == [
        "I'll check both.",
        f"Tool:          current_time: {NOW}",
        "Tool:          calculate: 0.175908",
    ]
    assert printed[-4:] == [
        "Stop reason:   end_turn",
        "Tool requests: 2 in 1 round",
        "Time:          first token 0.42 s, total 0.84 s",
        "Time:          first token 0.42 s, total 0.84 s",
    ]


def test_every_result_goes_back_in_one_message_in_the_order_asked(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, one_round())
    main(["ask", QUESTION])
    roles = [message["role"] for message in sent[1]["messages"]]
    assert roles == ["user", "assistant", "user"]
    _, turn, results = sent[1]["messages"]
    # The turn goes back as it came: thinking, text and both requests.
    assert turn["content"] == both_at_once()["content"]
    # One result per request, matched by id, results only: no text
    # before them, and no message of its own for each.
    assert results["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_01AskKitClock",
            "content": NOW,
        },
        {
            "type": "tool_result",
            "tool_use_id": "toolu_01AskKitSum",
            "content": "0.175908",
        },
    ]


def test_a_no_to_one_request_still_runs_the_other(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda: "n")
    ran = counted_runs(monkeypatch)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, one_round())
    main(["ask", QUESTION, "--confirm", "calculate"])
    assert ran == []
    # You were asked about the calculator only, and both results went
    # back: the clock's, and the refusal in the calculator's place.
    assert capsys.readouterr().err == (
        'Run calculate {"operation": "multiply", "a": 1284,'
        ' "b": 0.000137}? [y/N] '
    )
    assert sent[1]["messages"][2]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_01AskKitClock",
            "content": NOW,
        },
        {
            "type": "tool_result",
            "tool_use_id": "toolu_01AskKitSum",
            "content": "the user declined to run calculate",
            "is_error": True,
        },
    ]


def test_you_are_asked_about_each_request_in_the_order_asked(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    replies = iter(["y", "y"])
    monkeypatch.setattr("builtins.input", lambda: next(replies))
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, one_round())
    main(
        [
            "ask",
            QUESTION,
            "--confirm",
            "current_time",
            "--confirm",
            "calculate",
        ]
    )
    assert capsys.readouterr().err == (
        "Run current_time {}? [y/N] "
        'Run calculate {"operation": "multiply", "a": 1284,'
        ' "b": 0.000137}? [y/N] '
    )


def test_the_loop_runs_a_round_of_two_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("askkit.tools.current_time", lambda: NOW)
    ran = counted_runs(monkeypatch)
    sent: list[dict[str, Any]] = []
    answers = [streamed(body) for body in one_round()]
    answered = loop_with(client_for(sent, answers))
    assert ran == [PRODUCT]
    assert len(answered.replies) == 2
    assert answered.replies[-1].stop_reason == "end_turn"


def reply(body: dict[str, Any]) -> Message:
    return Message.model_validate(body)


def test_requests_line_counts_requests_and_rounds() -> None:
    answered = reply(ANSWERED)
    both = reply(both_at_once())
    assert requests_line([answered]) is None
    assert requests_line([both, answered]) == "2 in 1 round"
    assert requests_line([both, both, answered]) == "4 in 2 rounds"


def test_requests_line_leaves_out_a_last_request_that_never_ran() -> None:
    # At the round limit the last reply still asks; nothing of it ran.
    both = reply(both_at_once())
    assert requests_line([both, both]) == "2 in 1 round"
