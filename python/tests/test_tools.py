import json
import re
from copy import deepcopy
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic
from anthropic.types import Message, ToolUseBlock
from test_ask import replied

from askkit.cli import main
from askkit.tools import CLOCK, TOOLS, current_time, run_tool, tool_calls

# Written for the tests, as the API shapes them: the model thinks, then
# asks for the clock instead of answering.
ASKED: dict[str, Any] = {
    "id": "msg_01AskKitAsked",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [
        {"type": "thinking", "thinking": "", "signature": "test"},
        {
            "type": "tool_use",
            "id": "toolu_01AskKitClock",
            "name": "current_time",
            "input": {},
        },
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {"input_tokens": 471, "output_tokens": 41},
}

# The answer once the clock's result came back.
ANSWERED: dict[str, Any] = {
    "id": "msg_01AskKitAnswered",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [
        {"type": "text", "text": "Today is Tuesday, 17 November 2026."}
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 532, "output_tokens": 14},
}

NOW = "2026-11-17T18:40:12+02:00"


def asked_for(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    # ASKED, with another tool or another input in its request.
    body = deepcopy(ASKED)
    body["content"][1]["name"] = name
    body["content"][1]["input"] = tool_input
    return body


def call(name: str, tool_input: dict[str, object]) -> ToolUseBlock:
    return ToolUseBlock(
        type="tool_use", id="toolu_01AskKitClock", name=name, input=tool_input
    )


def use_answers(
    monkeypatch: pytest.MonkeyPatch,
    sent: list[dict[str, Any]],
    bodies: list[dict[str, Any]],
) -> None:
    # Each request gets the next written answer, in order.
    def answer(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return replied(request, bodies[len(sent) - 1])

    def make(**options: Any) -> Anthropic:
        return Anthropic(
            api_key="sk-ant-api03-a-test-key-Ab12",
            http_client=httpx2.Client(transport=httpx2.MockTransport(answer)),
        )

    monkeypatch.setattr("askkit.cli.Anthropic", make)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    monkeypatch.setattr("askkit.tools.current_time", lambda: NOW)


def test_current_time_is_the_local_time_with_its_offset() -> None:
    pattern = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d"
    assert re.fullmatch(pattern, current_time())


def test_the_clock_takes_no_input() -> None:
    assert TOOLS == [CLOCK]
    assert CLOCK["name"] == "current_time"
    assert CLOCK["input_schema"] == {"type": "object", "properties": {}}


def test_tool_calls_picks_the_requests_in_order() -> None:
    body = deepcopy(ASKED)
    body["content"].append(
        {
            "type": "tool_use",
            "id": "toolu_01Second",
            "name": "current_time",
            "input": {},
        }
    )
    response = Message.model_validate(body)
    ids = [block.id for block in tool_calls(response)]
    assert ids == ["toolu_01AskKitClock", "toolu_01Second"]


def test_an_answer_with_no_request_has_no_tool_calls() -> None:
    assert tool_calls(Message.model_validate(ANSWERED)) == []


def test_run_tool_answers_the_clock_with_its_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("askkit.tools.current_time", lambda: NOW)
    assert run_tool(call("current_time", {})) == {
        "type": "tool_result",
        "tool_use_id": "toolu_01AskKitClock",
        "content": NOW,
    }


def never_run() -> str:
    raise AssertionError("the clock ran for a request it should refuse")


def test_a_tool_askkit_never_offered_runs_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("askkit.tools.current_time", never_run)
    assert run_tool(call("cancel_order", {"order": 4417})) == {
        "type": "tool_result",
        "tool_use_id": "toolu_01AskKitClock",
        "content": "there is no tool named cancel_order",
        "is_error": True,
    }


def test_the_clock_with_an_input_runs_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("askkit.tools.current_time", never_run)
    result = run_tool(call("current_time", {"zone": "Europe/Kyiv"}))
    assert result["is_error"] is True
    assert result["content"] == "current_time takes no input"


def test_askkit_ask_offers_the_clock(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [ANSWERED])
    main(["ask", "What is 2 + 2?"])
    assert sent[0]["tools"] == [CLOCK]
    assert len(sent) == 1


def test_askkit_ask_runs_the_clock_and_sends_the_result_back(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [ASKED, ANSWERED])
    main(["ask", "What is today's date?"])
    assert len(sent) == 2
    question, turn, results = sent[1]["messages"]
    assert question == {"role": "user", "content": "What is today's date?"}
    # The model's own turn goes back as it came, thinking included.
    assert turn["role"] == "assistant"
    assert [block["type"] for block in turn["content"]] == [
        "thinking",
        "tool_use",
    ]
    assert turn["content"][1]["id"] == "toolu_01AskKitClock"
    assert results == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_01AskKitClock",
                "content": NOW,
            }
        ],
    }
    assert sent[1]["tools"] == [CLOCK]
    assert capsys.readouterr().out.splitlines() == [
        f"Tool:          current_time: {NOW}",
        "Today is Tuesday, 17 November 2026.",
        "",
        "Model:         claude-haiku-5-5",
        "Tokens:        1,003 in, 55 out",
        "Cost:          $0.000128",
        "Stop reason:   end_turn",
        "Time:          no text, total 0.42 s",
        "Time:          first token 0.42 s, total 0.84 s",
    ]


def test_askkit_ask_refuses_a_tool_it_never_offered(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(
        monkeypatch,
        sent,
        [asked_for("cancel_order", {"order": 4417}), ANSWERED],
    )
    main(["ask", "Cancel order 4417."])
    result = sent[1]["messages"][2]["content"][0]
    assert result["is_error"] is True
    assert capsys.readouterr().out.splitlines()[0] == (
        "Tool:          cancel_order: there is no tool named cancel_order"
    )


def test_askkit_ask_stops_when_the_model_asks_a_second_time(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [ASKED, ASKED])
    with pytest.raises(SystemExit) as stopped:
        main(["ask", "What is today's date?"])
    assert stopped.value.code == "askkit: the answer stopped early: tool_use"
    assert len(sent) == 2
