import json
from collections.abc import Iterator
from copy import deepcopy
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic
from anthropic.types import MessageParam
from test_ask import Interrupted, replied
from test_tools import ANSWERED, use_answers

from askkit.caching import cached
from askkit.cli import main
from askkit.loop import MAX_ROUNDS, TOO_MANY_ROUNDS, tool_loop
from askkit.permissions import DEFAULT_PERMISSIONS
from askkit.tools import CALCULATOR, CLOCK, TOOLS

PRODUCT = {"operation": "multiply", "a": 1284, "b": 0.000137}
TOTAL = {"operation": "add", "a": 0.175908, "b": 5}
QUESTION = "What do 1,284 calls at $0.000137 each cost, plus a $5 fee?"


def asked(
    number: int, tool_input: dict[str, Any], input_tokens: int
) -> dict[str, Any]:
    # The model thinks, then asks for the calculator instead of answering.
    # Each request has an id of its own, as the API gives it.
    return {
        "id": f"msg_01AskKitRound{number}",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "test"},
            {
                "type": "tool_use",
                "id": f"toolu_01AskKitSum{number}",
                "name": "calculate",
                "input": tool_input,
            },
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": 50},
    }


def two_sums() -> list[dict[str, Any]]:
    # The second sum needs the first one's result, so it takes two rounds.
    answered = deepcopy(ANSWERED)
    answered["content"][0]["text"] = "About $5.18 in all."
    answered["usage"] = {"input_tokens": 760, "output_tokens": 30}
    return [asked(1, PRODUCT, 618), asked(2, TOTAL, 689), answered]


def client_for(
    sent: list[dict[str, Any]], answers: list[httpx2.Response]
) -> Anthropic:
    # Each request gets the next response, in order.
    def answer(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return answers[len(sent) - 1]

    return Anthropic(
        api_key="sk-ant-api03-a-test-key-Ab12",
        http_client=httpx2.Client(transport=httpx2.MockTransport(answer)),
    )


def streamed(body: dict[str, Any]) -> httpx2.Response:
    request = httpx2.Request(
        "POST", "https://api.anthropic.com/v1/messages", json={"stream": True}
    )
    return replied(request, body)


def always_asking() -> Iterator[httpx2.Response]:
    number = 0
    while True:
        number += 1
        yield streamed(asked(number, PRODUCT, 618))


def never_asked(request: str) -> bool:
    raise AssertionError(f"asked about {request}")


def counted_runs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    # Every input the calculator really ran with, in order.
    ran: list[dict[str, object]] = []

    def counted(tool_input: dict[str, object]) -> str:
        ran.append(tool_input)
        return "0.175908"

    monkeypatch.setattr("askkit.tools.calculate", counted)
    return ran


def loop_with(
    client: Anthropic, history: list[MessageParam] | None = None
) -> Any:
    return tool_loop(
        client,
        QUESTION,
        "claude-haiku-5-5",
        1000,
        "You answer questions.",
        None,
        history or [],
        TOOLS,
        DEFAULT_PERMISSIONS,
        never_asked,
    )


def test_two_rounds_in_a_row_then_the_answer(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, two_sums())
    main(["ask", QUESTION])
    assert len(sent) == 3
    assert capsys.readouterr().out.splitlines() == [
        "Tool:          calculate: 0.175908",
        "Tool:          calculate: 5.175908",
        "About $5.18 in all.",
        "",
        "Model:         claude-haiku-5-5",
        "Tokens:        2,067 in, 130 out",
        "Cost:          $0.000272",
        "Stop reason:   end_turn",
        "Tool requests: 2 in 2 rounds",
        "Time:          no text, total 0.42 s",
        "Time:          no text, total 0.42 s",
        "Time:          first token 0.42 s, total 0.84 s",
    ]


def test_each_call_carries_every_earlier_round(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, two_sums())
    main(["ask", QUESTION])
    roles = [message["role"] for message in sent[2]["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]
    question, first_turn, first_results, second_turn, second_results = sent[
        2
    ]["messages"]
    assert question == {"role": "user", "content": cached(QUESTION)}
    # Each turn goes back as it came, thinking included, and each result
    # answers the request of the turn just before it.
    assert first_turn["content"][1]["id"] == "toolu_01AskKitSum1"
    assert first_results["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_01AskKitSum1",
            "content": "0.175908",
        }
    ]
    assert [block["type"] for block in second_turn["content"]] == [
        "thinking",
        "tool_use",
    ]
    assert second_results["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_01AskKitSum2",
            "content": "5.175908",
        }
    ]
    # Every call offers the same tools: with no site allowed, the clock
    # and the calculator.
    offered = [CLOCK, CALCULATOR]
    assert [request["tools"] for request in sent] == [offered] * 3


def test_every_round_asks_for_your_permission(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    replies = iter(["y", "y"])
    monkeypatch.setattr("builtins.input", lambda: next(replies))
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, two_sums())
    main(["ask", QUESTION, "--confirm", "calculate"])
    assert capsys.readouterr().err == (
        'Run calculate {"operation": "multiply", "a": 1284,'
        ' "b": 0.000137}? [y/N] '
        'Run calculate {"operation": "add", "a": 0.175908, "b": 5}? [y/N] '
    )


def test_a_no_in_the_second_round_runs_nothing_there(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    replies = iter(["y", "n"])
    monkeypatch.setattr("builtins.input", lambda: next(replies))
    ran = counted_runs(monkeypatch)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, two_sums())
    main(["ask", QUESTION, "--confirm", "calculate"])
    assert ran == [PRODUCT]
    assert sent[2]["messages"][4]["content"] == [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_01AskKitSum2",
            "content": (
                "the user declined to run calculate; do not ask for it again"
                " with the"
                " same input: answer without it, or ask the user what to do"
                " instead"
            ),
            "is_error": True,
        }
    ]


def test_the_loop_stops_after_the_last_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[dict[str, Any]] = []
    responses = always_asking()
    answers = [next(responses) for _ in range(MAX_ROUNDS + 2)]
    ran = counted_runs(monkeypatch)
    answered = loop_with(client_for(sent, answers))
    # One call per round, then one more whose request is never run. That
    # last call carries tool_choice "none"; this stand-in asks anyway.
    assert len(sent) == MAX_ROUNDS + 1
    assert sent[MAX_ROUNDS]["tool_choice"] == {"type": "none"}
    assert len(ran) == MAX_ROUNDS
    assert len(answered.replies) == MAX_ROUNDS + 1
    assert answered.replies[-1].stop_reason == "tool_use"
    assert answered.cancelled is False


def test_askkit_ask_says_why_it_stopped_asking(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    # A stand-in that asks even when the last call forbids it.
    bodies = [asked(number, PRODUCT, 618) for number in range(1, 8)]
    use_answers(monkeypatch, sent, bodies)
    with pytest.raises(SystemExit) as stopped:
        main(["ask", QUESTION])
    assert stopped.value.code == f"askkit: {TOO_MANY_ROUNDS}"
    assert TOO_MANY_ROUNDS == (
        "the model still asked for a tool after 5 rounds; nothing more ran"
    )
    printed = capsys.readouterr().out.splitlines()
    assert len(sent) == 6
    tool_lines = [line for line in printed if line.startswith("Tool:")]
    assert len(tool_lines) == 5
    assert "Stop reason:   tool_use" in printed


def test_an_answer_ends_the_loop_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[dict[str, Any]] = []
    ran = counted_runs(monkeypatch)
    answered = loop_with(client_for(sent, [streamed(ANSWERED)]))
    assert ran == []
    assert len(sent) == 1
    assert [reply.stop_reason for reply in answered.replies] == ["end_turn"]


def test_a_refusal_ends_the_loop() -> None:
    sent: list[dict[str, Any]] = []
    refused = deepcopy(ANSWERED)
    refused["stop_reason"] = "refusal"
    answers = [streamed(asked(1, PRODUCT, 618)), streamed(refused)]
    answered = loop_with(client_for(sent, answers))
    assert len(sent) == 2
    assert answered.replies[-1].stop_reason == "refusal"


def test_a_cancel_in_the_second_round_ends_the_loop(
    capsys: pytest.CaptureFixture[str],
) -> None:
    sent: list[dict[str, Any]] = []
    stream = Interrupted(ANSWERED, shown=1)
    answers = [
        streamed(asked(1, PRODUCT, 618)),
        httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=stream
        ),
    ]
    answered = loop_with(client_for(sent, answers))
    assert answered.cancelled is True
    assert len(answered.replies) == 1
    assert len(answered.timings) == 2
    assert stream.closed is True


def test_askkit_ask_counts_the_rounds_before_a_cancel(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    stream = Interrupted(ANSWERED, shown=1)
    answers = [
        streamed(asked(1, PRODUCT, 618)),
        httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, stream=stream
        ),
    ]
    sent: list[dict[str, Any]] = []

    def make(**options: Any) -> Anthropic:
        return client_for(sent, answers)

    monkeypatch.setattr("askkit.cli.Anthropic", make)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    with pytest.raises(SystemExit) as stopped:
        main(["ask", QUESTION])
    assert stopped.value.code == (
        "askkit: cancelled; the text above is not a whole answer"
    )
    assert capsys.readouterr().out.splitlines() == [
        "Tool:          calculate: 0.175908",
        "Today",
        "",
        "Model:         claude-haiku-5-5",
        "Tokens:        618 in, 50 out",
        "Cost:          $0.000087",
        "Time:          no text, total 0.42 s",
        "Time:          first token 0.42 s, cancelled at 0.84 s",
    ]


def test_the_history_you_pass_is_not_changed() -> None:
    sent: list[dict[str, Any]] = []
    history: list[MessageParam] = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hello! How can I help?"},
    ]
    kept = deepcopy(history)
    answers = [streamed(body) for body in two_sums()]
    loop_with(client_for(sent, answers), history)
    assert history == kept
    assert sent[2]["messages"][:2] == kept
