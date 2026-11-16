import json
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic

from askkit.ask import answer_text, ask, stop_warning
from askkit.cli import main
from askkit.prompts import ASK_SYSTEM

# A response shaped like the Messages API's, recorded for the tests:
# Haiku 5.5 may think first, so the text is not always the first block.
RECORDED = {
    "id": "msg_01AskKitTest",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [
        {"type": "thinking", "thinking": "", "signature": "test"},
        {"type": "text", "text": "Order 4417: a replacement mug."},
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 118, "output_tokens": 27},
}


def fake_client(
    sent: list[dict[str, Any]], body: dict[str, Any] = RECORDED
) -> Anthropic:
    def answer(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=body)

    transport = httpx2.MockTransport(answer)
    return Anthropic(
        api_key="sk-ant-api03-a-test-key-Ab12",
        http_client=httpx2.Client(transport=transport),
    )


def test_ask_sends_the_system_prompt_and_the_question() -> None:
    sent: list[dict[str, Any]] = []
    ask(
        fake_client(sent),
        "Where is my order?",
        "claude-haiku-5-5",
        1000,
        "Answer in one sentence.",
    )
    assert sent == [
        {
            "model": "claude-haiku-5-5",
            "max_tokens": 1000,
            "system": "Answer in one sentence.",
            "messages": [{"role": "user", "content": "Where is my order?"}],
        }
    ]


def test_answer_text_skips_the_thinking_block() -> None:
    response = ask(
        fake_client([]), "Hi", "claude-haiku-5-5", 1000, "Be brief."
    )
    assert answer_text(response) == "Order 4417: a replacement mug."


def test_answer_text_joins_every_text_block() -> None:
    body = dict(RECORDED)
    body["content"] = [
        {"type": "text", "text": "Order 4417: "},
        {"type": "text", "text": "a replacement mug."},
    ]
    response = ask(
        fake_client([], body), "Hi", "claude-haiku-5-5", 1000, "Be brief."
    )
    assert answer_text(response) == "Order 4417: a replacement mug."


def use_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    sent: list[dict[str, Any]],
    body: dict[str, Any] = RECORDED,
) -> None:
    def make(**options: Any) -> Anthropic:
        return fake_client(sent, body)

    monkeypatch.setattr("askkit.cli.Anthropic", make)


def test_ask_prints_the_answer_and_its_cost(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    main(["ask", "Where is my order?"])
    assert capsys.readouterr().out.splitlines() == [
        "Order 4417: a replacement mug.",
        "",
        "Model:         claude-haiku-5-5",
        "Tokens:        118 in, 27 out",
        "Cost:          $0.000025",
        "Stop reason:   end_turn",
    ]
    assert sent[0]["max_tokens"] == 1000
    assert sent[0]["system"] == ASK_SYSTEM


def test_ask_takes_the_model_and_max_tokens(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    main(
        [
            "ask",
            "Hi",
            "-m",
            "claude-haiku-5-5",
            "--max-tokens",
            "200",
        ]
    )
    assert sent[0]["model"] == "claude-haiku-5-5"
    assert sent[0]["max_tokens"] == 200
    lines = capsys.readouterr().out.splitlines()
    assert lines[-2] == "Cost:          $0.000025"


@pytest.mark.parametrize(
    "model", ["gemini-3.5-flash-lite", "gpt-6-luna", "qwen3.8-27b"]
)
def test_ask_refuses_a_model_anthropic_does_not_serve(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    model: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    with pytest.raises(SystemExit):
        main(["ask", "Hi", "-m", model])
    assert "invalid choice" in capsys.readouterr().err
    assert sent == []


def test_ask_without_a_key_is_refused_before_any_call(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    with pytest.raises(SystemExit) as stopped:
        main(["ask", "Hi"])
    assert stopped.value.code == (
        "askkit: ANTHROPIC_API_KEY is not set; copy .env.example to .env"
        " and put your key after the ="
    )
    assert sent == []


def test_ask_reads_the_system_prompt_from_a_file(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    (tmp_path / "ticket.txt").write_text(
        "You turn a customer's messages into a support ticket. Be brief.\n"
        "If the messages give no order number, write: no order number.\n",
        encoding="utf-8",
    )
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    main(["ask", "My lamp flickers.", "--system-file", "ticket.txt"])
    assert sent[0]["system"] == (
        "You turn a customer's messages into a support ticket. Be brief.\n"
        "If the messages give no order number, write: no order number."
    )
    assert sent[0]["messages"] == [
        {"role": "user", "content": "My lamp flickers."}
    ]


def test_the_system_prompt_sets_the_job_and_the_rules() -> None:
    assert ASK_SYSTEM.startswith("You answer questions for AskKit")
    assert "plain text" in ASK_SYSTEM
    assert "120 words" in ASK_SYSTEM
    assert ASK_SYSTEM.endswith("write exactly: I don't know.")


def stopped_with(reason: str) -> dict[str, Any]:
    body = dict(RECORDED)
    body["stop_reason"] = reason
    body["usage"] = {"input_tokens": 118, "output_tokens": 60}
    return body


@pytest.mark.parametrize("reason", ["end_turn", "stop_sequence"])
def test_a_finished_answer_has_no_warning(reason: str) -> None:
    response = ask(
        fake_client([], stopped_with(reason)),
        "Hi",
        "claude-haiku-5-5",
        60,
        "Be brief.",
    )
    assert stop_warning(response) is None


@pytest.mark.parametrize(
    ("reason", "warning"),
    [
        (
            "max_tokens",
            "the answer was cut off at 60 tokens;"
            " give it a larger --max-tokens",
        ),
        (
            "model_context_window_exceeded",
            "the answer was cut off at the end of the context window",
        ),
        ("refusal", "the model declined to answer this question"),
        ("tool_use", "the answer stopped early: tool_use"),
        ("pause_turn", "the answer stopped early: pause_turn"),
    ],
)
def test_an_answer_that_is_not_whole_has_a_warning(
    reason: str, warning: str
) -> None:
    response = ask(
        fake_client([], stopped_with(reason)),
        "Hi",
        "claude-haiku-5-5",
        60,
        "Be brief.",
    )
    assert stop_warning(response) == warning


def test_ask_stops_with_a_warning_when_the_answer_is_cut_off(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    use_fake_client(monkeypatch, [], stopped_with("max_tokens"))
    with pytest.raises(SystemExit) as stopped:
        main(["ask", "Hi", "--max-tokens", "60"])
    assert stopped.value.code == (
        "askkit: the answer was cut off at 60 tokens;"
        " give it a larger --max-tokens"
    )
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Order 4417: a replacement mug."
    assert lines[-1] == "Stop reason:   max_tokens"


def test_a_refusal_names_its_category() -> None:
    body = stopped_with("refusal")
    body["stop_details"] = {
        "type": "refusal",
        "category": "cyber",
        "explanation": None,
    }
    response = ask(
        fake_client([], body), "Hi", "claude-haiku-5-5", 60, "Be brief."
    )
    assert stop_warning(response) == (
        "the model declined to answer this question (category: cyber)"
    )


def test_ask_prints_no_part_of_a_declined_answer(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    use_fake_client(monkeypatch, [], stopped_with("refusal"))
    with pytest.raises(SystemExit) as stopped:
        main(["ask", "Hi"])
    assert stopped.value.code == (
        "askkit: the model declined to answer this question"
    )
    assert capsys.readouterr().out.splitlines() == [
        "Model:         claude-haiku-5-5",
        "Tokens:        118 in, 60 out",
        "Cost:          $0.000042",
        "Stop reason:   refusal",
    ]
