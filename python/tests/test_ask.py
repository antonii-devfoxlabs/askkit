import json
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic

from askkit.ask import answer_text, ask
from askkit.cli import main

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


def test_ask_sends_the_question_as_the_one_user_message() -> None:
    sent: list[dict[str, Any]] = []
    ask(fake_client(sent), "Where is my order?", "claude-haiku-5-5", 1000)
    assert sent == [
        {
            "model": "claude-haiku-5-5",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": "Where is my order?"}],
        }
    ]


def test_answer_text_skips_the_thinking_block() -> None:
    response = ask(fake_client([]), "Hi", "claude-haiku-5-5", 1000)
    assert answer_text(response) == "Order 4417: a replacement mug."


def test_answer_text_joins_every_text_block() -> None:
    body = dict(RECORDED)
    body["content"] = [
        {"type": "text", "text": "Order 4417: "},
        {"type": "text", "text": "a replacement mug."},
    ]
    response = ask(fake_client([], body), "Hi", "claude-haiku-5-5", 1000)
    assert answer_text(response) == "Order 4417: a replacement mug."


def use_fake_client(
    monkeypatch: pytest.MonkeyPatch, sent: list[dict[str, Any]]
) -> None:
    def make(**options: Any) -> Anthropic:
        return fake_client(sent)

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
    ]
    assert sent[0]["max_tokens"] == 1000


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
    assert lines[-1] == "Cost:          $0.000025"


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
