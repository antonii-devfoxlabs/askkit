import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic

from askkit.cli import main
from askkit.extract import TICKET_SCHEMA, extract_ticket
from askkit.prompts import render

EMAIL = (
    "Hello, order 4417 came yesterday and the mug inside was broken."
    " I'd like a replacement, not a refund. Thanks, Maria"
)

# A response shaped like the Messages API's to a call with a JSON schema:
# the JSON is the text block, after the model's thinking.
RECORDED = {
    "id": "msg_01AskKitExtract",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [
        {"type": "thinking", "thinking": "", "signature": "test"},
        {
            "type": "text",
            "text": (
                '{"order":4417,"problem":"mug broken","wish":"replacement"}'
            ),
        },
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 236, "output_tokens": 31},
}


def with_answer(text: str, stop_reason: str) -> dict[str, Any]:
    body = dict(RECORDED)
    body["content"] = [{"type": "text", "text": text}]
    body["stop_reason"] = stop_reason
    return body


def use_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    sent: list[dict[str, Any]],
    body: dict[str, Any] = RECORDED,
) -> None:
    def answer(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=body)

    def make(api_key: str) -> Anthropic:
        transport = httpx2.MockTransport(answer)
        return Anthropic(
            api_key=api_key, http_client=httpx2.Client(transport=transport)
        )

    monkeypatch.setattr("askkit.cli.Anthropic", make)


@pytest.fixture
def email_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    path = tmp_path / "email.txt"
    path.write_text(EMAIL, encoding="utf-8")
    return path


def test_extract_ticket_sends_the_schema_beside_the_email() -> None:
    sent: list[dict[str, Any]] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=RECORDED)

    client = Anthropic(
        api_key="sk-ant-api03-a-test-key-Ab12",
        http_client=httpx2.Client(transport=httpx2.MockTransport(answer)),
    )
    extract_ticket(client, EMAIL, "claude-haiku-5-5", 1000)
    assert sent == [
        {
            "model": "claude-haiku-5-5",
            "max_tokens": 1000,
            "system": render("extract_ticket", {}),
            "messages": [{"role": "user", "content": EMAIL}],
            "output_config": {
                "format": {"type": "json_schema", "schema": TICKET_SCHEMA}
            },
        }
    ]


def test_the_schema_is_one_the_api_accepts() -> None:
    # Anthropic refuses a JSON schema whose object allows other keys, and
    # every optional field counts toward a limit: all three are required.
    assert TICKET_SCHEMA["additionalProperties"] is False
    assert TICKET_SCHEMA["required"] == list(TICKET_SCHEMA["properties"])


def test_every_field_has_a_way_out() -> None:
    # Without one, an email with no order number forces a made-up number.
    properties = TICKET_SCHEMA["properties"]
    assert "null" in properties["order"]["type"]
    assert "not stated" in properties["wish"]["enum"]


def test_extract_prints_the_ticket_and_the_call(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    main(["extract", str(email_file)])
    assert capsys.readouterr().out.splitlines() == [
        "{",
        '  "order": 4417,',
        '  "problem": "mug broken",',
        '  "wish": "replacement"',
        "}",
        "",
        "Model:         claude-haiku-5-5",
        "Tokens:        236 in, 31 out",
        "Cost:          $0.000039",
        "Stop reason:   end_turn",
    ]
    assert sent[0]["messages"][0]["content"] == EMAIL


def test_extract_prints_null_for_no_order_number(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = '{"order":null,"problem":"shade wrong colour","wish":"not stated"}'
    use_fake_client(monkeypatch, [], with_answer(text, "end_turn"))
    main(["extract", str(email_file)])
    assert capsys.readouterr().out.splitlines()[:5] == [
        "{",
        '  "order": null,',
        '  "problem": "shade wrong colour",',
        '  "wish": "not stated"',
        "}",
    ]


def test_a_cut_off_answer_is_not_parsed(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Parsed, this half of a ticket raises JSONDecodeError: the stop
    # reason has to be read before the JSON.
    cut = '{"order":4417,"problem":"mug bro'
    use_fake_client(monkeypatch, [], with_answer(cut, "max_tokens"))
    with pytest.raises(SystemExit) as stopped:
        main(["extract", str(email_file), "--max-tokens", "12"])
    assert stopped.value.code == (
        "askkit: the answer was cut off at 31 tokens;"
        " give it a larger --max-tokens"
    )
    assert capsys.readouterr().out.splitlines() == [
        "Model:         claude-haiku-5-5",
        "Tokens:        236 in, 31 out",
        "Cost:          $0.000039",
        "Stop reason:   max_tokens",
    ]


def test_a_declined_answer_is_not_parsed(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    body = with_answer("I can't help with that.", "refusal")
    use_fake_client(monkeypatch, [], body)
    with pytest.raises(SystemExit) as stopped:
        main(["extract", str(email_file)])
    assert stopped.value.code == (
        "askkit: the model declined to answer this question"
    )
    assert "I can't" not in capsys.readouterr().out


def test_extract_takes_max_tokens(
    email_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    main(["extract", str(email_file), "--max-tokens", "300"])
    assert sent[0]["model"] == "claude-haiku-5-5"
    assert sent[0]["max_tokens"] == 300
