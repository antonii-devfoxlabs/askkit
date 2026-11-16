import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic
from pydantic import ValidationError

from askkit.classify import (
    CLASSIFICATION_SCHEMA,
    QUEUES,
    check_classification,
    classification_error,
    classify_email,
    repair_queue,
)
from askkit.cli import main
from askkit.prompts import render

GIFT = (
    "Hello, do you sell gift cards? I would like to give one to a friend"
    " for her birthday."
)

# A response shaped like the Messages API's to a call with a JSON schema,
# written for these tests: the JSON is the text block.
ANSWER = {
    "id": "msg_01AskKitClassify",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": '{"queue":"other"}'}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 214, "output_tokens": 12},
}


def with_answer(text: str, stop_reason: str = "end_turn") -> dict[str, Any]:
    body = dict(ANSWER)
    body["content"] = [{"type": "text", "text": text}]
    body["stop_reason"] = stop_reason
    return body


def use_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    sent: list[dict[str, Any]],
    body: dict[str, Any] = ANSWER,
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
    path = tmp_path / "gift.txt"
    path.write_text(GIFT, encoding="utf-8")
    return path


def test_the_schema_allows_the_four_queues_and_nothing_else() -> None:
    assert CLASSIFICATION_SCHEMA == {
        "type": "object",
        "title": "Classification",
        "properties": {
            "queue": {
                "enum": ["refund", "damaged", "delivery", "other"],
                "title": "Queue",
            }
        },
        "required": ["queue"],
        "additionalProperties": False,
    }
    assert QUEUES == ("refund", "damaged", "delivery", "other")


def test_every_queue_is_defined_in_the_template() -> None:
    # The model sees the queues twice: as the schema's values and as the
    # template's definitions. A queue added to Queue without a definition
    # would leave the model guessing what it means.
    lines = render("classify_email", {}).splitlines()
    for queue in QUEUES:
        assert any(line.startswith(f"- {queue}: ") for line in lines), queue


def test_classify_email_sends_the_schema_beside_the_email() -> None:
    sent: list[dict[str, Any]] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=ANSWER)

    client = Anthropic(
        api_key="sk-ant-api03-a-test-key-Ab12",
        http_client=httpx2.Client(transport=httpx2.MockTransport(answer)),
    )
    classify_email(client, GIFT, "claude-haiku-5-5", 1000)
    assert sent == [
        {
            "model": "claude-haiku-5-5",
            "max_tokens": 1000,
            "system": render("classify_email", {}),
            "messages": [{"role": "user", "content": GIFT}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": CLASSIFICATION_SCHEMA,
                }
            },
        }
    ]


def test_capitals_of_an_allowed_queue_are_repaired() -> None:
    assert repair_queue('{"queue":"Other"}') == (
        '{"queue": "other"}',
        "queue 'Other' to 'other'",
    )
    classification, change = check_classification('{"queue":"Damaged"}')
    assert classification.queue == "damaged"
    assert change == "queue 'Damaged' to 'damaged'"


def test_nothing_but_capitals_is_repaired() -> None:
    assert repair_queue('{"queue":"other"}') is None
    assert repair_queue('{"queue":"gift cards"}') is None
    assert repair_queue('{"queue":"Others"}') is None
    assert repair_queue('["other"]') is None
    assert repair_queue("other") is None


def test_a_queue_not_on_the_list_is_stopped() -> None:
    with pytest.raises(ValidationError) as raised:
        check_classification('{"queue":"gift cards"}')
    assert classification_error(raised.value) == (
        "the queue does not match its model: queue: Input should be"
        " 'refund', 'damaged', 'delivery' or 'other', not 'gift cards'"
    )


def test_classify_prints_the_queue_and_the_call(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    main(["classify", str(email_file)])
    assert capsys.readouterr().out.splitlines() == [
        "{",
        '  "queue": "other"',
        "}",
        "",
        "Model:         claude-haiku-5-5",
        "Tokens:        214 in, 12 out",
        "Cost:          $0.000027",
        "Stop reason:   end_turn",
    ]
    assert sent[0]["messages"][0]["content"] == GIFT


def test_classify_says_what_it_repaired(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    use_fake_client(monkeypatch, [], with_answer('{"queue":"Other"}'))
    main(["classify", str(email_file)])
    out = capsys.readouterr().out.splitlines()
    assert out[1] == '  "queue": "other"'
    assert out[-1] == "Repaired:      queue 'Other' to 'other'"


def test_classify_reads_the_cleaned_email(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The quoted dispatch email says "sent": left in, its words could pull
    # a gift-card question into the delivery queue.
    email_file.write_text(
        GIFT + "\n\nOn 3 Nov 2026, the shop wrote:\n> Your order was sent.",
        encoding="utf-8",
    )
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    main(["classify", str(email_file)])
    assert sent[0]["messages"][0]["content"] == GIFT
    out = capsys.readouterr().out.splitlines()
    assert out[-1] == "Left out:      2 quoted lines"


def test_a_cut_off_answer_is_not_read(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    body = with_answer('{"queue":"oth', "max_tokens")
    use_fake_client(monkeypatch, [], body)
    with pytest.raises(SystemExit) as stopped:
        main(["classify", str(email_file), "--max-tokens", "12"])
    assert stopped.value.code == (
        "askkit: the answer was cut off at 12 tokens;"
        " give it a larger --max-tokens"
    )
    assert capsys.readouterr().out.splitlines()[-1] == (
        "Stop reason:   max_tokens"
    )


def test_a_declined_email_names_what_was_declined(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    body = with_answer("", "refusal")
    body["stop_details"] = {
        "type": "refusal",
        "category": None,
        "explanation": None,
    }
    use_fake_client(monkeypatch, [], body)
    with pytest.raises(SystemExit) as stopped:
        main(["classify", str(email_file)])
    assert stopped.value.code == (
        "askkit: the model declined to sort this email into a queue"
    )
    assert capsys.readouterr().out.splitlines()[-1] == (
        "Stop reason:   refusal"
    )


def test_a_queue_not_on_the_list_goes_to_a_person(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent, with_answer('{"queue":"gifts"}'))
    with pytest.raises(SystemExit) as stopped:
        main(["classify", str(email_file)])
    assert stopped.value.code == (
        "askkit: the queue does not match its model: queue: Input should"
        " be 'refund', 'damaged', 'delivery' or 'other', not 'gifts'"
    )
    assert len(sent) == 1
    assert '"queue"' not in capsys.readouterr().out
