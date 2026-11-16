import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic, transform_schema
from pydantic import ValidationError

from askkit.cli import main
from askkit.extract import (
    TICKET_SCHEMA,
    WISHES,
    Ticket,
    check_ticket,
    extract_ticket,
    repair_ticket,
    retry_ticket,
    ticket_error,
    ticket_faults,
)
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


def use_fake_answers(
    monkeypatch: pytest.MonkeyPatch,
    sent: list[dict[str, Any]],
    bodies: list[dict[str, Any]],
) -> None:
    # One recorded response per call, in order, for a command that asks
    # more than once.
    def answer(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=bodies[len(sent) - 1])

    def make(api_key: str) -> Anthropic:
        transport = httpx2.MockTransport(answer)
        return Anthropic(
            api_key=api_key, http_client=httpx2.Client(transport=transport)
        )

    monkeypatch.setattr("askkit.cli.Anthropic", make)


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
    assert {"type": "null"} in properties["order"]["anyOf"]
    assert "not stated" in properties["wish"]["enum"]


def test_the_schema_sent_is_made_from_ticket() -> None:
    # One class for both sides: the schema the API holds the answer to
    # and the check in our code cannot drift apart.
    assert TICKET_SCHEMA == transform_schema(Ticket)
    assert TICKET_SCHEMA["properties"]["order"]["anyOf"] == [
        {"type": "integer"},
        {"type": "null"},
    ]
    assert TICKET_SCHEMA["properties"]["problem"]["type"] == "string"
    assert TICKET_SCHEMA["properties"]["wish"]["enum"] == [
        "replacement",
        "refund",
        "not stated",
    ]


def test_ticket_reads_a_finished_answer() -> None:
    ticket = Ticket.model_validate_json(
        '{"order":4417,"problem":"mug broken","wish":"replacement"}'
    )
    assert ticket == Ticket(
        order=4417, problem="mug broken", wish="replacement"
    )


def test_ticket_rejects_a_field_it_does_not_name() -> None:
    with pytest.raises(ValidationError) as rejected:
        Ticket.model_validate_json(
            '{"order":4417,"problem":"mug broken","wish":"refund",'
            '"urgent":true}'
        )
    assert ticket_error(rejected.value) == (
        "the ticket does not match its model:"
        " urgent: Extra inputs are not permitted, not True"
    )


def test_ticket_error_names_every_field_that_broke() -> None:
    with pytest.raises(ValidationError) as rejected:
        Ticket.model_validate_json(
            '{"order":"Order 4417","problem":"mug broken",'
            '"wish":"a replacement"}'
        )
    assert ticket_error(rejected.value) == (
        "the ticket does not match its model:"
        " order: Input should be a valid integer, unable to parse string"
        " as an integer, not 'Order 4417';"
        " wish: Input should be 'replacement', 'refund' or 'not stated',"
        " not 'a replacement'"
    )


def test_ticket_error_does_not_print_the_whole_answer_for_a_gap() -> None:
    with pytest.raises(ValidationError) as rejected:
        Ticket.model_validate_json('{"order":4417,"problem":"mug broken"}')
    assert ticket_error(rejected.value) == (
        "the ticket does not match its model: wish: Field required"
    )


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


def test_a_wish_in_other_capitals_is_repaired(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The API holds the shape but not the capitals of an allowed value,
    # so "not Stated" can come back with end_turn. It has one right
    # reading, so it is repaired in code, for free, and said so.
    sent: list[dict[str, Any]] = []
    text = '{"order":null,"problem":"lamp flickers","wish":"not Stated"}'
    use_fake_client(monkeypatch, sent, with_answer(text, "end_turn"))
    main(["extract", str(email_file)])
    assert capsys.readouterr().out.splitlines() == [
        "{",
        '  "order": null,',
        '  "problem": "lamp flickers",',
        '  "wish": "not stated"',
        "}",
        "",
        "Model:         claude-haiku-5-5",
        "Tokens:        236 in, 31 out",
        "Cost:          $0.000039",
        "Stop reason:   end_turn",
        "Repaired:      wish 'not Stated' to 'not stated'",
    ]
    assert len(sent) == 1


def test_an_order_number_written_as_text_comes_out_a_number(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Pydantic's default (lax) mode turns text of digits into an int.
    text = '{"order":"4417","problem":"mug broken","wish":"replacement"}'
    use_fake_client(monkeypatch, [], with_answer(text, "end_turn"))
    main(["extract", str(email_file)])
    assert capsys.readouterr().out.splitlines()[1] == '  "order": 4417,'


def test_ticket_error_names_the_answer_when_it_is_not_a_ticket() -> None:
    # Valid JSON that is not an object fails as a whole: Pydantic gives
    # the problem no field ("loc" is empty), and the message must still
    # be one line, not an IndexError.
    with pytest.raises(ValidationError) as rejected:
        Ticket.model_validate_json("[]")
    assert ticket_error(rejected.value) == (
        "the ticket does not match its model:"
        " answer: Input should be an object, not []"
    )


LONG = "desk lamp flickers when it is switched on"
TOO_LONG = (
    '{"order":null,"problem":"desk lamp flickers when it is switched on",'
    '"wish":"not stated"}'
)
SHORT = '{"order":null,"problem":"lamp flickers","wish":"not stated"}'
TOO_LONG_FAULT = (
    "problem: String should have at most 40 characters,"
    " not 'desk lamp flickers when it is switched on'"
)


def test_the_wishes_are_the_ones_ticket_allows() -> None:
    assert WISHES == ("replacement", "refund", "not stated")
    assert TICKET_SCHEMA["properties"]["wish"]["enum"] == list(WISHES)


def test_the_problem_length_is_checked_here_not_by_the_api() -> None:
    # The API takes no maxLength: the SDK moves it into the description,
    # so the model is told, and only Ticket holds the answer to it.
    problem = TICKET_SCHEMA["properties"]["problem"]
    assert "maxLength" not in problem
    assert problem["description"] == "{maxLength: 40}"
    assert len(LONG) == 41
    with pytest.raises(ValidationError) as rejected:
        Ticket.model_validate_json(TOO_LONG)
    assert ticket_faults(rejected.value) == TOO_LONG_FAULT


def test_repair_changes_only_the_capitals_of_an_allowed_wish() -> None:
    assert repair_ticket(
        '{"order":4417,"problem":"mug broken","wish":"Refund"}'
    ) == (
        '{"order": 4417, "problem": "mug broken", "wish": "refund"}',
        "wish 'Refund' to 'refund'",
    )


def test_repair_never_guesses_a_wish() -> None:
    # "money back" probably means a refund, but probably is a guess, and
    # a guess is the model's job: it is asked again instead.
    assert (
        repair_ticket(
            '{"order":4417,"problem":"mug broken","wish":"money back"}'
        )
        is None
    )


def test_repair_leaves_what_it_cannot_read() -> None:
    assert repair_ticket('{"order":4417,"problem":"mug bro') is None
    assert repair_ticket("[]") is None
    assert repair_ticket('{"order":4417,"problem":"mug broken"}') is None
    assert repair_ticket(SHORT) is None


def test_a_repaired_answer_is_checked_again_in_full() -> None:
    text = TOO_LONG.replace('"not stated"', '"not Stated"')
    with pytest.raises(ValidationError) as rejected:
        check_ticket(text)
    assert ticket_faults(rejected.value) == TOO_LONG_FAULT


def test_check_ticket_says_nothing_was_repaired_for_a_good_answer() -> None:
    ticket, change = check_ticket(SHORT)
    assert ticket.problem == "lamp flickers"
    assert change is None


def test_retry_ticket_sends_the_answer_and_its_faults() -> None:
    sent: list[dict[str, Any]] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=RECORDED)

    client = Anthropic(
        api_key="sk-ant-api03-a-test-key-Ab12",
        http_client=httpx2.Client(transport=httpx2.MockTransport(answer)),
    )
    retry_ticket(
        client, EMAIL, TOO_LONG, TOO_LONG_FAULT, "claude-haiku-5-5", 1000
    )
    assert sent[0]["messages"] == [
        {"role": "user", "content": EMAIL},
        {"role": "assistant", "content": TOO_LONG},
        {
            "role": "user",
            "content": render("retry_ticket", {"faults": TOO_LONG_FAULT}),
        },
    ]
    assert sent[0]["system"] == render("extract_ticket", {})
    assert sent[0]["output_config"]["format"]["schema"] == TICKET_SCHEMA


def test_a_broken_ticket_is_asked_for_again_with_its_faults(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sent: list[dict[str, Any]] = []
    second = with_answer(SHORT, "end_turn")
    second["usage"] = {"input_tokens": 312, "output_tokens": 29}
    use_fake_answers(
        monkeypatch, sent, [with_answer(TOO_LONG, "end_turn"), second]
    )
    main(["extract", str(email_file)])
    assert capsys.readouterr().out.splitlines() == [
        "{",
        '  "order": null,',
        '  "problem": "lamp flickers",',
        '  "wish": "not stated"',
        "}",
        "",
        "Model:         claude-haiku-5-5",
        "Tokens:        548 in, 60 out",
        "Cost:          $0.000085",
        "Stop reason:   end_turn",
        f"Asked again:   {TOO_LONG_FAULT}",
    ]
    assert len(sent) == 2
    assert sent[1]["messages"][1] == {
        "role": "assistant",
        "content": TOO_LONG,
    }


def test_a_ticket_broken_twice_goes_to_a_person(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # One retry, then exit code 1: never a loop that pays for answers
    # until one happens to pass.
    sent: list[dict[str, Any]] = []
    broken = with_answer(TOO_LONG, "end_turn")
    # The retry sends the first answer and its faults too: more input.
    broken_again = with_answer(TOO_LONG, "end_turn")
    broken_again["usage"] = {"input_tokens": 312, "output_tokens": 31}
    use_fake_answers(monkeypatch, sent, [broken, broken_again, broken])
    with pytest.raises(SystemExit) as stopped:
        main(["extract", str(email_file)])
    assert stopped.value.code == (
        f"askkit: the ticket does not match its model: {TOO_LONG_FAULT}"
    )
    assert len(sent) == 2
    assert capsys.readouterr().out.splitlines() == [
        "Model:         claude-haiku-5-5",
        "Tokens:        548 in, 62 out",
        "Cost:          $0.000086",
        "Stop reason:   end_turn",
        f"Asked again:   {TOO_LONG_FAULT}",
    ]


def test_retries_0_never_asks_again(
    email_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []
    broken = with_answer(TOO_LONG, "end_turn")
    use_fake_answers(monkeypatch, sent, [broken, broken])
    with pytest.raises(SystemExit):
        main(["extract", str(email_file), "--retries", "0"])
    assert len(sent) == 1


def test_retries_below_zero_are_refused(email_file: Path) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["extract", str(email_file), "--retries", "-1"])
    assert stopped.value.code == (
        "askkit: --retries -1 is below zero; give 0 to never ask again"
    )


def test_a_cut_off_answer_is_not_asked_for_again(
    email_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same room cuts the same answer off: only a larger --max-tokens
    # helps, and how much more to pay is the person's choice.
    sent: list[dict[str, Any]] = []
    cut = with_answer('{"order":4417,"problem":"mug bro', "max_tokens")
    use_fake_answers(monkeypatch, sent, [cut, cut])
    with pytest.raises(SystemExit):
        main(["extract", str(email_file)])
    assert len(sent) == 1
