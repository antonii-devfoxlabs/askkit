import json
from pathlib import Path
from typing import Any, Self

import httpx2
import pytest
from anthropic import Anthropic, transform_schema
from pydantic import BaseModel, ValidationError, model_validator
from test_ask import Interrupted, cancelling_client, replied
from test_label import printed_json

from askkit.ask import CANCELLED
from askkit.cli import main
from askkit.extract import (
    REPAIRED,
    TICKET_CHECK,
    TICKET_SCHEMA,
    WISHES,
    Ticket,
    check_ticket,
    extract_ticket,
    fault_place,
    not_stated,
    ticket_error,
    ticket_faults,
    ticket_messages,
)
from askkit.label import LABEL_KEY
from askkit.prompts import render
from askkit.rules import EMAIL_CHECK, ORDERS_CHECK

EMAIL = (
    "Hello, order 4417 came yesterday and the mug inside was broken."
    " I'd like a replacement, not a refund. Thanks, Maria"
)

# The ticket of the lamp emails, which give no order and no wish.
LAMP_TICKET = {
    "order": None,
    "problem": "lamp flickers",
    "wish": "not stated",
}

# Every ticket of this file is labelled as written at one fixed moment.
pytestmark = pytest.mark.usefixtures("labelled_now")

# A response shaped like the Messages API's to a call with a JSON schema:
# the JSON is the text block, after the model's thinking.
WRITTEN = {
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
    body = dict(WRITTEN)
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
        return replied(request, bodies[len(sent) - 1])

    def make(api_key: str) -> Anthropic:
        transport = httpx2.MockTransport(answer)
        return Anthropic(
            api_key=api_key, http_client=httpx2.Client(transport=transport)
        )

    monkeypatch.setattr("askkit.cli.Anthropic", make)


def use_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    sent: list[dict[str, Any]],
    body: dict[str, Any] = WRITTEN,
) -> None:
    def answer(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return replied(request, body)

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
        return httpx2.Response(200, json=WRITTEN)

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
    ticket, lines = printed_json(capsys.readouterr().out)
    assert ticket == {
        "order": 4417,
        "problem": "mug broken",
        "wish": "replacement",
        LABEL_KEY: {
            "model": "claude-haiku-5-5",
            "via": "askkit extract",
            "written": "2026-11-21T09:30:00Z",
            "checked_by_code": [TICKET_CHECK, EMAIL_CHECK],
            "read_by_a_person": False,
        },
    }
    assert lines == [
        "",
        "Model:         claude-haiku-5-5",
        "Tokens:        236 in, 31 out",
        "Cost:          $0.000039",
        "Stop reason:   end_turn",
        "Time:          first token 0.42 s, total 0.84 s",
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
    out = capsys.readouterr().out
    assert out.splitlines()[:4] == [
        "{",
        '  "order": null,',
        '  "problem": "shade wrong colour",',
        '  "wish": "not stated",',
    ]
    # No order, so no rule about the order checked anything, and the
    # label does not say one did.
    ticket, _ = printed_json(out)
    assert ticket[LABEL_KEY]["checked_by_code"] == [TICKET_CHECK]


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
        "Time:          first token 0.42 s, total 0.84 s",
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
        "askkit: the model declined to turn this email into a ticket"
    )
    out = capsys.readouterr().out
    assert "I can't" not in out
    assert out.splitlines()[-2] == "Stop reason:   refusal"


def test_a_declined_ticket_says_why_in_the_apis_words(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A refusal can come after part of the answer. That part is never
    # read, however much of a ticket it looks like: the person gets the
    # category and the explanation instead.
    body = with_answer('{"order":8125,"problem":"account', "refusal")
    # Declined after nine tokens of a ticket: billed for those, so the
    # cost line counts them.
    body["usage"] = {"input_tokens": 251, "output_tokens": 9}
    body["stop_details"] = {
        "type": "refusal",
        "category": "cyber",
        "explanation": (
            "This request was declined because it could enable cyber harm."
        ),
    }
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent, body)
    with pytest.raises(SystemExit) as stopped:
        main(["extract", str(email_file)])
    assert stopped.value.code == (
        "askkit: the model declined to turn this email into a ticket"
        " (category: cyber). This request was declined because it could"
        " enable cyber harm."
    )
    # The part of a ticket written before the decline is not printed:
    # only the lines about the call.
    assert capsys.readouterr().out.splitlines() == [
        "Model:         claude-haiku-5-5",
        "Tokens:        251 in, 9 out",
        "Cost:          $0.000030",
        "Stop reason:   refusal",
        "Time:          first token 0.42 s, total 0.84 s",
    ]
    # The same email to the same model is declined again: no retry.
    assert len(sent) == 1


def test_not_stated_lists_the_fields_the_email_does_not_give() -> None:
    assert not_stated(
        Ticket(order=None, problem="lamp flickers", wish="not stated")
    ) == ["order", "wish"]
    assert not_stated(
        Ticket(order=6083, problem="shade wrong colour", wish="not stated")
    ) == ["wish"]
    assert (
        not_stated(Ticket(order=4417, problem="mug broken", wish="refund"))
        == []
    )


def test_a_ticket_with_a_way_out_says_what_to_ask_for(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A null order or "not stated" is a true answer, the email does not
    # say, so the ticket is printed; the note tells the person who acts on
    # it what to ask the customer.
    text = '{"order":null,"problem":"lamp flickers","wish":"not stated"}'
    use_fake_client(monkeypatch, [], with_answer(text, "end_turn"))
    main(["extract", str(email_file)])
    ticket, lines = printed_json(capsys.readouterr().out)
    del ticket[LABEL_KEY]
    assert ticket == LAMP_TICKET
    assert lines == [
        "",
        "Model:         claude-haiku-5-5",
        "Tokens:        236 in, 31 out",
        "Cost:          $0.000039",
        "Stop reason:   end_turn",
        "Time:          first token 0.42 s, total 0.84 s",
        "Not stated:    order, wish",
    ]


def test_a_whole_ticket_has_no_not_stated_note(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    use_fake_client(monkeypatch, [])
    main(["extract", str(email_file)])
    assert "Not stated:" not in capsys.readouterr().out


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
    ticket, lines = printed_json(capsys.readouterr().out)
    del ticket[LABEL_KEY]
    assert ticket == LAMP_TICKET
    assert lines == [
        "",
        "Model:         claude-haiku-5-5",
        "Tokens:        236 in, 31 out",
        "Cost:          $0.000039",
        "Stop reason:   end_turn",
        "Time:          first token 0.42 s, total 0.84 s",
        "Repaired:      wish 'not Stated' to 'not stated'",
        "Not stated:    order, wish",
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


def test_ticket_itself_repairs_the_capitals_of_an_allowed_wish() -> None:
    # The repair is one of the class's checks, so any loop that checks
    # Ticket gets it, a structured-output library's too: called with no
    # context, as a library may call it, the answer still passes.
    ticket = Ticket.model_validate_json(
        '{"order":4417,"problem":"mug broken","wish":"Refund"}'
    )
    assert ticket.wish == "refund"


def test_the_repair_writes_what_it_changed_into_the_context() -> None:
    notes: dict[str, list[str]] = {}
    Ticket.model_validate_json(
        '{"order":4417,"problem":"mug broken","wish":"not Stated"}',
        context=notes,
    )
    assert notes == {REPAIRED: ["wish 'not Stated' to 'not stated'"]}


def test_check_ticket_names_the_repair() -> None:
    ticket, change = check_ticket(
        '{"order":4417,"problem":"mug broken","wish":"Refund"}'
    )
    assert ticket.wish == "refund"
    assert change == "wish 'Refund' to 'refund'"


def test_repair_never_guesses_a_wish() -> None:
    # "money back" probably means a refund, but probably is a guess, and
    # a guess is the model's job: it is asked again instead.
    notes: dict[str, list[str]] = {}
    with pytest.raises(ValidationError) as rejected:
        Ticket.model_validate_json(
            '{"order":4417,"problem":"mug broken","wish":"money back"}',
            context=notes,
        )
    assert notes == {}
    assert ticket_faults(rejected.value).startswith("wish: Input should be")


def test_repair_leaves_what_it_cannot_read() -> None:
    for text in ['{"order":4417,"problem":"mug bro', "[]", '{"wish":7}']:
        notes: dict[str, list[str]] = {}
        with pytest.raises(ValidationError):
            Ticket.model_validate_json(text, context=notes)
        assert notes == {}


def test_a_context_that_is_not_a_dict_is_left_alone() -> None:
    # A library may pass a context of its own shape: the repair happens,
    # and nothing is written into what is not AskKit's dict.
    ticket = Ticket.model_validate_json(
        '{"order":null,"problem":"lamp flickers","wish":"Refund"}',
        context="the library's own",
    )
    assert ticket.wish == "refund"


def test_the_repair_does_not_change_the_schema_sent() -> None:
    # A before-validator is code, not schema: the request is the same.
    assert TICKET_SCHEMA == transform_schema(Ticket)
    assert set(TICKET_SCHEMA["properties"]) == {"order", "problem", "wish"}


def test_a_repaired_answer_is_checked_again_in_full() -> None:
    text = TOO_LONG.replace('"not stated"', '"not Stated"')
    with pytest.raises(ValidationError) as rejected:
        check_ticket(text)
    assert ticket_faults(rejected.value) == TOO_LONG_FAULT


def test_check_ticket_says_nothing_was_repaired_for_a_good_answer() -> None:
    ticket, change = check_ticket(SHORT)
    assert ticket.problem == "lamp flickers"
    assert change is None


def test_ticket_messages_ask_again_with_the_answer_and_its_faults() -> None:
    assert ticket_messages(EMAIL, TOO_LONG, TOO_LONG_FAULT) == [
        {"role": "user", "content": EMAIL},
        {"role": "assistant", "content": TOO_LONG},
        {
            "role": "user",
            "content": render("retry_ticket", {"faults": TOO_LONG_FAULT}),
        },
    ]


def test_ticket_messages_of_a_first_call_hold_the_email_alone() -> None:
    assert ticket_messages(EMAIL) == [{"role": "user", "content": EMAIL}]


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
    ticket, lines = printed_json(capsys.readouterr().out)
    del ticket[LABEL_KEY]
    assert ticket == LAMP_TICKET
    assert lines == [
        "",
        "Model:         claude-haiku-5-5",
        "Tokens:        548 in, 60 out",
        "Cost:          $0.000085",
        "Stop reason:   end_turn",
        "Time:          first token 0.42 s, total 0.84 s",
        "Time:          first token 0.42 s, total 0.84 s",
        f"Asked again:   {TOO_LONG_FAULT}",
        "Not stated:    order, wish",
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
        "Time:          first token 0.42 s, total 0.84 s",
        "Time:          first token 0.42 s, total 0.84 s",
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


# The business rules, after Ticket (lesson 4.5).
RIGHT = '{"order":4417,"problem":"mug broken","wish":"replacement"}'
# Valid against Ticket in every field, and wrong: Maria's email says 4417.
SWAPPED = '{"order":4471,"problem":"mug broken","wish":"replacement"}'
SWAPPED_FAULT = "order: 4471 is not a number in the email"


@pytest.fixture
def maria(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    (tmp_path / "orders.json").write_text(
        json.dumps({"4417": "mug", "6083": "lamp shade", "7302": "vase"}),
        encoding="utf-8",
    )
    path = tmp_path / "maria.txt"
    path.write_text(EMAIL, encoding="utf-8")
    return path


def test_a_valid_but_wrong_order_is_asked_for_again(
    maria: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # It used to be printed as Maria's ticket, with exit code 0: Ticket
    # passes 4471, a whole number.
    sent: list[dict[str, Any]] = []
    second = with_answer(RIGHT, "end_turn")
    second["usage"] = {"input_tokens": 312, "output_tokens": 31}
    use_fake_answers(
        monkeypatch, sent, [with_answer(SWAPPED, "end_turn"), second]
    )
    main(["extract", str(maria), "--orders", "orders.json"])
    ticket, lines = printed_json(capsys.readouterr().out)
    # --orders ran one more check, and the label names it.
    assert ticket.pop(LABEL_KEY)["checked_by_code"] == [
        TICKET_CHECK,
        EMAIL_CHECK,
        ORDERS_CHECK,
    ]
    assert ticket == {
        "order": 4417,
        "problem": "mug broken",
        "wish": "replacement",
    }
    assert lines == [
        "",
        "Model:         claude-haiku-5-5",
        "Tokens:        548 in, 62 out",
        "Cost:          $0.000086",
        "Stop reason:   end_turn",
        "Time:          first token 0.42 s, total 0.84 s",
        "Time:          first token 0.42 s, total 0.84 s",
        f"Asked again:   {SWAPPED_FAULT}",
    ]
    assert len(sent) == 2
    assert sent[1]["messages"][1] == {"role": "assistant", "content": SWAPPED}


def test_a_number_wrong_twice_goes_to_a_person(
    maria: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []
    swapped = with_answer(SWAPPED, "end_turn")
    use_fake_answers(monkeypatch, sent, [swapped, swapped, swapped])
    with pytest.raises(SystemExit) as stopped:
        main(["extract", str(maria)])
    assert stopped.value.code == (
        f"askkit: the ticket does not match the email: {SWAPPED_FAULT}"
    )
    assert len(sent) == 2


def test_an_order_the_shop_has_is_printed(
    maria: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    use_fake_answers(monkeypatch, [], [with_answer(RIGHT, "end_turn")])
    main(["extract", str(maria), "--orders", "orders.json"])
    assert capsys.readouterr().out.splitlines()[1] == '  "order": 4417,'


def test_an_order_the_shop_lacks_goes_to_a_person_at_once(
    maria: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The email itself says 4471, so asking again returns 4471 again.
    maria.write_text(EMAIL.replace("4417", "4471"), encoding="utf-8")
    sent: list[dict[str, Any]] = []
    swapped = with_answer(SWAPPED, "end_turn")
    use_fake_answers(monkeypatch, sent, [swapped, swapped])
    with pytest.raises(SystemExit) as stopped:
        main(["extract", str(maria), "--orders", "orders.json"])
    assert stopped.value.code == (
        "askkit: order: 4471 is not in the shop's orders;"
        " the email gives this number, so a person checks it"
    )
    assert len(sent) == 1
    assert capsys.readouterr().out.splitlines() == [
        "Model:         claude-haiku-5-5",
        "Tokens:        236 in, 31 out",
        "Cost:          $0.000039",
        "Stop reason:   end_turn",
        "Time:          first token 0.42 s, total 0.84 s",
    ]


def test_another_customers_order_is_asked_for_again(
    maria: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 6083 is a real order, so the shop's orders alone would pass it: only
    # the email rule sees it is not Maria's.
    sent: list[dict[str, Any]] = []
    other = with_answer(SWAPPED.replace("4471", "6083"), "end_turn")
    use_fake_answers(
        monkeypatch, sent, [other, with_answer(RIGHT, "end_turn")]
    )
    main(["extract", str(maria), "--orders", "orders.json"])
    assert len(sent) == 2


# Maria writes again, as a reply to the shop's last email, which was about
# another of her orders. The order she means is in the P.S.; the quoted
# email below it names the vase's order, 7302, a real order of the shop.
REPLY = """Hi again,

My mug came today in pieces, and the box was torn as well. I'd like
a new one, or my money back if you have run out.

Thanks,
Maria
Tel. 0161 496 0731

P.S. The order number is 4417.

On 3 Nov 2026, at 10:14, the shop wrote:
> Your vase, order 7302, was sent today. It should reach you
> within 14 days.
"""
# The ticket the reply asks for: both things that went wrong, her first
# wish, and the order in the P.S.
REPLY_RIGHT = (
    '{"order":4417,"problem":"mug broken, box torn","wish":"replacement"}'
)
# Valid, and the vase's order from the quoted reply.
QUOTED = '{"order":7302,"problem":"mug broken","wish":"replacement"}'


@pytest.fixture
def reply(maria: Path) -> Path:
    maria.write_text(REPLY, encoding="utf-8")
    return maria


def test_the_quoted_reply_is_not_sent_to_the_model(
    reply: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []
    use_fake_answers(
        monkeypatch, sent, [with_answer(REPLY_RIGHT, "end_turn")]
    )
    main(["extract", str(reply), "--orders", "orders.json"])
    email = sent[0]["messages"][0]["content"]
    assert "7302" not in email
    assert "the shop wrote" not in email
    assert email.endswith("P.S. The order number is 4417.")


def test_an_order_only_in_the_quoted_reply_is_asked_for_again(
    reply: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # It used to be printed as Maria's ticket, with exit code 0: 7302 is a
    # number in the email file and an order the shop has, so both rules
    # passed it.
    sent: list[dict[str, Any]] = []
    use_fake_answers(
        monkeypatch,
        sent,
        [
            with_answer(QUOTED, "end_turn"),
            with_answer(REPLY_RIGHT, "end_turn"),
        ],
    )
    main(["extract", str(reply), "--orders", "orders.json"])
    out = capsys.readouterr().out.splitlines()
    assert out[1] == '  "order": 4417,'
    assert out[-2:] == [
        "Left out:      3 quoted lines",
        "Asked again:   order: 7302 is not a number in the email",
    ]
    assert len(sent) == 2


def test_an_email_with_nothing_quoted_says_nothing_was_left_out(
    maria: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sent: list[dict[str, Any]] = []
    use_fake_answers(monkeypatch, sent, [with_answer(RIGHT, "end_turn")])
    main(["extract", str(maria)])
    assert (
        capsys.readouterr().out.splitlines()[-2] == "Stop reason:   end_turn"
    )
    assert sent[0]["messages"][0]["content"] == EMAIL


def test_extract_takes_a_max_tokens_only_a_stream_may_send(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The SDK refuses an ordinary call that could take over ten minutes
    # (lesson 2.5), so with this much room the ticket must stream.
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    main(["extract", str(email_file), "--max-tokens", "64000"])
    assert sent[0]["max_tokens"] == 64_000
    assert sent[0]["stream"] is True
    assert '  "order": 4417,' in capsys.readouterr().out.splitlines()


def test_extract_previews_settled_fields_apart_from_the_ticket(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    use_fake_client(monkeypatch, [])
    main(["extract", str(email_file)])
    captured = capsys.readouterr()
    assert captured.err.splitlines() == [
        "Settled:       order 4417",
        'Settled:       problem "mug broken"',
        'Settled:       wish "replacement"',
    ]
    assert "Settled" not in captured.out


def test_extract_cancelled_with_ctrl_c_prints_no_ticket(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stream = Interrupted(WRITTEN, 1)

    def make(api_key: str) -> Anthropic:
        return cancelling_client(stream)

    monkeypatch.setattr("askkit.cli.Anthropic", make)
    with pytest.raises(SystemExit) as stopped:
        main(["extract", str(email_file)])
    assert stopped.value.code == f"askkit: {CANCELLED}"
    assert stream.closed
    assert capsys.readouterr().out.splitlines() == [
        "Time:          first token 0.42 s, cancelled at 0.84 s"
    ]


def test_a_place_inside_fields_is_joined_with_dots() -> None:
    assert fault_place(("lines", 1, "amount")) == "lines.1.amount"
    assert fault_place(("wish",)) == "wish"
    assert fault_place(()) == "answer"


def test_a_fault_inside_a_nested_model_names_its_whole_place() -> None:
    # Before Invoice had rows, only the first part of a place was shown,
    # so every row's fault read "lines" and no row could be told apart.
    class Row(BaseModel):
        amount: float

    class Rows(BaseModel):
        lines: list[Row]

    with pytest.raises(ValidationError) as raised:
        Rows.model_validate({"lines": [{"amount": 1}, {"amount": "x"}]})
    assert ticket_faults(raised.value) == (
        "lines.1.amount: Input should be a valid number, unable to parse"
        " string as a number, not 'x'"
    )


def test_a_validators_own_words_are_shown_as_written() -> None:
    # A ValueError from a validator: Pydantic's msg starts with "Value
    # error, " and its input is the whole object, which said nothing the
    # validator's words did not.
    class Sum(BaseModel):
        total: int

        @model_validator(mode="after")
        def positive(self) -> Self:
            raise ValueError(f"{self.total} is not above 0")

    with pytest.raises(ValidationError) as raised:
        Sum.model_validate({"total": -3})
    assert ticket_faults(raised.value) == "answer: -3 is not above 0"
