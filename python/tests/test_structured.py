import json
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic
from test_ask import Interrupted, cancelling_client, replied

from askkit.ask import Timing
from askkit.extract import TICKET_SCHEMA
from askkit.structured import settled_fields, settled_value, stream_json

# A ticket written with spaces, so the fake API sends it in several
# pieces, one after each space, as a stream would.
TICKET = (
    '{"order": 4417, "problem": "mug broken, box torn",'
    ' "wish": "replacement"}'
)

WRITTEN = {
    "id": "msg_01AskKitStructured",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [
        {"type": "thinking", "thinking": "", "signature": "test"},
        {"type": "text", "text": TICKET},
    ],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 236, "output_tokens": 31},
}


def fake_client(sent: list[dict[str, Any]]) -> Anthropic:
    def answer(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return replied(request, WRITTEN)

    transport = httpx2.MockTransport(answer)
    return Anthropic(
        api_key="sk-ant-api03-a-test-key-Ab12",
        http_client=httpx2.Client(transport=transport),
    )


def stream_ticket(client: Anthropic) -> tuple[Any, Timing]:
    return stream_json(
        client,
        "claude-haiku-5-5",
        1000,
        "Turn the email into a ticket.",
        [{"role": "user", "content": "Order 4417: the mug came broken."}],
        TICKET_SCHEMA,
    )


def test_nothing_is_settled_before_the_first_comma() -> None:
    assert settled_fields("") == {}
    assert settled_fields('{"order": 4417') == {}


def test_a_number_still_being_written_is_not_settled() -> None:
    # A partial parser reads "44" here; the stream may still add "17".
    assert settled_fields('{"order": 44') == {}


def test_a_field_settles_at_the_comma_after_it() -> None:
    assert settled_fields('{"order": 4417,') == {"order": 4417}
    assert settled_fields('{"order": null, "pro') == {"order": None}


def test_a_comma_inside_a_text_value_settles_nothing() -> None:
    text = '{"order": 4417, "problem": "mug broken,'
    assert settled_fields(text) == {"order": 4417}


def test_a_brace_inside_a_text_value_settles_nothing() -> None:
    text = '{"order": 4417, "problem": "mug } broken'
    assert settled_fields(text) == {"order": 4417}


def test_the_last_field_settles_at_the_closing_brace() -> None:
    assert settled_fields(TICKET[:-1]) == {
        "order": 4417,
        "problem": "mug broken, box torn",
    }
    assert settled_fields(TICKET) == json.loads(TICKET)


def test_stream_json_streams_the_schema_beside_the_messages() -> None:
    sent: list[dict[str, Any]] = []
    stream_ticket(fake_client(sent))
    assert sent == [
        {
            "model": "claude-haiku-5-5",
            "max_tokens": 1000,
            "system": "Turn the email into a ticket.",
            "messages": [
                {
                    "role": "user",
                    "content": "Order 4417: the mug came broken.",
                }
            ],
            "output_config": {
                "format": {"type": "json_schema", "schema": TICKET_SCHEMA}
            },
            "stream": True,
        }
    ]


def test_each_field_is_shown_once_when_it_settles(
    capsys: pytest.CaptureFixture[str],
) -> None:
    response, timing = stream_ticket(fake_client([]))
    captured = capsys.readouterr()
    # The preview goes to the error output, so a program that reads the
    # ticket from the output never takes a field from it.
    assert captured.out == ""
    assert captured.err.splitlines() == [
        "Settled:       order 4417",
        'Settled:       problem "mug broken, box torn"',
        'Settled:       wish "replacement"',
    ]
    assert response is not None
    assert response.stop_reason == "end_turn"
    assert timing == Timing(first_token=0.42, total=0.84)


def test_a_large_max_tokens_goes_as_a_stream() -> None:
    # Lesson 2.5: the SDK refuses an ordinary call this large.
    sent: list[dict[str, Any]] = []
    stream_json(
        fake_client(sent),
        "claude-haiku-5-5",
        64_000,
        "Turn the email into a ticket.",
        [{"role": "user", "content": "Order 4417"}],
        TICKET_SCHEMA,
    )
    assert sent[0]["max_tokens"] == 64_000


def test_ctrl_c_returns_no_message_and_shows_only_settled_fields(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Two pieces arrive, '{"order":' and ' 4417,': the order has settled,
    # the problem has not begun.
    stream = Interrupted(WRITTEN, 2)
    response, timing = stream_ticket(cancelling_client(stream))
    assert response is None
    assert timing == Timing(first_token=0.42, total=0.84)
    assert stream.closed
    assert capsys.readouterr().err.splitlines() == [
        "Settled:       order 4417"
    ]


def test_a_list_is_previewed_by_its_length() -> None:
    # An invoice's rows were printed as one line of JSON holding every
    # row, longer than the terminal, before the invoice was checked.
    rows = [{"item": "Teapot", "amount": 307.2}, {"item": "Mug"}]
    assert settled_value(rows) == "(a list of 2)"
    assert settled_value("mug broken") == '"mug broken"'
    assert settled_value(4417) == "4417"
