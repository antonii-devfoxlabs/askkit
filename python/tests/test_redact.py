import json
from pathlib import Path
from typing import Any

import pytest
from test_ask import WRITTEN
from test_calllog import answering, content_lines, log_folder, run_here
from test_classify import use_fake_client

from askkit.classify import EMAIL_SOURCE, sorting_messages
from askkit.cli import main
from askkit.extract import ticket_messages
from askkit.redact import redacted
from askkit.untrusted import marked

__all__ = ["run_here"]

# Written for these tests: 4242 4242 4242 4242 is the card number Stripe
# publishes for tests, 07700 900xxx a mobile range the UK regulator keeps
# for drama, and example.com a domain no one can own.
MARIA = (
    "Order 4417 came with a broken mug. Please refund it to my card"
    " 4242 4242 4242 4242, or call me on +44 7700 900123. My email is"
    " maria.k@example.com.\nMaria"
)
MARIA_REDACTED = (
    "Order 4417 came with a broken mug. Please refund it to my card"
    " [CARD], or call me on [PHONE]. My email is [EMAIL].\nMaria"
)


def test_card_email_and_phone_become_placeholders() -> None:
    assert redacted(MARIA) == MARIA_REDACTED


@pytest.mark.parametrize(
    ("text", "shown"),
    [
        ("card 4242424242424242", "card [CARD]"),
        ("card 4242-4242-4242-4242", "card [CARD]"),
        ("write to tom.b+shop@mail.example.co.uk", "write to [EMAIL]"),
        ("call 07700 900123 today", "call [PHONE] today"),
        ("call +447700900123", "call [PHONE]"),
    ],
)
def test_each_kind_in_the_forms_people_write_it(
    text: str, shown: str
) -> None:
    assert redacted(text) == shown


def test_a_card_is_named_a_card_not_a_phone() -> None:
    # Its 16 digits match the phone pattern too: the card goes first.
    assert redacted("4242 4242 4242 4242") == "[CARD]"


def test_order_numbers_and_the_full_stop_after_an_address_stay() -> None:
    assert redacted("Order 4417, from ana@example.com.") == (
        "Order 4417, from [EMAIL]."
    )


def test_redacting_twice_changes_nothing_more() -> None:
    assert redacted(MARIA_REDACTED) == MARIA_REDACTED


def test_names_and_street_addresses_pass() -> None:
    # No pattern finds them: the limit the README and the lesson name.
    text = "Maria Keller, 12 Elm Road"
    assert redacted(text) == text


def test_the_content_file_keeps_placeholders_not_the_numbers(
    run_here: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []
    answering(monkeypatch, sent, WRITTEN)
    main(["--log-content", "ask", MARIA])
    # The question goes to the model as it was asked: an answer about
    # it may need every part of it.
    assert sent[0]["messages"][0]["content"][0]["text"] == MARIA
    [kept] = content_lines()
    assert kept["prompt"].endswith("user: " + MARIA_REDACTED)
    for path in log_folder().iterdir():
        text = path.read_text(encoding="utf-8")
        assert "4242" not in text
        assert "900123" not in text
        assert "maria.k@example.com" not in text


def test_an_answer_that_repeats_a_number_is_redacted_in_the_log(
    run_here: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    echoed = dict(WRITTEN)
    echoed["content"] = [
        {"type": "text", "text": "We will call +44 7700 900123."}
    ]
    answering(monkeypatch, [], echoed)
    main(["--log-content", "ask", "When will you call me?"])
    [kept] = content_lines()
    assert kept["answer"] == "We will call [PHONE]."


def test_a_sorting_call_sends_the_email_without_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    path = tmp_path / "maria.txt"
    path.write_text(MARIA, encoding="utf-8")
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    main(["classify", str(path)])
    content = sent[0]["messages"][0]["content"]
    assert content == marked(EMAIL_SOURCE, MARIA_REDACTED)
    assert json.loads(content)["content"] == MARIA_REDACTED
    assert sorting_messages(MARIA) == [{"role": "user", "content": content}]


def test_an_extract_call_sends_the_email_without_them() -> None:
    [message] = ticket_messages(MARIA)
    assert message == {"role": "user", "content": MARIA_REDACTED}
