import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic
from test_ask import replied

from askkit.cli import main
from askkit.documents import (
    documents_block,
    missing_quotes,
    quotes_in,
    unchecked,
    with_documents,
)
from askkit.prompts import template_path

REQUEST = template_path("quote_request", 2).read_text(encoding="utf-8")

# Lesson 3.4's returns policy, as the file breaks its lines.
POLICY = """Returns policy

1. Changing your mind
You can return an unused item in its original box within 30 days of
delivery. You pay the return postage. We refund the price of the item
within 5 working days of receiving it.

3. Wrong items
If we sent the wrong item or colour, we pay the return postage. Tell us
within 30 days of delivery and we email you a prepaid label.
"""

ANSWER = (
    "<quote>If we sent the wrong item or colour, we pay the return"
    " postage.</quote>\n"
    "<quote>Tell us within 30 days of delivery and we email you a prepaid"
    " label.</quote>\n"
    "<answer>We pay: we email the customer a prepaid label.</answer>"
)


def test_documents_block_numbers_each_document_with_its_source() -> None:
    block = documents_block(
        {"returns-policy.txt": "  Returns policy\n", "faq.txt": "FAQ"}
    )
    assert block == (
        "<documents>\n"
        '<document index="1">\n'
        "<source>returns-policy.txt</source>\n"
        "<document_content>\n"
        "Returns policy\n"
        "</document_content>\n"
        "</document>\n"
        '<document index="2">\n'
        "<source>faq.txt</source>\n"
        "<document_content>\n"
        "FAQ\n"
        "</document_content>\n"
        "</document>\n"
        "</documents>"
    )


def test_with_documents_puts_the_documents_first_and_the_question_last() -> (
    None
):
    message = with_documents("Who pays?", {"returns-policy.txt": POLICY})
    request = message.index("Answer the question below")
    assert message.startswith("<documents>\n")
    assert message.index("</documents>") < request
    assert request < message.index("<question>")
    assert message.endswith("<question>\nWho pays?\n</question>")


def test_with_documents_sends_what_it_sent_before_the_templates() -> None:
    # The message as an f-string built it before quote_request moved into
    # a template file: the move must not change one character.
    documents = {"returns-policy.txt": POLICY}
    request = REQUEST.split("\n\n")[1]
    assert with_documents("Who pays?", documents) == (
        f"{documents_block(documents)}\n\n{request}\n\n"
        "<question>\nWho pays?\n</question>"
    )


def test_the_request_asks_for_quotes_first_and_a_way_out() -> None:
    assert "<quote></quote>" in REQUEST
    assert "<answer></answer>" in REQUEST
    assert "write exactly: I don't know.\n\n<question>" in REQUEST


def test_the_request_asks_for_a_check_before_the_answer() -> None:
    check = REQUEST.index("Before you write the")
    assert REQUEST.index("<quote></quote>") < check
    assert check < REQUEST.index("<answer></answer>")
    assert "exception" in REQUEST


def test_quotes_in_finds_every_quote() -> None:
    assert quotes_in(ANSWER) == [
        "If we sent the wrong item or colour, we pay the return postage.",
        "Tell us within 30 days of delivery and we email you a prepaid"
        " label.",
    ]


def test_quotes_in_finds_none_in_an_answer_without_quotes() -> None:
    assert quotes_in("<answer>I don't know.</answer>") == []


def test_quotes_in_finds_none_in_a_tag_with_an_attribute() -> None:
    # The model may write <quote index="1">: not the tag the request
    # asked for, so quotes_in reads no quote in it.
    assert quotes_in('<quote index="1">You pay.</quote>') == []


def test_an_answer_that_quotes_nothing_is_unchecked() -> None:
    answer = (
        '<quote index="1">You pay the return postage.</quote>'
        "<answer>The customer pays.</answer>"
    )
    assert unchecked(answer)


def test_the_way_out_is_not_unchecked() -> None:
    assert not unchecked("<answer>I don't know.</answer>")


def test_an_answer_with_quotes_is_not_unchecked() -> None:
    assert not unchecked(ANSWER)


def test_a_quote_across_a_line_break_is_found() -> None:
    assert missing_quotes(ANSWER, {"returns-policy.txt": POLICY}) == []


def test_a_quote_in_any_document_is_found() -> None:
    documents = {"faq.txt": "Opening hours.", "returns-policy.txt": POLICY}
    assert missing_quotes(ANSWER, documents) == []


def test_a_second_quote_not_in_the_documents_is_missing() -> None:
    answer = (
        "<quote>You pay the return postage.</quote>"
        "<quote>Sale items are returned free of charge.</quote>"
        "<answer>The customer pays.</answer>"
    )
    assert missing_quotes(answer, {"returns-policy.txt": POLICY}) == [
        "Sale items are returned free of charge."
    ]


def test_a_reworded_quote_is_missing() -> None:
    answer = "<quote>We pay the postage for a wrong colour.</quote>"
    assert missing_quotes(answer, {"returns-policy.txt": POLICY}) == [
        "We pay the postage for a wrong colour."
    ]


def recorded(text: str) -> dict[str, Any]:
    return {
        "id": "msg_01AskKitDocuments",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 412, "output_tokens": 61},
    }


def use_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    sent: list[dict[str, Any]],
    text: str,
) -> None:
    def answer(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return replied(request, recorded(text))

    def make(**options: Any) -> Anthropic:
        return Anthropic(
            api_key="sk-ant-api03-a-test-key-Ab12",
            http_client=httpx2.Client(transport=httpx2.MockTransport(answer)),
        )

    monkeypatch.setattr("askkit.cli.Anthropic", make)


@pytest.fixture
def in_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    (tmp_path / "returns-policy.txt").write_text(POLICY, encoding="utf-8")
    return tmp_path


def test_ask_sends_the_document_first_and_the_question_last(
    in_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent, ANSWER)
    main(
        [
            "ask",
            "Who pays the postage for order 6083?",
            "--document",
            "returns-policy.txt",
        ]
    )
    documents, question = sent[0]["messages"][0]["content"]
    # Two blocks, each with a cache breakpoint: the documents with the
    # request, which stay the same from question to question, then the
    # question. Together they are the words with_documents writes.
    assert documents["cache_control"] == {"type": "ephemeral"}
    asked = "Who pays the postage for order 6083?"
    assert question == {
        "type": "text",
        "text": f"<question>\n{asked}\n</question>",
        "cache_control": {"type": "ephemeral"},
    }
    assert documents["text"] + "\n\n" + question["text"] == with_documents(
        "Who pays the postage for order 6083?",
        {"returns-policy.txt": POLICY},
    )
    lines = capsys.readouterr().out.splitlines()
    assert lines[-3:] == [
        "Stop reason:   end_turn",
        "Time:          first token 0.42 s, total 0.84 s",
        "Quotes:        2 of 2 found in the documents",
    ]


def test_ask_takes_several_documents_in_order(
    in_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (in_project / "faq.txt").write_text("Opening hours.\n", encoding="utf-8")
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent, ANSWER)
    main(
        [
            "ask",
            "Who pays?",
            "--document",
            "returns-policy.txt",
            "--document",
            "faq.txt",
        ]
    )
    content = sent[0]["messages"][0]["content"][0]["text"]
    assert content.index("<source>returns-policy.txt</source>") < (
        content.index("<source>faq.txt</source>")
    )


def test_ask_stops_on_a_quote_that_is_not_in_the_documents(
    in_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    answer = (
        "<quote>You pay the return postage.</quote>"
        "<quote>Sale items are returned free of charge.</quote>"
        "<answer>The customer pays.</answer>"
    )
    use_fake_client(monkeypatch, [], answer)
    with pytest.raises(SystemExit) as stopped:
        main(["ask", "Who pays?", "--document", "returns-policy.txt"])
    assert stopped.value.code == (
        'askkit: not in the documents: "Sale items are returned free of'
        ' charge."'
    )
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "Quotes:        1 of 2 found in the documents"


def test_ask_says_when_the_answer_quotes_nothing(
    in_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    use_fake_client(monkeypatch, [], "<answer>I don't know.</answer>")
    main(["ask", "Who pays?", "--document", "returns-policy.txt"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "Quotes:        none in the answer"


def test_ask_stops_on_an_answer_it_cannot_check(
    in_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Before lesson 4.1 this answer passed: quotes_in found no quote, so
    # missing_quotes had nothing to report, and AskKit ended without an
    # error on an answer nothing had checked.
    answer = (
        '<quote index="1">Sale items are returned free of charge.</quote>'
        "<answer>The customer pays nothing.</answer>"
    )
    use_fake_client(monkeypatch, [], answer)
    with pytest.raises(SystemExit) as stopped:
        main(["ask", "Who pays?", "--document", "returns-policy.txt"])
    assert stopped.value.code == (
        "askkit: the answer quotes nothing from the documents, so it"
        " cannot be checked"
    )
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "Quotes:        none in the answer"


def test_ask_without_documents_sends_the_question_alone(
    in_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent, "Plain answer.")
    main(["ask", "Who pays?"])
    assert sent[0]["messages"][0]["content"] == [
        {
            "type": "text",
            "text": "Who pays?",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert "Quotes:" not in capsys.readouterr().out
