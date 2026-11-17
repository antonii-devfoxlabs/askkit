import json
import re
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic, APIError

from askkit.ask import (
    answer_text,
    ask,
    stop_warning,
    stream_answer,
    thinking_tokens,
)
from askkit.cli import main
from askkit.prompts import ASK_SYSTEM, render

# A response shaped like the Messages API's, written for the tests:
# Haiku 5.5 may think first, so the text is not always the first block.
WRITTEN = {
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


def event(name: str, data: dict[str, Any]) -> str:
    # One server-sent event: its name, then its data as JSON.
    return f"event: {name}\ndata: {json.dumps(data)}\n\n"


def pieces(text: str) -> list[str]:
    # The text in word-sized pieces, each with the spaces before it, as
    # the API sends an answer in a stream.
    return re.findall(r"\s*\S+|\s+$", text)


def opening(body: dict[str, Any]) -> list[str]:
    # A stream's first events: the message with no content yet, then each
    # block opened, filled piece by piece, and closed.
    start = dict(body)
    start["content"] = []
    start["stop_reason"] = None
    start["usage"] = {
        "input_tokens": body["usage"]["input_tokens"],
        "output_tokens": 1,
    }
    events = [
        event("message_start", {"type": "message_start", "message": start})
    ]
    for index, block in enumerate(body["content"]):
        if block["type"] == "thinking":
            empty = {"type": "thinking", "thinking": "", "signature": ""}
            deltas = [
                {"type": "thinking_delta", "thinking": block["thinking"]},
                {"type": "signature_delta", "signature": block["signature"]},
            ]
        else:
            empty = {"type": "text", "text": ""}
            deltas = []
            for piece in pieces(block["text"]):
                deltas.append({"type": "text_delta", "text": piece})
        events.append(
            event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": empty,
                },
            )
        )
        for delta in deltas:
            events.append(
                event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": delta,
                    },
                )
            )
        events.append(
            event(
                "content_block_stop",
                {"type": "content_block_stop", "index": index},
            )
        )
    return events


def closing(body: dict[str, Any]) -> list[str]:
    # A stream's last events: the stop reason and the usage, then the end.
    return [
        event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": body["stop_reason"],
                    "stop_sequence": None,
                    "stop_details": body.get("stop_details"),
                },
                "usage": body["usage"],
            },
        ),
        event("message_stop", {"type": "message_stop"}),
    ]


def streamed(body: dict[str, Any]) -> httpx2.Response:
    # The written response, sent the way the API sends a stream.
    return httpx2.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text="".join(opening(body) + closing(body)),
    )


def broken_off(body: dict[str, Any], shown: int) -> httpx2.Response:
    # The stream ends with an error event after the first pieces of text:
    # the overload a busy API sends in place of a 529 once it has started.
    events = opening(body)
    cut = events.index(
        next(text for text in events if '"text_delta"' in text)
    )
    error = event(
        "error",
        {
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Overloaded"},
        },
    )
    return httpx2.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text="".join(events[: cut + shown] + [error]),
    )


def replied(request: httpx2.Request, body: dict[str, Any]) -> httpx2.Response:
    # A request that asks for a stream gets one; any other gets the JSON.
    if json.loads(request.content).get("stream"):
        return streamed(body)
    return httpx2.Response(200, json=body)


def fake_client(
    sent: list[dict[str, Any]], body: dict[str, Any] = WRITTEN
) -> Anthropic:
    def answer(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return replied(request, body)

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
    body = dict(WRITTEN)
    body["content"] = [
        {"type": "text", "text": "Order 4417: "},
        {"type": "text", "text": "a replacement mug."},
    ]
    response = ask(
        fake_client([], body), "Hi", "claude-haiku-5-5", 1000, "Be brief."
    )
    assert answer_text(response) == "Order 4417: a replacement mug."


def test_ask_sends_the_effort_inside_output_config() -> None:
    sent: list[dict[str, Any]] = []
    ask(fake_client(sent), "Hi", "claude-haiku-5-5", 1000, "Be brief.", "low")
    assert sent[0]["output_config"] == {"effort": "low"}
    assert "effort" not in sent[0]


def test_ask_without_an_effort_leaves_the_model_default() -> None:
    sent: list[dict[str, Any]] = []
    ask(fake_client(sent), "Hi", "claude-haiku-5-5", 1000, "Be brief.", None)
    assert "output_config" not in sent[0]


def use_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    sent: list[dict[str, Any]],
    body: dict[str, Any] = WRITTEN,
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


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_ask_takes_every_effort_level(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch, effort: str
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    main(["ask", "Hi", "--effort", effort])
    assert sent[0]["output_config"] == {"effort": effort}


def test_ask_refuses_an_effort_the_api_does_not_take(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    with pytest.raises(SystemExit):
        main(["ask", "Hi", "--effort", "Low"])
    assert "invalid choice: 'Low'" in capsys.readouterr().err
    assert sent == []


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


def test_the_word_limit_leaves_copied_sentences_out() -> None:
    # The quote request asks for every sentence that applies; a limit that
    # counted them would leave the model a choice between the two rules.
    request = render("quote_request", {"documents": "", "question": ""})
    assert "Sentences you copy from documents do not count" in ASK_SYSTEM
    assert "word\nfor word" in request


def test_both_prompts_end_on_the_same_way_out() -> None:
    way_out = "write exactly: I don't know."
    request = render("quote_request", {"documents": "", "question": ""})
    assert ASK_SYSTEM.endswith(way_out)
    assert f"{way_out}\n\n<question>" in request


def stopped_with(reason: str) -> dict[str, Any]:
    body = dict(WRITTEN)
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


def test_a_refusal_shows_the_apis_explanation() -> None:
    # The explanation is the API's own words for the person who picks the
    # question up; it is shown as it came, never parsed.
    body = stopped_with("refusal")
    body["stop_details"] = {
        "type": "refusal",
        "category": "cyber",
        "explanation": (
            "This request was declined because it could enable cyber harm."
        ),
    }
    response = ask(
        fake_client([], body), "Hi", "claude-haiku-5-5", 60, "Be brief."
    )
    assert stop_warning(response) == (
        "the model declined to answer this question (category: cyber)."
        " This request was declined because it could enable cyber harm."
    )


def test_a_refusal_names_what_was_declined() -> None:
    body = stopped_with("refusal")
    response = ask(
        fake_client([], body), "Hi", "claude-haiku-5-5", 60, "Be brief."
    )
    assert stop_warning(response, "sort this email into a queue") == (
        "the model declined to sort this email into a queue"
    )


def test_ask_marks_the_shown_part_of_a_declined_answer(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Streamed, the text before the refusal is on screen already, so AskKit
    # says not to use it.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    use_fake_client(monkeypatch, [], stopped_with("refusal"))
    with pytest.raises(SystemExit) as stopped:
        main(["ask", "Hi"])
    assert stopped.value.code == (
        "askkit: the model declined to answer this question"
    )
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        "Order 4417: a replacement mug.",
        "",
        "Model:         claude-haiku-5-5",
        "Tokens:        118 in, 60 out",
        "Cost:          $0.000042",
        "Stop reason:   refusal",
    ]
    assert captured.err == (
        "askkit: the text above stopped where the model declined;"
        " do not use it\n"
    )


def test_ask_prints_nothing_for_a_refusal_before_any_text(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    body = stopped_with("refusal")
    body["content"] = []
    use_fake_client(monkeypatch, [], body)
    with pytest.raises(SystemExit):
        main(["ask", "Hi"])
    captured = capsys.readouterr()
    assert captured.out.splitlines()[0] == "Model:         claude-haiku-5-5"
    assert captured.err == ""


def test_stream_answer_sends_the_request_of_ask_as_a_stream() -> None:
    asked: list[dict[str, Any]] = []
    streamed_requests: list[dict[str, Any]] = []
    ask(
        fake_client(asked), "Hi", "claude-haiku-5-5", 1000, "Be brief.", "low"
    )
    stream_answer(
        fake_client(streamed_requests),
        "Hi",
        "claude-haiku-5-5",
        1000,
        "Be brief.",
        "low",
    )
    assert streamed_requests == [{**asked[0], "stream": True}]


def test_stream_answer_prints_the_pieces_and_returns_the_message(
    capsys: pytest.CaptureFixture[str],
) -> None:
    response = stream_answer(
        fake_client([]), "Hi", "claude-haiku-5-5", 1000, "Be brief."
    )
    assert capsys.readouterr().out == "Order 4417: a replacement mug.\n"
    assert answer_text(response) == "Order 4417: a replacement mug."
    assert response.stop_reason == "end_turn"
    assert response.usage.input_tokens == 118
    assert response.usage.output_tokens == 27


def test_the_stream_arrives_in_pieces() -> None:
    # The fake sends the answer word by word, as the API does: the test
    # above would pass on one piece too, so this one counts them.
    shown: list[str] = []
    client = fake_client([])
    with client.messages.stream(
        model="claude-haiku-5-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": "Hi"}],
    ) as stream:
        for text in stream.text_stream:
            shown.append(text)
    assert shown == ["Order", " 4417:", " a", " replacement", " mug."]


def test_a_large_max_tokens_needs_the_stream() -> None:
    # Lesson 2.5: the SDK refuses an ordinary call that could take over ten
    # minutes. Streamed, the same request goes.
    with pytest.raises(ValueError, match="Streaming is required"):
        ask(fake_client([]), "Hi", "claude-haiku-5-5", 64_000, "Be brief.")
    sent: list[dict[str, Any]] = []
    stream_answer(fake_client(sent), "Hi", "claude-haiku-5-5", 64_000, "Hi")
    assert sent[0]["max_tokens"] == 64_000


def test_a_stream_that_breaks_off_ends_its_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def answer(request: httpx2.Request) -> httpx2.Response:
        return broken_off(WRITTEN, 2)

    client = Anthropic(
        api_key="sk-ant-api03-a-test-key-Ab12",
        http_client=httpx2.Client(transport=httpx2.MockTransport(answer)),
        max_retries=0,
    )
    with pytest.raises(APIError):
        stream_answer(client, "Hi", "claude-haiku-5-5", 1000, "Be brief.")
    assert capsys.readouterr().out == "Order 4417:\n"


def test_askkit_ask_says_when_the_stream_breaks_off(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")

    def answer(request: httpx2.Request) -> httpx2.Response:
        return broken_off(WRITTEN, 2)

    def make(**options: Any) -> Anthropic:
        return Anthropic(
            api_key="sk-ant-api03-a-test-key-Ab12",
            http_client=httpx2.Client(transport=httpx2.MockTransport(answer)),
            max_retries=0,
        )

    monkeypatch.setattr("askkit.cli.Anthropic", make)
    with pytest.raises(SystemExit) as stopped:
        main(["ask", "Hi"])
    assert stopped.value.code == (
        "askkit: the answer broke off midway (overloaded_error): Overloaded;"
        "\nask again"
    )
    assert capsys.readouterr().out == "Order 4417:\n"


def with_thinking(thinking: int, output: int) -> dict[str, Any]:
    body = dict(WRITTEN)
    usage: dict[str, Any] = {
        "input_tokens": 118,
        "output_tokens": output,
        "output_tokens_details": {"thinking_tokens": thinking},
    }
    body["usage"] = usage
    return body


def test_thinking_tokens_reads_the_split_of_the_output() -> None:
    response = ask(
        fake_client([], with_thinking(180, 230)),
        "Hi",
        "claude-haiku-5-5",
        1000,
        "Be brief.",
    )
    assert thinking_tokens(response) == 180


def test_thinking_tokens_is_none_when_the_api_sends_no_split() -> None:
    response = ask(
        fake_client([]), "Hi", "claude-haiku-5-5", 1000, "Be brief."
    )
    assert thinking_tokens(response) is None


def test_ask_prints_how_many_output_tokens_were_thinking(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    use_fake_client(monkeypatch, [], with_thinking(180, 230))
    main(["ask", "Where is my order?"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[3] == "Tokens:        118 in, 230 out (180 thinking)"
