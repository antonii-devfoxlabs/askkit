import json
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic

from askkit.ask import answer_text, ask, stop_warning, thinking_tokens
from askkit.cli import main
from askkit.prompts import ASK_SYSTEM, render

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


def with_thinking(thinking: int, output: int) -> dict[str, Any]:
    body = dict(RECORDED)
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
