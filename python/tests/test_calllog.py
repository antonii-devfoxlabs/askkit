import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic
from anthropic.types import DocumentBlockParam, Message
from openai import OpenAI
from test_ask import WRITTEN, replied
from test_audio import VOICEMAIL, keyed, use_fake_openai
from test_classify import with_answer
from test_fallback import luna_body
from test_invoices import FIELDS, INVOICE
from test_retry import OVERLOADED, Step, answer_body, answer_with, ask_set

import askkit.cli
from askkit import calllog
from askkit.calllog import (
    CONTENT_MAX_AGE,
    LoggedCall,
    day_calls,
    day_lines,
    log_call,
    logged_call,
    prompt_text,
    start_log,
)
from askkit.cli import main
from askkit.prices import PRICES
from askkit.prompts import ASK_SYSTEM
from askkit.trim import summarise

__all__ = ["ask_set", "keyed"]

MORNING = datetime(2026, 11, 20, 9, 14, tzinfo=UTC)
REQUEST_ID = "req_011CAskKitLogged"
OPENAI_ID = "req_4f1c0e8a2b7d49c6a3e5f0b1d2c3e4f5"
QUESTION = "Order 4417 came with a broken mug, from maria@example.com."


def log_folder() -> Path:
    # The folder the conftest gave this test's CLI runs, read from the
    # module as the CLI reads it.
    folder: Path = vars(askkit.cli)["LOG_FOLDER"]
    return folder


def logged() -> list[LoggedCall]:
    return day_calls(log_folder(), MORNING.date())


def content_lines() -> list[dict[str, Any]]:
    path = log_folder() / "content-2026-11-20.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


@pytest.fixture
def run_here(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    monkeypatch.setattr("askkit.calllog.clock", lambda: MORNING)
    monkeypatch.setattr("askkit.cli.clock", lambda: MORNING)


def answering(
    monkeypatch: pytest.MonkeyPatch,
    sent: list[dict[str, Any]],
    body: dict[str, Any],
) -> None:
    # Every client the CLI makes answers with body and the request ID
    # header the API sends with each response.
    def answer(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        response = replied(request, body)
        response.headers["request-id"] = REQUEST_ID
        return response

    def make(**options: Any) -> Anthropic:
        return Anthropic(
            api_key="sk-ant-api03-a-test-key-Ab12",
            http_client=httpx2.Client(transport=httpx2.MockTransport(answer)),
        )

    monkeypatch.setattr("askkit.cli.Anthropic", make)


@pytest.mark.parametrize(
    "arguments, body",
    [
        (["ask", QUESTION], WRITTEN),
        (["read", str(INVOICE)], with_answer(json.dumps(FIELDS))),
        (["classify", "gift.txt"], with_answer('{"queue":"damaged"}')),
    ],
)
def test_every_request_a_command_sends_is_one_line_of_the_log(
    run_here: None,
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    body: dict[str, Any],
) -> None:
    # The mechanism, not one command: whatever path a command's calls
    # take, each request that reached the API is one line.
    Path("gift.txt").write_text(QUESTION, encoding="utf-8")
    sent: list[dict[str, Any]] = []
    answering(monkeypatch, sent, body)
    main(arguments)
    calls = logged()
    assert len(calls) == len(sent) == 1
    assert calls[0].command == arguments[0]
    assert calls[0].request_id == REQUEST_ID


def test_the_log_holds_numbers_and_names_but_no_words(
    run_here: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    answering(monkeypatch, [], WRITTEN)
    main(["ask", QUESTION])
    [call] = logged()
    # $0.10 and $0.50 a million tokens, worked out at the day's prices.
    assert call.cost == pytest.approx(0.0000253)
    assert call.model_copy(update={"cost": 0.0000253}) == LoggedCall(
        at=MORNING,
        command="ask",
        model="claude-haiku-5-5",
        input_tokens=118,
        cache_read=0,
        cache_written=0,
        output_tokens=27,
        seconds=0.84,
        cost=0.0000253,
        stop_reason="end_turn",
        retries=0,
        request_id=REQUEST_ID,
    )
    # Neither the question nor the answer is anywhere in the folder.
    for path in log_folder().iterdir():
        text = path.read_text(encoding="utf-8")
        assert "4417" not in text
        assert "maria@example.com" not in text
    assert [path.name for path in log_folder().iterdir()] == [
        "calls-2026-11-20.jsonl"
    ]


def test_log_content_keeps_what_was_sent_and_what_came_back(
    run_here: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    answering(monkeypatch, [], WRITTEN)
    main(["--log-content", "ask", QUESTION])
    [kept] = content_lines()
    assert kept["request_id"] == REQUEST_ID
    assert kept["prompt"] == f"system: {ASK_SYSTEM}\nuser: {QUESTION}"
    # The thinking block is named; its text is never the log's.
    assert kept["answer"] == "[thinking]\nOrder 4417: a replacement mug."


def test_a_pdf_or_an_image_is_named_in_the_content_never_copied() -> None:
    document: DocumentBlockParam = {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": "JVBERi0xLjcK",
        },
    }
    text = prompt_text(
        "Read the invoice.", [{"role": "user", "content": [document]}]
    )
    assert text == "system: Read the invoice.\nuser: [document]"


def test_content_older_than_a_week_is_removed(
    run_here: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = log_folder()
    folder.mkdir()
    old = folder / "content-2026-11-12.jsonl"
    week = folder / "content-2026-11-13.jsonl"
    numbers = folder / "calls-2026-11-01.jsonl"
    for path in (old, week, numbers):
        path.write_text("{}\n", encoding="utf-8")
    answering(monkeypatch, [], WRITTEN)
    main(["--log-content", "ask", QUESTION])
    # 8 days old: removed; 7 days: kept; the numbers are never removed.
    assert CONTENT_MAX_AGE.days == 7
    assert not old.exists()
    assert week.exists()
    assert numbers.exists()


def test_a_call_that_fails_its_checks_is_still_logged(
    run_here: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    wrong = dict(FIELDS, total=819.96)
    answering(monkeypatch, [], with_answer(json.dumps(wrong)))
    with pytest.raises(SystemExit):
        main(["read", str(INVOICE)])
    # Paid for, so logged, although no invoice was printed.
    assert [call.command for call in logged()] == ["read"]


def test_a_kept_answer_makes_no_call_and_no_line(
    run_here: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []
    answering(monkeypatch, sent, with_answer(json.dumps(FIELDS)))
    main(["read", str(INVOICE)])
    main(["read", str(INVOICE)])
    assert len(sent) == 1
    assert len(logged()) == 1


def test_answer_logs_each_question_with_its_retries(
    ask_set: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("askkit.calllog.clock", lambda: MORNING)
    sent: list[int] = []
    steps: list[Step] = [
        (200, answer_body(), {}),
        (529, OVERLOADED, {}),
        (529, OVERLOADED, {}),
        (200, answer_body(), {}),
    ]
    answer_with(steps, monkeypatch, [], sent)
    main(["answer", str(ask_set), "--out", "answers.json"])
    # Four requests, two answers: a failed try has no tokens to log, and
    # its answer's line says how many tries came before it.
    assert len(sent) == 4
    assert [call.retries for call in logged()] == [0, 2]
    assert [call.request_id for call in logged()] == [
        "req_011CAskKitRetry",
        "req_011CAskKitRetry",
    ]


def test_a_chat_summary_is_a_call_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("askkit.calllog.clock", lambda: MORNING)
    start_log(tmp_path, "chat", content=False)
    sent: list[dict[str, Any]] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(200, json=WRITTEN)

    client = Anthropic(
        api_key="sk-ant-api03-a-test-key-Ab12",
        http_client=httpx2.Client(transport=httpx2.MockTransport(answer)),
    )
    summarise(client, "claude-haiku-5-5", "", [])
    [call] = day_calls(tmp_path, MORNING.date())
    assert call.command == "chat"
    assert len(sent) == 1


def test_a_transcription_is_logged_with_its_price_by_the_minute(
    keyed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # cli.py stamps a transcription with its own imported clock, which
    # names the day's file: both clocks are held, or the test reads the
    # file of the day it was written on and fails on every other day.
    monkeypatch.setattr("askkit.calllog.clock", lambda: MORNING)
    monkeypatch.setattr("askkit.cli.clock", lambda: MORNING)
    use_fake_openai(monkeypatch, [])
    main(["transcribe", str(VOICEMAIL)])
    [call] = logged()
    assert (call.command, call.model) == ("transcribe", "gpt-transcribe")
    assert (call.input_tokens, call.output_tokens) == (154, 41)
    assert call.cost == pytest.approx(0.000831, abs=0.0000005)


def test_nothing_is_logged_outside_the_cli(tmp_path: Path) -> None:
    # The eval and tests call AskKit's functions directly: no run started.
    assert calllog.current is None
    response = Message.model_validate(WRITTEN)
    log_call(response, "claude-haiku-5-5", 0.5, "Be brief.", [], None)
    assert list(tmp_path.iterdir()) == []


def call_at(
    command: str, model: str, cost: float | None, **fields: Any
) -> LoggedCall:
    values: dict[str, Any] = {
        "at": MORNING,
        "command": command,
        "model": model,
        "input_tokens": 100,
        "cache_read": 0,
        "cache_written": 0,
        "output_tokens": 10,
        "seconds": 1.0,
        "cost": cost,
        "stop_reason": "end_turn",
        "retries": 0,
        "request_id": "req_1",
    }
    values.update(fields)
    return LoggedCall(**values)


def test_the_cost_is_the_price_on_the_day_of_the_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = Message.model_validate(WRITTEN)
    before = logged_call(
        "ask", response, "claude-haiku-5-5", 0.5, 0, None, MORNING
    )
    monkeypatch.setitem(
        PRICES, "claude-haiku-5-5", {"input": 1.0, "output": 5.0}
    )
    # A new price changes the next call's cost, never a logged one.
    assert before.cost == pytest.approx(0.0000253)
    assert day_lines([before], MORNING.date())[1] == (
        "Calls:         1, $0.000025"
    )


def test_a_day_is_added_up_by_command_and_by_model() -> None:
    calls = [
        call_at("answer", "claude-haiku-5-5", 0.0004, retries=2, seconds=9.4),
        call_at("answer", "claude-sonnet-5-5", 0.0020, seconds=3.0),
        call_at("answer", "claude-haiku-5-5", 0.0003, seconds=2.0),
        call_at("read", "claude-haiku-5-5", 0.0004, cache_read=611),
        call_at("transcribe", "gpt-transcribe", None),
    ]
    assert day_lines(calls, date(2026, 11, 20)) == [
        "Day:           2026-11-20 (UTC)",
        "Calls:         5, $0.003100",
        "Not priced:    1 (length unknown)",
        "Slowest:       answer, 9.40 s (req_1)",
        "",
        "Command      Calls  Retries  Cache read   Median        Cost",
        "answer           3        2           0    3.00s   $0.002700",
        "read             1        0         611    1.00s   $0.000400",
        "transcribe       1        0           0    1.00s   $0.000000",
        "",
        "Model                 Calls        Cost",
        "claude-sonnet-5-5         1   $0.002000",
        "claude-haiku-5-5          3   $0.001100",
        "gpt-transcribe            1   $0.000000",
    ]


def test_day_reads_the_log_of_the_day_asked_for(
    run_here: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    answering(monkeypatch, [], WRITTEN)
    main(["ask", QUESTION])
    capsys.readouterr()
    main(["day", "2026-11-20"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == [
        "Day:           2026-11-20 (UTC)",
        "Calls:         1, $0.000025",
    ]


def test_a_day_with_no_log_says_so(
    run_here: None, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stop:
        main(["day", "2026-11-19"])
    assert stop.value.code == "askkit: no calls logged on 2026-11-19 (UTC)"


def test_a_log_that_is_not_askkits_says_so(run_here: None) -> None:
    log_folder().mkdir()
    (log_folder() / "calls-2026-11-20.jsonl").write_text(
        "not json\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit) as stop:
        main(["day"])
    assert stop.value.code == "askkit: the log of 2026-11-20 is not AskKit's"


def test_a_question_that_falls_back_logs_the_retries_before_it(
    ask_set: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Haiku is overloaded twice, its tries are used up, and Sonnet answers.
    # The line names the model that answered, the one its cost is priced
    # at, and keeps the retry made on Haiku: before the fix it named the
    # response's model and counted Sonnet's retries only, so a fallback
    # day showed no retries at all.
    monkeypatch.setattr("askkit.calllog.clock", lambda: MORNING)
    steps: list[Step] = [
        (200, answer_body(), {}),
        (529, OVERLOADED, {}),
        (529, OVERLOADED, {}),
        (200, answer_body(), {}),
    ]
    answer_with(steps, monkeypatch, [], [])
    main(
        [
            "answer",
            str(ask_set),
            "--tries",
            "2",
            "--fallback",
            "claude-sonnet-5-5",
            "--out",
            "answers.json",
        ]
    )
    calls = logged()
    assert [call.model for call in calls] == [
        "claude-haiku-5-5",
        "claude-sonnet-5-5",
    ]
    assert [call.retries for call in calls] == [0, 1]
    # 410 in and 52 out at Sonnet 5.5's $2 and $10 a million tokens.
    assert calls[1].cost == pytest.approx(0.00134)


def test_a_question_that_falls_back_to_openai_keeps_its_retries_and_id(
    ask_set: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Haiku is overloaded twice and GPT-6 Luna answers. Before the fix the
    # OpenAI line had no request ID, since the answer in Claude's shape
    # lost the one OpenAI sent, and no retries, since only a Claude
    # model's line counted the retries made before it.
    monkeypatch.setattr("askkit.calllog.clock", lambda: MORNING)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-a-test-key-Cd34")
    steps: list[Step] = [
        (200, answer_body(), {}),
        (529, OVERLOADED, {}),
        (529, OVERLOADED, {}),
    ]
    answer_with(steps, monkeypatch, [], [])

    def reply(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, json=luna_body(), headers={"x-request-id": OPENAI_ID}
        )

    def make_openai(**options: Any) -> OpenAI:
        return OpenAI(
            api_key="sk-proj-a-test-key-Cd34",
            max_retries=0,
            http_client=httpx2.Client(transport=httpx2.MockTransport(reply)),
        )

    monkeypatch.setattr("askkit.cli.OpenAI", make_openai)
    main(
        [
            "answer",
            str(ask_set),
            "--tries",
            "2",
            "--fallback",
            "gpt-6-luna",
            "--out",
            "answers.json",
        ]
    )
    calls = logged()
    assert [call.model for call in calls] == [
        "claude-haiku-5-5",
        "gpt-6-luna",
    ]
    assert [call.retries for call in calls] == [0, 1]
    assert calls[1].request_id == OPENAI_ID


def test_a_turn_the_sdk_returned_is_logged_by_its_blocks() -> None:
    # A tool loop sends the model's own turn back as the SDK's block
    # objects, not dicts: they were logged as "[None]". A tool request is
    # named with its tool, never with its input.
    turn = Message.model_validate(
        {
            **WRITTEN,
            "content": [
                {"type": "text", "text": "Let me work it out."},
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "calculate",
                    "input": {"operation": "multiply", "a": 4417, "b": 0.18},
                },
            ],
        }
    )
    text = prompt_text(
        "Be brief.",
        [
            {"role": "user", "content": "What is 18% of 4,417?"},
            {"role": "assistant", "content": turn.content},
        ],
    )
    assert text == (
        "system: Be brief.\n"
        "user: What is 18% of 4,417?\n"
        "assistant: Let me work it out.\n[tool_use: calculate]"
    )


def test_both_files_write_the_moment_the_same_way(
    run_here: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    answering(monkeypatch, [], WRITTEN)
    main(["--log-content", "ask", QUESTION])
    numbers = json.loads(
        (log_folder() / "calls-2026-11-20.jsonl").read_text(encoding="utf-8")
    )
    assert content_lines()[0]["at"] == numbers["at"] == "2026-11-20T09:14:00Z"


def test_any_run_removes_content_older_than_a_week(
    run_here: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Before the fix only a run with --log-content removed old content,
    # so the words of a week ago stayed until someone asked for the words
    # again; the week is a promise every logged call keeps.
    folder = log_folder()
    folder.mkdir()
    old = folder / "content-2026-11-12.jsonl"
    old.write_text("{}\n", encoding="utf-8")
    answering(monkeypatch, [], WRITTEN)
    main(["ask", QUESTION])
    assert not old.exists()
