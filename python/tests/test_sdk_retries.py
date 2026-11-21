import asyncio
import json
from pathlib import Path
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic, AsyncAnthropic
from test_ask import WRITTEN, replied
from test_calllog import MORNING, QUESTION, logged, run_here
from test_classify import with_answer
from test_invoices import FIELDS, INVOICE
from test_retry import OVERLOADED

from askkit.calllog import day_calls, start_log
from askkit.cli import main
from askkit.sdk_retries import RetryCount
from askkit.structured import ask_json
from askkit.trim import summarise

__all__ = ["run_here"]

KEY = "sk-ant-api03-a-test-key-Ab12"
SCHEMA = {"type": "object", "properties": {"queue": {"type": "string"}}}


def overloaded() -> httpx2.Response:
    # A 529 that asks for a wait of one millisecond (retry-after-ms, which
    # the SDK reads before retry-after), so a test does not wait half a
    # second for the SDK's first backoff.
    return httpx2.Response(
        529, json=OVERLOADED, headers={"retry-after-ms": "1"}
    )


def busy_then(
    sent: list[dict[str, Any]], body: dict[str, Any], busy: int
) -> Any:
    # The API is overloaded for the first `busy` requests, then answers.
    def answer(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        if len(sent) <= busy:
            return overloaded()
        return replied(request, body)

    return answer


def client_for(answer: Any) -> Anthropic:
    return Anthropic(
        api_key=KEY,
        http_client=httpx2.Client(transport=httpx2.MockTransport(answer)),
    )


def messages_create(client: Anthropic) -> Any:
    return client.messages.create(
        model="claude-haiku-5-5",
        max_tokens=100,
        messages=[{"role": "user", "content": QUESTION}],
    )


@pytest.mark.parametrize("busy", [0, 1, 2])
def test_the_count_is_the_retries_before_the_answer(busy: int) -> None:
    sent: list[dict[str, Any]] = []
    count = RetryCount()
    client = client_for(busy_then(sent, WRITTEN, busy))
    messages_create(client.with_middleware(count))
    assert len(sent) == busy + 1
    assert count.retries == busy


def test_a_call_that_fails_after_every_retry_is_counted_too() -> None:
    # The SDK's default: two retries, three requests, then the error.
    sent: list[dict[str, Any]] = []
    count = RetryCount()
    client = client_for(busy_then(sent, WRITTEN, busy=3))
    with pytest.raises(Exception, match="Overloaded"):
        messages_create(client.with_middleware(count))
    assert len(sent) == 3
    assert count.retries == 2


def test_the_count_belongs_to_one_call_not_to_the_client() -> None:
    # with_middleware gives a copy of the client; the client itself keeps
    # no middleware, so a second call starts its own count.
    sent: list[dict[str, Any]] = []
    client = client_for(busy_then(sent, WRITTEN, busy=1))
    first = RetryCount()
    messages_create(client.with_middleware(first))
    second = RetryCount()
    messages_create(client.with_middleware(second))
    assert (first.retries, second.retries) == (1, 0)
    assert len(sent) == 3


def answering_after_a_529(
    monkeypatch: pytest.MonkeyPatch,
    sent: list[dict[str, Any]],
    body: dict[str, Any],
) -> None:
    # Every client the CLI makes is overloaded once, then answers.
    answer = busy_then(sent, body, busy=1)

    def make(**options: Any) -> Anthropic:
        return client_for(answer)

    monkeypatch.setattr("askkit.cli.Anthropic", make)


@pytest.mark.parametrize(
    "arguments, body",
    [
        (["ask", QUESTION], WRITTEN),
        (["read", str(INVOICE)], with_answer(json.dumps(FIELDS))),
        (["classify", "gift.txt"], with_answer('{"queue":"damaged"}')),
    ],
)
def test_the_log_counts_the_retries_the_sdk_made(
    run_here: None,
    monkeypatch: pytest.MonkeyPatch,
    arguments: list[str],
    body: dict[str, Any],
) -> None:
    # The mechanism, whatever path a command's call takes: the SDK sent
    # the request twice, so its line says one retry, not 0.
    Path("gift.txt").write_text(QUESTION, encoding="utf-8")
    sent: list[dict[str, Any]] = []
    answering_after_a_529(monkeypatch, sent, body)
    main(arguments)
    [call] = logged()
    assert len(sent) == 2
    assert call.retries == 1


def test_a_chat_summary_logs_its_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("askkit.calllog.clock", lambda: MORNING)
    start_log(tmp_path, "chat", content=False)
    sent: list[dict[str, Any]] = []
    summarise(
        client_for(busy_then(sent, WRITTEN, busy=1)),
        "claude-haiku-5-5",
        "",
        [],
    )
    [call] = day_calls(tmp_path, MORNING.date())
    assert call.retries == 1


def test_an_inbox_call_logs_its_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # askkit inbox sends its emails through ask_json on the async client.
    monkeypatch.setattr("askkit.calllog.clock", lambda: MORNING)
    start_log(tmp_path, "inbox", content=False)
    sent: list[dict[str, Any]] = []
    answer = busy_then(sent, with_answer('{"queue":"damaged"}'), busy=2)
    client = AsyncAnthropic(
        api_key=KEY,
        http_client=httpx2.AsyncClient(
            transport=httpx2.MockTransport(answer)
        ),
    )
    messages: list[Any] = [{"role": "user", "content": QUESTION}]
    model = "claude-haiku-5-5"
    asyncio.run(ask_json(client, model, 100, "Sort it.", messages, SCHEMA))
    [call] = day_calls(tmp_path, MORNING.date())
    assert call.retries == 2
