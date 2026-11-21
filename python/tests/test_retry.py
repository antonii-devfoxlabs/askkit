import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic import Anthropic
from anthropic.types import Message

from askkit.ask import ask
from askkit.cli import main
from askkit.retry import (
    ANSWER_POLICY,
    Retry,
    RetryPolicy,
    retries_line,
    timeout_fault,
    wait_before,
    with_retries,
)

REQUEST_ID = "req_011CAskKitRetry"
POLICY = RetryPolicy(tries=5, first_wait=1.0, max_wait=30.0, deadline=120.0)
MAIN = Path(__file__).parent.parent / "evals" / "ask" / "main"


def error_body(error_type: str, message: str) -> dict[str, Any]:
    return {
        "type": "error",
        "error": {"type": error_type, "message": message},
        "request_id": REQUEST_ID,
    }


OVERLOADED = error_body("overloaded_error", "Overloaded")


def answer_body() -> dict[str, Any]:
    return {
        "id": "msg_01AskKitRetry",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": "<answer>We do.</answer>"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 410, "output_tokens": 52},
    }


Step = (
    tuple[int, dict[str, Any], dict[str, str]] | type[httpx2.TransportError]
)


def scripted_client(steps: list[Step], sent: list[int]) -> Anthropic:
    # Each request gets the next step: a status, its body and headers, or
    # a network failure raised instead of any response. The SDK's own
    # retries are off, as askkit answer turns them off.
    def reply(request: httpx2.Request) -> httpx2.Response:
        sent.append(len(sent) + 1)
        step = steps[min(len(sent), len(steps)) - 1]
        if isinstance(step, type):
            raise step("no answer", request=request)
        status, body, headers = step
        return httpx2.Response(
            status, json=body, headers={"request-id": REQUEST_ID, **headers}
        )

    return Anthropic(
        api_key="sk-ant-api03-a-test-key-Ab12",
        max_retries=0,
        http_client=httpx2.Client(transport=httpx2.MockTransport(reply)),
    )


def sender(client: Anthropic) -> Callable[[], Message]:
    def send() -> Message:
        return ask(client, "Who pays?", "claude-haiku-5-5", 1000, "Be brief.")

    return send


class FakeTime:
    # A clock that moves only when AskKit sleeps, so no test waits.
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def run(
    steps: list[Step], policy: RetryPolicy = POLICY
) -> tuple[list[int], list[Retry], FakeTime, Message | anthropic.APIError]:
    sent: list[int] = []
    retries: list[Retry] = []
    fake = FakeTime()
    result: Message | anthropic.APIError
    try:
        result = with_retries(
            sender(scripted_client(steps, sent)),
            policy,
            retries,
            sleep=fake.sleep,
            draw=lambda: 0.5,
        )
    except anthropic.APIError as error:
        result = error
    return sent, retries, fake, result


def test_an_overloaded_api_is_tried_again_until_it_answers() -> None:
    sent, retries, fake, result = run(
        [
            (529, OVERLOADED, {}),
            (529, OVERLOADED, {}),
            (200, answer_body(), {}),
        ]
    )
    assert isinstance(result, Message)
    assert len(sent) == 3
    # Half of the longest wait: 1 s before the first retry, 2 s before
    # the second.
    assert retries == [Retry("529", 0.5), Retry("529", 1.0)]
    assert fake.slept == [0.5, 1.0]


def test_the_api_s_retry_after_is_obeyed() -> None:
    limited = error_body("rate_limit_error", "Too many requests.")
    _, retries, fake, result = run(
        [(429, limited, {"retry-after": "7"}), (200, answer_body(), {})]
    )
    assert isinstance(result, Message)
    assert retries == [Retry("429", 7.0)]
    assert fake.slept == [7.0]


def test_a_rejected_key_is_sent_once() -> None:
    rejected = error_body("authentication_error", "invalid x-api-key")
    sent, retries, _, result = run([(401, rejected, {})])
    assert isinstance(result, anthropic.AuthenticationError)
    assert sent == [1]
    assert retries == []


def test_the_api_saying_no_retry_wins_over_the_status() -> None:
    sent, _, _, result = run([(529, OVERLOADED, {"x-should-retry": "false"})])
    assert isinstance(result, anthropic.OverloadedError)
    assert sent == [1]


def test_a_spent_monthly_cap_is_not_tried_again() -> None:
    body = error_body("rate_limit_error", "You will regain access on 1 Dec.")
    body["error"]["details"] = {"error_code": "enforced_spend_limit_reached"}
    sent, _, _, result = run([(429, body, {})])
    assert isinstance(result, anthropic.EnforcedSpendLimitReachedError)
    assert sent == [1]


def test_every_try_used_raises_the_last_error() -> None:
    policy = RetryPolicy(tries=3, first_wait=1.0, max_wait=30.0, deadline=120)
    sent, retries, _, result = run([(529, OVERLOADED, {})], policy)
    assert isinstance(result, anthropic.OverloadedError)
    assert sent == [1, 2, 3]
    assert len(retries) == 2


def test_no_wait_ends_past_the_deadline() -> None:
    # The API asks for three minutes; the deadline is two. Waiting would
    # only fail later, so the error is raised at once.
    limited = error_body("rate_limit_error", "Too many requests.")
    sent, retries, fake, result = run(
        [(429, limited, {"retry-after": "180"})]
    )
    assert isinstance(result, anthropic.RateLimitError)
    assert sent == [1]
    assert retries == []
    assert fake.slept == []


def test_a_request_that_hangs_times_out_and_is_tried_again() -> None:
    sent, retries, _, result = run(
        [httpx2.ReadTimeout, (200, answer_body(), {})]
    )
    assert isinstance(result, Message)
    assert len(sent) == 2
    assert retries == [Retry("timeout", 0.5)]


def test_no_connection_is_tried_again() -> None:
    _, retries, _, result = run(
        [httpx2.ConnectError, (200, answer_body(), {})]
    )
    assert isinstance(result, Message)
    assert retries == [Retry("no connection", 0.5)]


def test_the_longest_wait_doubles_up_to_its_limit() -> None:
    error = anthropic.APIConnectionError(
        request=httpx2.Request("POST", "https://api.anthropic.com")
    )
    longest = []
    for retry in range(7):
        longest.append(wait_before(retry, POLICY, error, draw=lambda: 1.0))
    assert longest == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]
    # A draw of 0 waits not at all: full jitter goes from 0 to the most.
    assert wait_before(3, POLICY, error, draw=lambda: 0.0) == 0.0


def test_settings_that_cannot_work_are_refused_before_any_call() -> None:
    assert timeout_fault(5, 60, 1000) is None
    assert timeout_fault(0, 60, 1000) == (
        "--tries 0 is below one; give 1 to never try again"
    )
    assert timeout_fault(5, 900, 1000) == (
        "--timeout 900 is not from 1 to 600 seconds; a call that may take"
        " longer than ten minutes should be streamed"
    )
    # 4,000 tokens at the SDK's slowest pace take 112.5 s.
    assert timeout_fault(5, 60, 4000) == (
        "--max-tokens 4,000 may take up to 112 s, longer than the --timeout"
        " of 60 s;\ngive a smaller --max-tokens or a longer --timeout"
    )


def test_the_retries_line_counts_reasons_and_waiting() -> None:
    line = retries_line([Retry("529", 0.62), Retry("timeout", 1.71)])
    assert line == "Retries:       2 (529, timeout), 2.3 s waiting"


@pytest.fixture
def ask_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    part = tmp_path / "main"
    part.mkdir()
    cases = json.loads((MAIN / "cases.json").read_text(encoding="utf-8"))
    (part / "cases.json").write_text(json.dumps(cases[:2]), encoding="utf-8")
    documents = tmp_path / "documents"
    documents.mkdir()
    for path in (MAIN.parent / "documents").iterdir():
        (documents / path.name).write_bytes(path.read_bytes())
    return part


def answer_with(
    steps: list[Step],
    monkeypatch: pytest.MonkeyPatch,
    made: list[dict[str, Any]],
    sent: list[int],
) -> None:
    def make(**options: Any) -> Anthropic:
        made.append(options)
        return scripted_client(steps, sent)

    def no_waiting(
        send: Callable[[], Message], policy: RetryPolicy, retries: list[Retry]
    ) -> Message:
        fake = FakeTime()
        return with_retries(send, policy, retries, fake.sleep, lambda: 0.5)

    monkeypatch.setattr("askkit.cli.Anthropic", make)
    monkeypatch.setattr("askkit.cli.with_retries", no_waiting)


def test_answer_retries_one_question_not_the_run(
    ask_set: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    made: list[dict[str, Any]] = []
    sent: list[int] = []
    steps: list[Step] = [
        (200, answer_body(), {}),
        (529, OVERLOADED, {}),
        httpx2.ReadTimeout,
        (200, answer_body(), {}),
    ]
    answer_with(steps, monkeypatch, made, sent)
    main(["answer", str(ask_set), "--out", "answers.json"])
    lines = capsys.readouterr().out.splitlines()
    # The first answer is sent once; only the second question is tried
    # again, and both answers are saved.
    assert len(sent) == 4
    assert "Retries:       2 (529, timeout), 1.5 s waiting" in lines
    assert len(json.loads(Path("answers.json").read_text())) == 2
    # AskKit owns the retries, with the timeout of --timeout.
    assert made[0]["max_retries"] == 0
    assert made[0]["timeout"] == httpx2.Timeout(60.0, connect=5.0)


def test_answer_says_how_many_tries_it_made_before_giving_up(
    ask_set: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[dict[str, Any]] = []
    sent: list[int] = []
    answer_with([(529, OVERLOADED, {})], monkeypatch, made, sent)
    with pytest.raises(SystemExit) as stopped:
        main(["answer", str(ask_set), "--out", "answers.json"])
    assert stopped.value.code == (
        "askkit: postage: Anthropic's servers are overloaded (529) after 5"
        " tries;\ntry again in a few minutes"
        f" (request ID {REQUEST_ID})"
    )
    assert len(sent) == ANSWER_POLICY.tries
    assert not Path("answers.json").exists()


def test_answer_takes_its_number_of_tries(
    ask_set: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[dict[str, Any]] = []
    sent: list[int] = []
    answer_with([(529, OVERLOADED, {})], monkeypatch, made, sent)
    with pytest.raises(SystemExit) as stopped:
        main(["answer", str(ask_set), "--out", "a.json", "--tries", "1"])
    assert "overloaded (529) after 1 try;" in str(stopped.value.code)
    assert sent == [1]
