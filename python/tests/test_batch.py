import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic
from anthropic.types.messages import MessageBatch
from test_classify import with_answer
from test_inbox import EMAILS, MORNING, write_inbox

import askkit.cli
from askkit.batch import (
    MOST_REQUESTS,
    POLL_SECONDS,
    email_request,
    id_fault,
    status_text,
)
from askkit.calllog import day_calls
from askkit.classify import (
    CLASSIFICATION_SCHEMA,
    EMAIL_SOURCE,
    sorting_messages,
)
from askkit.clean import clean_email
from askkit.cli import main
from askkit.inbox import InboxEmail
from askkit.prompts import render
from askkit.structured import json_request
from askkit.untrusted import marked

BATCH_ID = "msgbatch_01AskKitTestBatch"
RESULTS_URL = (
    f"https://api.anthropic.com/v1/messages/batches/{BATCH_ID}/results"
)
SENT_AT = datetime(2026, 11, 20, 9, 30, tzinfo=UTC)
# The batch below took 41 minutes and 10 seconds to end.
ENDED_AT = SENT_AT + timedelta(minutes=41, seconds=10)


def batch_body(processing: int, ended: bool, **counts: int) -> dict[str, Any]:
    request_counts = {
        "processing": processing,
        "succeeded": 0,
        "errored": 0,
        "canceled": 0,
        "expired": 0,
    }
    request_counts.update(counts)
    return {
        "id": BATCH_ID,
        "type": "message_batch",
        "processing_status": "ended" if ended else "in_progress",
        "request_counts": request_counts,
        "created_at": SENT_AT.isoformat(),
        "expires_at": (SENT_AT + timedelta(hours=24)).isoformat(),
        "ended_at": ENDED_AT.isoformat() if ended else None,
        "archived_at": None,
        "cancel_initiated_at": None,
        "results_url": RESULTS_URL if ended else None,
    }


def succeeded(custom_id: str, queue: str) -> dict[str, Any]:
    message = with_answer(json.dumps({"queue": queue}))
    return {
        "custom_id": custom_id,
        "result": {"type": "succeeded", "message": message},
    }


def failed(custom_id: str, kind: str) -> dict[str, Any]:
    result: dict[str, Any] = {"type": kind}
    if kind == "errored":
        result["error"] = {
            "type": "error",
            "error": {"type": "overloaded_error", "message": "Overloaded"},
            "request_id": None,
        }
    return {"custom_id": custom_id, "result": result}


class BatchesAPI:
    # A stand-in for the Message Batches API: it keeps what was sent,
    # answers each status check with the next of statuses, and gives
    # out results, a .jsonl file, in the order they are given here.
    def __init__(
        self,
        statuses: list[dict[str, Any]],
        results: list[dict[str, Any]],
    ) -> None:
        self.created: list[dict[str, Any]] = []
        self.paths: list[str] = []
        # The SDK checks the status once more before it reads the
        # results, so the last status stays once the others are used.
        self.statuses = statuses
        self.results = results

    def answer(self, request: httpx2.Request) -> httpx2.Response:
        self.paths.append(f"{request.method} {request.url.path}")
        path = request.url.path
        if request.method == "POST" and path == "/v1/messages/batches":
            self.created.append(json.loads(request.content))
            count = len(self.created[-1]["requests"])
            return httpx2.Response(200, json=batch_body(count, False))
        if path == f"/v1/messages/batches/{BATCH_ID}":
            status = self.statuses[0]
            if len(self.statuses) > 1:
                self.statuses = self.statuses[1:]
            return httpx2.Response(200, json=status)
        if path == f"/v1/messages/batches/{BATCH_ID}/results":
            lines = "".join(json.dumps(line) + "\n" for line in self.results)
            return httpx2.Response(200, content=lines.encode())
        message = {"type": "not_found_error", "message": "Not found"}
        return httpx2.Response(404, json={"type": "error", "error": message})


@pytest.fixture
def inbox_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    monkeypatch.setattr("askkit.calllog.clock", lambda: MORNING)
    monkeypatch.setattr("askkit.cli.clock", lambda: MORNING)
    return write_inbox(tmp_path / "inbox.csv", EMAILS)


def use_batches(
    monkeypatch: pytest.MonkeyPatch, api: BatchesAPI
) -> Callable[[str], Anthropic]:
    def make(api_key: str) -> Anthropic:
        transport = httpx2.MockTransport(api.answer)
        return Anthropic(
            api_key=api_key, http_client=httpx2.Client(transport=transport)
        )

    monkeypatch.setattr("askkit.cli.Anthropic", make)
    return make


def test_one_request_sends_every_email_named_by_its_id(
    inbox_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = BatchesAPI([], [])
    use_batches(monkeypatch, api)
    main(["inbox", str(inbox_file), "--batch"])
    assert api.paths == ["POST /v1/messages/batches"]
    requests = api.created[0]["requests"]
    assert [request["custom_id"] for request in requests] == [
        "e0001",
        "e0002",
        "e0003",
    ]
    assert capsys.readouterr().out.splitlines() == [
        f"Batch:         {BATCH_ID}",
        "Emails:        3 of 3, in one batch",
        "Status:        in_progress, 3 of 3 still processing",
        "Expires:       2026-11-21 09:30 UTC",
        f"Results:       askkit batch {BATCH_ID}",
    ]


def test_a_batch_request_holds_what_askkit_inbox_sends_for_it(
    inbox_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = BatchesAPI([], [])
    use_batches(monkeypatch, api)
    main(["inbox", str(inbox_file), "--batch", "--first", "1"])
    cleaned, _ = clean_email(EMAILS[0][1])
    expected = json_request(
        "claude-haiku-5-5",
        1000,
        render("classify_email", {}),
        sorting_messages(cleaned),
        CLASSIFICATION_SCHEMA,
    )
    assert api.created[0]["requests"] == [
        {"custom_id": "e0001", "params": expected}
    ]


def test_an_id_the_api_cannot_take_stops_before_anything_is_sent(
    tmp_path: Path, inbox_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = BatchesAPI([], [])
    use_batches(monkeypatch, api)
    path = write_inbox(tmp_path / "spaces.csv", [("order 4417", "Hi")])
    with pytest.raises(SystemExit) as stop:
        main(["inbox", str(path), "--batch"])
    assert stop.value.code == (
        "askkit: the id 'order 4417' cannot name a batch request; an id"
        " is 1 to 64 letters, digits, - or _"
    )
    assert api.paths == []


def test_an_id_used_twice_is_refused() -> None:
    emails = [InboxEmail("e0001", "Hi"), InboxEmail("e0001", "Hello")]
    assert id_fault(emails) == "the id e0001 is in the file twice"


def test_more_emails_than_one_batch_may_hold_are_refused() -> None:
    emails = [InboxEmail("e1", "Hi")] * (MOST_REQUESTS + 1)
    assert id_fault(emails) == (
        "100,001 emails are more than the 100,000 one batch may hold;"
        " send them with --first in parts"
    )


def test_ids_of_64_letters_digits_hyphens_and_underscores_are_taken() -> None:
    emails = [InboxEmail("a" * 64, "Hi"), InboxEmail("e-1_B", "Hi")]
    assert id_fault(emails) is None
    assert id_fault([InboxEmail("a" * 65, "Hi")]) is not None
    assert id_fault([InboxEmail("", "Hi")]) is not None


def test_a_batch_or_a_number_at_once_never_both(
    inbox_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stop:
        main(["inbox", str(inbox_file), "--batch", "--at-once", "4"])
    assert stop.value.code == 2
    assert "--at-once: not allowed with argument --batch" in (
        capsys.readouterr().err
    )


def test_a_batch_takes_max_tokens_a_call_that_waits_may_not(
    inbox_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = BatchesAPI([], [])
    use_batches(monkeypatch, api)
    main(["inbox", str(inbox_file), "--batch", "--max-tokens", "64000"])
    params = api.created[0]["requests"][0]["params"]
    assert params["max_tokens"] == 64000


def test_a_batch_still_processing_says_so_and_reads_no_results(
    inbox_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = BatchesAPI([batch_body(2, False, succeeded=0)], [])
    use_batches(monkeypatch, api)
    main(["batch", BATCH_ID])
    assert api.paths == [f"GET /v1/messages/batches/{BATCH_ID}"]
    assert capsys.readouterr().out.splitlines() == [
        f"Batch:         {BATCH_ID}",
        "Status:        in_progress, 2 of 2 still processing",
        "Results:       not yet; run this again later, or add --wait",
    ]


def test_wait_checks_once_a_minute_until_the_batch_ends(
    inbox_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = BatchesAPI(
        [
            batch_body(2, False),
            batch_body(2, False),
            batch_body(0, True, succeeded=2),
        ],
        [succeeded("e0001", "damaged"), succeeded("e0002", "delivery")],
    )
    use_batches(monkeypatch, api)
    slept: list[float] = []
    monkeypatch.setattr("askkit.cli.sleep", slept.append)
    main(["batch", BATCH_ID, "--wait"])
    assert slept == [POLL_SECONDS, POLL_SECONDS]
    assert POLL_SECONDS == 60
    captured = capsys.readouterr()
    assert captured.err.splitlines() == [
        "Status:        in_progress, 2 of 2 still processing",
        "Status:        in_progress, 2 of 2 still processing",
    ]
    assert captured.out.splitlines()[:2] == [
        "e0001   damaged",
        "e0002   delivery",
    ]


def test_results_in_any_order_each_go_to_their_own_email(
    inbox_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The file gives the results last email first: each queue must
    # still go with the email its custom_id names.
    api = BatchesAPI(
        [batch_body(0, True, succeeded=3)],
        [
            succeeded("e0003", "other"),
            succeeded("e0001", "damaged"),
            succeeded("e0002", "delivery"),
        ],
    )
    use_batches(monkeypatch, api)
    main(["batch", BATCH_ID])
    assert capsys.readouterr().out.splitlines() == [
        "e0003   other",
        "e0001   damaged",
        "e0002   delivery",
        "",
        f"Batch:         {BATCH_ID}",
        "Emails:        3 sorted, 0 not sorted",
        "Queues:        damaged 1, delivery 1, other 1",
        "Model:         claude-haiku-5-5",
        "Tokens:        642 in, 36 out",
        "Cost:          $0.000041, 50% of $0.000082 at standard prices",
        "Time:          about 41 minutes",
    ]


def test_errored_and_expired_requests_are_named_and_not_billed(
    inbox_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = BatchesAPI(
        [batch_body(0, True, succeeded=1, errored=1, expired=1)],
        [
            failed("e0002", "errored"),
            succeeded("e0001", "damaged"),
            failed("e0003", "expired"),
        ],
    )
    use_batches(monkeypatch, api)
    main(["batch", BATCH_ID])
    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == [
        "e0002   not sorted: errored (overloaded_error)",
        "e0001   damaged",
        "e0003   not sorted: expired",
    ]
    assert "Emails:        1 sorted, 2 not sorted" in lines
    assert (
        "Not sorted:    e0002 errored (overloaded_error), e0003 expired"
        in lines
    )
    # Only the answered request is priced: 214 in, 12 out, at half.
    assert (
        "Cost:          $0.000014, 50% of $0.000027 at standard prices"
        in lines
    )


def test_a_batch_with_no_answer_prints_no_cost(
    inbox_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api = BatchesAPI(
        [batch_body(0, True, expired=1)], [failed("e0001", "expired")]
    )
    use_batches(monkeypatch, api)
    main(["batch", BATCH_ID])
    lines = capsys.readouterr().out.splitlines()
    assert "Emails:        0 sorted, 1 not sorted" in lines
    assert not any(line.startswith("Cost:") for line in lines)


def test_each_answer_is_logged_once_at_the_batch_price(
    inbox_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two runs of askkit batch on the same ended batch: the second shows
    # the queues again, but the log keeps the two answers once.
    def results() -> list[dict[str, Any]]:
        return [succeeded("e0002", "delivery"), succeeded("e0001", "damaged")]

    for _ in range(2):
        api = BatchesAPI([batch_body(0, True, succeeded=2)], results())
        use_batches(monkeypatch, api)
        main(["batch", BATCH_ID])
    log_folder: Path = vars(askkit.cli)["LOG_FOLDER"]
    calls = day_calls(log_folder, MORNING.date())
    assert [call.command for call in calls] == ["batch", "batch"]
    assert [call.cost for call in calls] == [0.0000137, 0.0000137]
    assert [call.seconds for call in calls] == [2470.0, 2470.0]
    assert [call.request_id for call in calls] == [None, None]


def test_with_log_content_a_batch_answer_names_its_request(
    inbox_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = BatchesAPI(
        [batch_body(0, True, succeeded=1)], [succeeded("e0001", "damaged")]
    )
    use_batches(monkeypatch, api)
    main(["--log-content", "batch", BATCH_ID])
    log_folder: Path = vars(askkit.cli)["LOG_FOLDER"]
    path = log_folder / f"content-{MORNING.date().isoformat()}.jsonl"
    kept = json.loads(path.read_text(encoding="utf-8"))
    assert kept["prompt"] == f"[request e0001 of batch {BATCH_ID}]"
    assert kept["answer"] == '{"queue": "damaged"}'


def test_a_batch_id_the_workspace_does_not_have_is_named(
    inbox_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_batches(monkeypatch, BatchesAPI([], []))
    with pytest.raises(SystemExit) as stop:
        main(["batch", "msgbatch_01NotOurs"])
    assert stop.value.code == (
        "askkit: no batch msgbatch_01NotOurs in this workspace (404);"
        " check the ID askkit inbox --batch printed"
    )


def test_status_counts_every_request_of_the_batch() -> None:
    batch = MessageBatch.model_validate(batch_body(5, False, succeeded=7))
    assert status_text(batch) == "in_progress, 5 of 12 still processing"


def test_an_email_request_cleans_the_quoted_part_away() -> None:
    email = InboxEmail("e0003", EMAILS[2][1])
    request = email_request(email, "claude-haiku-5-5", 1000, "Sort it.")
    assert request["custom_id"] == "e0003"
    messages = list(request["params"]["messages"])
    assert messages == [
        {
            "role": "user",
            "content": marked(EMAIL_SOURCE, "Thank you, the vase is lovely!"),
        }
    ]
