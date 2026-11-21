import json
from pathlib import Path
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic import AsyncAnthropic
from test_ask import replied
from test_batch import (
    BatchesAPI,
    batch_body,
    failed,
    succeeded,
    use_batches,
)
from test_classify import with_answer
from test_inbox import (
    EMAILS,
    KEY_REJECTED,
    MORNING,
    QUEUES,
    REJECTED,
    answering,
    write_inbox,
)

from askkit.checkpoint import (
    read_checkpoint,
    record_sorted,
    start_checkpoint,
)
from askkit.cli import main
from askkit.errors import error_name, stops_the_run
from askkit.inbox import progress_step, progress_text

# Twelve emails, each a broken mug: enough for a tenth of the run to be
# more than one email.
TWELVE = [
    (f"e{number:04}", f"Order {4400 + number} came with a broken mug.")
    for number in range(1, 13)
]
DAMAGED = '{"queue":"damaged"}'


@pytest.fixture
def inbox_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    monkeypatch.setattr("askkit.calllog.clock", lambda: MORNING)
    monkeypatch.setattr("askkit.cli.clock", lambda: MORNING)
    return write_inbox(tmp_path / "inbox.csv", EMAILS)


def lines_of(path: Path) -> list[dict[str, str]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def status_error(status: int, kind: str) -> anthropic.APIStatusError:
    # The error the SDK raises for a response of this status.
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"type": "error", "error": {"type": kind, "message": "x"}}
    response = httpx2.Response(status, json=body, request=request)
    client = anthropic.Anthropic(api_key="sk-ant-api03-a-test-key-Ab12")
    return client._make_status_error("x", body=body, response=response)


def test_no_checkpoint_file_yet_means_nothing_is_sorted(
    tmp_path: Path,
) -> None:
    assert read_checkpoint(tmp_path / "sorted.jsonl") == {}


def test_each_line_names_an_email_and_its_queue(tmp_path: Path) -> None:
    path = tmp_path / "sorted.jsonl"
    record_sorted(path, "e0001", "damaged")
    record_sorted(path, "e0002", "delivery")
    assert path.read_text(encoding="utf-8") == (
        '{"id": "e0001", "queue": "damaged"}\n'
        '{"id": "e0002", "queue": "delivery"}\n'
    )
    assert read_checkpoint(path) == {"e0001": "damaged", "e0002": "delivery"}


def test_a_line_cut_off_by_a_crash_is_skipped(tmp_path: Path) -> None:
    path = tmp_path / "sorted.jsonl"
    path.write_text(
        '{"id": "e0001", "queue": "damaged"}\n{"id": "e0002", "que',
        encoding="utf-8",
    )
    assert read_checkpoint(path) == {"e0001": "damaged"}


def test_after_a_cut_off_line_the_next_record_starts_a_line_of_its_own(
    tmp_path: Path,
) -> None:
    # Without the line break, e0002's record would be glued to the cut
    # off part and lost with it, and e0002 sent at every run after.
    path = tmp_path / "sorted.jsonl"
    path.write_text(
        '{"id": "e0001", "queue": "damaged"}\n{"id": "e0003", "que',
        encoding="utf-8",
    )
    assert start_checkpoint(path) == {"e0001": "damaged"}
    record_sorted(path, "e0002", "delivery")
    assert read_checkpoint(path) == {"e0001": "damaged", "e0002": "delivery"}


def test_a_progress_line_comes_after_each_tenth_of_the_emails() -> None:
    assert progress_step(1000) == 100
    assert progress_step(12) == 2
    assert progress_step(3) == 1


def test_the_time_left_assumes_the_rest_take_as_long_as_the_first() -> None:
    assert progress_text(100, 1000, 0, 14.0) == (
        "100 of 1,000 done, about 2 minutes left"
    )
    assert progress_text(4, 12, 1, 2.8) == (
        "4 of 12 done, 1 not sorted, about 6 seconds left"
    )


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
    ],
)
def test_an_error_about_the_key_or_the_model_stops_the_run(
    status: int, kind: str
) -> None:
    assert stops_the_run(status_error(status, kind))


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, "invalid_request_error"),
        (413, "request_too_large"),
        (429, "rate_limit_error"),
        (500, "api_error"),
        (529, "overloaded_error"),
    ],
)
def test_an_error_about_one_request_or_one_moment_does_not(
    status: int, kind: str
) -> None:
    error = status_error(status, kind)
    assert not stops_the_run(error)
    assert error_name(error) == f"{status} {kind}"


def test_billing_and_your_own_spend_limit_stop_the_run() -> None:
    assert stops_the_run(status_error(402, "billing_error"))
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    message = "You have reached your specified API usage limits. You will"
    body = {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": message},
    }
    response = httpx2.Response(400, json=body, request=request)
    limit = anthropic.BadRequestError("x", response=response, body=body)
    assert stops_the_run(limit)


def test_a_spent_monthly_cap_stops_the_run() -> None:
    error = status_error(429, "rate_limit_error")
    capped = anthropic.EnforcedSpendLimitReachedError(
        "x", response=error.response, body=error.body
    )
    assert stops_the_run(capped)


def test_each_sorted_email_is_written_down(
    inbox_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sent: list[dict[str, Any]] = []
    answering(monkeypatch, sent, [with_answer(queue) for queue in QUEUES])
    main(["inbox", str(inbox_file), "--checkpoint", "sorted.jsonl"])
    assert lines_of(inbox_file.parent / "sorted.jsonl") == [
        {"id": "e0001", "queue": "damaged"},
        {"id": "e0002", "queue": "delivery"},
        {"id": "e0003", "queue": "other"},
    ]
    assert capsys.readouterr().out.splitlines()[-1] == (
        "Checkpoint:    3 of 3 sorted, in sorted.jsonl"
    )


def test_the_next_run_sends_only_the_emails_not_written_down(
    inbox_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    record_sorted(inbox_file.parent / "sorted.jsonl", "e0001", "damaged")
    sent: list[dict[str, Any]] = []
    answering(monkeypatch, sent, [with_answer(queue) for queue in QUEUES[1:]])
    main(["inbox", str(inbox_file), "--checkpoint", "sorted.jsonl"])
    lines = capsys.readouterr().out.splitlines()
    assert [request["messages"][0]["content"] for request in sent] == [
        EMAILS[1][1],
        "Thank you, the vase is lovely!",
    ]
    assert lines[0] == (
        "Checkpoint:    1 of 3 already sorted; sending the other 2"
    )
    assert lines[4] == "Emails:        2 of 3, 10 at once"
    assert lines[-1] == "Checkpoint:    3 of 3 sorted, in sorted.jsonl"


def test_a_job_already_done_sends_nothing(
    inbox_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = inbox_file.parent / "sorted.jsonl"
    for (email_id, _), queue in zip(
        EMAILS, ["damaged", "delivery", "other"], strict=True
    ):
        record_sorted(path, email_id, queue)
    sent: list[dict[str, Any]] = []
    answering(monkeypatch, sent, [])
    main(["inbox", str(inbox_file), "--checkpoint", "sorted.jsonl"])
    assert sent == []
    assert capsys.readouterr().out.splitlines() == [
        "Checkpoint:    3 of 3 already sorted; sending the other 0",
        "Emails:        none left to send",
    ]


def test_an_email_left_unsorted_is_sent_again_by_the_next_run(
    inbox_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Two runs of the same command: the first leaves e0002 unsorted
    # (400), the second sends e0002 alone and finishes the job.
    sent: list[dict[str, Any]] = []
    answering(
        monkeypatch,
        sent,
        [with_answer(QUEUES[0]), REJECTED, with_answer(QUEUES[2])],
    )
    command = [
        "inbox",
        str(inbox_file),
        "--at-once",
        "1",
        "--checkpoint",
        "sorted.jsonl",
    ]
    main(command)
    first = capsys.readouterr().out.splitlines()
    assert first[-1] == (
        "Checkpoint:    2 of 3 sorted, in sorted.jsonl; run the same"
        " command to send the 1 left"
    )
    again: list[dict[str, Any]] = []
    answering(monkeypatch, again, [with_answer(QUEUES[1])])
    main(command)
    assert [request["messages"][0]["content"] for request in again] == [
        EMAILS[1][1]
    ]
    assert capsys.readouterr().out.splitlines()[-1] == (
        "Checkpoint:    3 of 3 sorted, in sorted.jsonl"
    )


def test_after_a_crash_the_next_run_sends_only_the_rest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The run is stopped (Ctrl+C) while the fourth answer is on its way:
    # the three answered emails are in the checkpoint, and the same
    # command then sends the other nine and no more.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    monkeypatch.setattr("askkit.calllog.clock", lambda: MORNING)
    monkeypatch.setattr("askkit.cli.clock", lambda: MORNING)
    inbox = write_inbox(tmp_path / "inbox.csv", TWELVE)
    sent: list[str] = []

    async def answer(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content)["messages"][0]["content"])
        if len(sent) == 4:
            raise KeyboardInterrupt
        return replied(request, with_answer(DAMAGED))

    def make(api_key: str) -> AsyncAnthropic:
        transport = httpx2.MockTransport(answer)
        return AsyncAnthropic(
            api_key=api_key,
            http_client=httpx2.AsyncClient(transport=transport),
        )

    monkeypatch.setattr("askkit.cli.AsyncAnthropic", make)
    command = [
        "inbox",
        str(inbox),
        "--at-once",
        "1",
        "--checkpoint",
        "sorted.jsonl",
    ]
    with pytest.raises(SystemExit) as stopped:
        main(command)
    assert stopped.value.code == (
        "askkit: cancelled; 3 of 12 sorted, in sorted.jsonl; run the same"
        " command to send the 9 left"
    )
    again: list[dict[str, Any]] = []
    answering(monkeypatch, again, [with_answer(DAMAGED)] * 9)
    main(command)
    assert [request["messages"][0]["content"] for request in again] == [
        text for _, text in TWELVE[3:]
    ]
    assert capsys.readouterr().out.splitlines()[-1] == (
        "Checkpoint:    12 of 12 sorted, in sorted.jsonl"
    )


def test_progress_goes_to_standard_error_after_each_tenth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Twelve emails: a line after every second one, none after the last,
    # where the summary says it all. The clock is read at the start and
    # at each line, 0.7 s later each time: after 2 emails, 0.7 s have
    # passed, so the other 10 should take 3.5 s; after 4, 1.4 s, so 2.8.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    inbox = write_inbox(tmp_path / "inbox.csv", TWELVE)
    ticks = iter(range(100))
    monkeypatch.setattr("askkit.cli.monotonic", lambda: next(ticks) * 0.7)
    sent: list[dict[str, Any]] = []
    answering(monkeypatch, sent, [with_answer(DAMAGED)] * 12)
    main(["inbox", str(inbox), "--at-once", "1"])
    captured = capsys.readouterr()
    assert captured.err.splitlines() == [
        "Progress:      2 of 12 done, about 4 seconds left",
        "Progress:      4 of 12 done, about 3 seconds left",
        "Progress:      6 of 12 done, about 2 seconds left",
        "Progress:      8 of 12 done, about 1 second left",
        "Progress:      10 of 12 done, about 1 second left",
    ]
    assert "Progress:" not in captured.out


def test_an_error_that_stops_the_run_writes_nothing_more_down(
    inbox_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []
    answering(monkeypatch, sent, [with_answer(QUEUES[0]), KEY_REJECTED])
    with pytest.raises(SystemExit):
        main(
            [
                "inbox",
                str(inbox_file),
                "--at-once",
                "1",
                "--checkpoint",
                "sorted.jsonl",
            ]
        )
    assert len(sent) == 2
    assert lines_of(inbox_file.parent / "sorted.jsonl") == [
        {"id": "e0001", "queue": "damaged"}
    ]


def test_a_checkpoint_in_a_missing_folder_is_refused_before_any_call(
    inbox_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []
    answering(monkeypatch, sent, [])
    with pytest.raises(SystemExit) as stopped:
        main(["inbox", str(inbox_file), "--checkpoint", "runs/sorted.jsonl"])
    assert stopped.value.code == (
        "askkit: no folder runs for the checkpoint; make it first"
    )
    assert sent == []


def test_a_batch_writes_down_what_it_sorted_and_a_new_batch_sends_the_rest(
    inbox_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The batch sorts e0001 and e0003; e0002 expired. The next
    # askkit inbox --batch with the same checkpoint sends e0002 alone.
    api = BatchesAPI(
        [batch_body(0, True, succeeded=2, expired=1)],
        [
            succeeded("e0003", "other"),
            failed("e0002", "expired"),
            succeeded("e0001", "damaged"),
        ],
    )
    use_batches(monkeypatch, api)
    batch_id = "msgbatch_01AskKitTestBatch"
    main(["batch", batch_id, "--checkpoint", "sorted.jsonl"])
    assert capsys.readouterr().out.splitlines()[-1] == (
        "Checkpoint:    2 written to sorted.jsonl, for askkit inbox"
        " --checkpoint"
    )
    # Collected twice, the batch writes nothing more down.
    main(["batch", batch_id, "--checkpoint", "sorted.jsonl"])
    assert lines_of(inbox_file.parent / "sorted.jsonl") == [
        {"id": "e0003", "queue": "other"},
        {"id": "e0001", "queue": "damaged"},
    ]
    capsys.readouterr()
    main(
        [
            "inbox",
            str(inbox_file),
            "--batch",
            "--checkpoint",
            "sorted.jsonl",
        ]
    )
    requests = api.created[-1]["requests"]
    assert [request["custom_id"] for request in requests] == ["e0002"]
    assert capsys.readouterr().out.splitlines()[:2] == [
        "Checkpoint:    2 of 3 already sorted; sending the other 1",
        f"Batch:         {batch_id}",
    ]
