import asyncio
import csv
import json
import runpy
from collections.abc import Sequence
from datetime import UTC, datetime
from itertools import count
from pathlib import Path
from typing import Any

import httpx2
import pytest
from anthropic import AsyncAnthropic
from test_ask import replied
from test_classify import with_answer

import askkit.cli
from askkit.calllog import day_calls
from askkit.cli import main
from askkit.inbox import (
    JobTimes,
    Rate,
    duration_text,
    pace_text,
    queue_counts,
    rate_text,
    read_inbox,
    run_rate,
    waiting_share,
    whole_job_seconds,
)

SAMPLE = Path(__file__).parent.parent / "samples" / "inbox.csv"
MORNING = datetime(2026, 11, 20, 9, 30, tzinfo=UTC)

# Three emails as a CSV file holds them: the second has a comma and a
# line break of its own, the third quotes the shop's earlier email.
EMAILS = [
    ("e0001", "Order 4417 came with a broken mug. I'd like a new one."),
    ("e0002", "Hi, where is order 4812?\nIt's a gift, so soon, please."),
    (
        "e0003",
        "Thank you, the vase is lovely!\n\n"
        "On 3 Nov 2026, the shop wrote:\n> Your vase is on its way.",
    ),
]
QUEUES = ['{"queue":"damaged"}', '{"queue":"delivery"}', '{"queue":"other"}']


def write_inbox(path: Path, rows: list[tuple[str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "email"])
        writer.writerows(rows)
    return path


@pytest.fixture
def inbox_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    monkeypatch.setattr("askkit.calllog.clock", lambda: MORNING)
    monkeypatch.setattr("askkit.cli.clock", lambda: MORNING)
    return write_inbox(tmp_path / "inbox.csv", EMAILS)


def answering(
    monkeypatch: pytest.MonkeyPatch,
    sent: list[dict[str, Any]],
    answers: Sequence[dict[str, Any] | httpx2.Response],
    waits: Sequence[float] = (),
    in_flight: list[int] | None = None,
) -> None:
    # Each request gets the next answer, in the order the requests come,
    # after the next of waits (none: at once). in_flight, when given,
    # records how many requests were waiting each time one arrived.
    left = iter(answers)
    waits_left = iter(waits)
    waiting = 0

    async def answer(request: httpx2.Request) -> httpx2.Response:
        nonlocal waiting
        sent.append(json.loads(request.content))
        body = next(left)
        waiting += 1
        if in_flight is not None:
            in_flight.append(waiting)
        await asyncio.sleep(next(waits_left, 0.0))
        waiting -= 1
        if isinstance(body, httpx2.Response):
            return body
        return replied(request, body)

    def make(api_key: str) -> AsyncAnthropic:
        transport = httpx2.MockTransport(answer)
        return AsyncAnthropic(
            api_key=api_key,
            http_client=httpx2.AsyncClient(transport=transport),
        )

    monkeypatch.setattr("askkit.cli.AsyncAnthropic", make)


def job_clocks(
    monkeypatch: pytest.MonkeyPatch, wall: float, cpu: float
) -> None:
    # The two clocks of the run: read once at the start and once at the
    # end, so the job took wall seconds, cpu of them AskKit's own work.
    walls = iter([100.0, 100.0 + wall])
    cpus = iter([5.0, 5.0 + cpu])
    monkeypatch.setattr("askkit.inbox.perf_counter", lambda: next(walls))
    monkeypatch.setattr("askkit.inbox.process_time", lambda: next(cpus))


def test_a_comma_and_a_line_break_stay_inside_their_email(
    tmp_path: Path,
) -> None:
    emails = read_inbox(write_inbox(tmp_path / "inbox.csv", EMAILS))
    assert [(email.id, email.email) for email in emails] == EMAILS


def test_a_file_with_other_columns_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "mine.csv"
    path.write_text("name,text\nMaria,Hello\n", encoding="utf-8")
    with pytest.raises(ValueError) as error:
        read_inbox(path)
    assert str(error.value) == (
        f"{path}: the columns must be id,email; this file has name,text"
    )


def test_a_file_with_no_emails_is_refused(tmp_path: Path) -> None:
    path = write_inbox(tmp_path / "empty.csv", [])
    with pytest.raises(ValueError) as error:
        read_inbox(path)
    assert str(error.value) == f"{path} holds no emails"


def test_the_sample_inbox_holds_a_thousand_emails_each_named_once() -> None:
    emails = read_inbox(SAMPLE)
    assert len(emails) == 1000
    assert [email.id for email in emails] == [
        f"e{number:04}" for number in range(1, 1001)
    ]


def test_the_sample_is_the_one_its_script_writes(tmp_path: Path) -> None:
    # The file in the repository and the script that writes it agree, so
    # the script is how the file is changed.
    script = SAMPLE.parent / "make_inbox.py"
    copy = tmp_path / "make_inbox.py"
    copy.write_text(script.read_text(encoding="utf-8"), encoding="utf-8")
    runpy.run_path(str(copy), run_name="__main__")
    assert (tmp_path / "inbox.csv").read_bytes() == SAMPLE.read_bytes()


def test_the_waiting_share_is_the_time_askkit_code_did_not_run() -> None:
    assert waiting_share(JobTimes(wall=20.0, cpu=0.1)) == pytest.approx(0.995)
    assert waiting_share(JobTimes(wall=10.0, cpu=10.0)) == 0.0
    assert waiting_share(JobTimes(wall=0.0, cpu=0.0)) == 0.0


def test_the_whole_file_takes_an_average_call_per_email_one_at_a_time() -> (
    None
):
    assert whole_job_seconds([1.3, 1.5], 1000, 1) == pytest.approx(1400.0)


def test_the_whole_file_takes_an_average_call_per_email_of_each_place() -> (
    None
):
    # Ten places, 1,000 emails: each place answers 100 of them, one
    # after another; 1,001 leaves one place a 101st.
    assert whole_job_seconds([1.3, 1.5], 1000, 10) == pytest.approx(140.0)
    assert whole_job_seconds([1.4], 1001, 10) == pytest.approx(141.4)


def test_the_rate_is_the_places_times_what_one_place_sends_a_minute() -> None:
    # A call of 1.5 s: one place sends 40 requests a minute, ten 400.
    rate = run_rate([1.5, 1.5], [200, 220], 10)
    assert rate.requests == pytest.approx(400.0)
    assert rate.input_tokens == pytest.approx(84_000.0)


def test_the_rate_reads_as_a_rate_limit_is_written() -> None:
    assert rate_text(Rate(416.67, 88_651.2)) == (
        "about 417 requests, 88,700 input tokens a minute"
    )


def test_the_pace_reads_as_a_person_says_it() -> None:
    assert pace_text(1) == "one at a time"
    assert pace_text(10) == "10 at once"


@pytest.mark.parametrize(
    "seconds, text",
    [
        (1.2, "about 1 second"),
        (45.2, "about 45 seconds"),
        (1400.0, "about 23 minutes"),
        (9000.0, "about 2.5 hours"),
    ],
)
def test_a_duration_reads_as_a_person_plans(
    seconds: float, text: str
) -> None:
    assert duration_text(seconds) == text


def test_the_queues_are_counted_in_their_order() -> None:
    queues = ["other", "damaged", "to a person", "damaged", "delivery"]
    assert queue_counts(queues) == (
        "damaged 2, delivery 1, other 1, to a person 1"
    )


def test_at_once_1_sorts_every_email_one_at_a_time(
    inbox_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sent: list[dict[str, Any]] = []
    answering(monkeypatch, sent, [with_answer(queue) for queue in QUEUES])
    job_clocks(monkeypatch, wall=4.23, cpu=0.02)
    main(["inbox", str(inbox_file), "--at-once", "1"])
    assert capsys.readouterr().out.splitlines() == [
        "e0001   damaged",
        "e0002   delivery",
        "e0003   other",
        "",
        "Emails:        3 of 3, one at a time",
        "Queues:        damaged 1, delivery 1, other 1",
        "Model:         claude-haiku-5-5",
        "Tokens:        642 in, 36 out",
        "Cost:          $0.000082",
        "Time:          4.23 s, 1.41 s an email",
        "Waiting:       99.5% of it; AskKit's own work took 0.02 s",
        "Rate:          about 143 requests, 30,600 input tokens a minute",
    ]
    # One request per email, in the order of the file, the quoted
    # earlier email left out, as classify sends one.
    assert [request["messages"][0]["content"] for request in sent] == [
        EMAILS[0][1],
        EMAILS[1][1],
        "Thank you, the vase is lovely!",
    ]


def test_first_tries_a_few_and_tells_the_time_of_the_whole_file(
    inbox_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sent: list[dict[str, Any]] = []
    answering(monkeypatch, sent, [with_answer(QUEUES[0])])
    job_clocks(monkeypatch, wall=1.5, cpu=0.01)
    # The one call took 1.5 s: the file is three such calls.
    call_clock = iter([10.0, 11.5])
    monkeypatch.setattr("askkit.cli.perf_counter", lambda: next(call_clock))
    main(["inbox", str(inbox_file), "--first", "1", "--at-once", "1"])
    lines = capsys.readouterr().out.splitlines()
    assert len(sent) == 1
    assert lines[2] == "Emails:        1 of 3, one at a time"
    assert lines[-1] == "All 3:         about 4 seconds, one at a time"


def test_first_must_be_a_number_of_emails(
    inbox_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []
    answering(monkeypatch, sent, [])
    with pytest.raises(SystemExit) as stopped:
        main(["inbox", str(inbox_file), "--first", "0"])
    assert stopped.value.code == (
        "askkit: --first 0 is not a number of emails; give 1 or more"
    )
    assert sent == []


def test_an_answer_that_fails_its_check_goes_to_a_person_and_the_run_goes_on(
    inbox_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sent: list[dict[str, Any]] = []
    answers = [
        with_answer(QUEUES[0]),
        with_answer('{"queue":"gift cards"}'),
        with_answer(QUEUES[2]),
    ]
    answering(monkeypatch, sent, answers)
    job_clocks(monkeypatch, wall=4.0, cpu=0.02)
    main(["inbox", str(inbox_file)])
    lines = capsys.readouterr().out.splitlines()
    assert len(sent) == 3
    assert lines[1].startswith("e0002   to a person: the queue does not")
    assert lines[2] == "e0003   other"
    assert lines[5] == "Queues:        damaged 1, other 1, to a person 1"


def test_a_rejected_request_names_its_email(
    inbox_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rejected = httpx2.Response(
        400,
        json={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "messages: text content blocks must be non-empty",
            },
        },
    )
    sent: list[dict[str, Any]] = []
    answering(monkeypatch, sent, [with_answer(QUEUES[0]), rejected])
    # One at a time, the run stops at the rejected email: the third is
    # never sent.
    with pytest.raises(SystemExit) as stopped:
        main(["inbox", str(inbox_file), "--at-once", "1"])
    assert str(stopped.value.code).startswith("askkit: e0002: ")
    assert len(sent) == 2


def test_each_email_is_one_line_of_the_call_log(
    inbox_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []
    answering(monkeypatch, sent, [with_answer(queue) for queue in QUEUES])
    main(["inbox", str(inbox_file)])
    folder: Path = vars(askkit.cli)["LOG_FOLDER"]
    calls = day_calls(folder, MORNING.date())
    assert [call.command for call in calls] == ["inbox"] * 3


def test_a_missing_file_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as stopped:
        main(["inbox", "nowhere.csv"])
    assert stopped.value.code == (
        "askkit: cannot read nowhere.csv: No such file or directory"
    )


def test_the_clocks_are_read_only_around_the_emails(
    inbox_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Each clock is read twice, at the start and at the end: a third
    # reading would be a clock started or stopped somewhere else.
    readings = count()

    def reading() -> float:
        return float(next(readings))

    monkeypatch.setattr("askkit.inbox.perf_counter", reading)
    monkeypatch.setattr("askkit.inbox.process_time", reading)
    sent: list[dict[str, Any]] = []
    answering(monkeypatch, sent, [with_answer(queue) for queue in QUEUES])
    main(["inbox", str(inbox_file)])
    assert next(readings) == 4


def test_a_new_email_goes_out_the_moment_one_answers(
    tmp_path: Path,
    inbox_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The mechanism, not a time: with --at-once 2, the third request
    # arrives while the second still waits, as soon as the first has
    # answered; rounds would have sent it alone, after both.
    four = write_inbox(
        tmp_path / "four.csv",
        [*EMAILS, ("e0004", "Is the blue teapot back in stock?")],
    )
    sent: list[dict[str, Any]] = []
    in_flight: list[int] = []
    answering(
        monkeypatch,
        sent,
        [with_answer(queue) for queue in [*QUEUES, QUEUES[2]]],
        waits=[0.05, 0.2, 0.05, 0.05],
        in_flight=in_flight,
    )
    main(["inbox", str(four), "--at-once", "2"])
    assert in_flight == [1, 2, 2, 2]
    assert "Emails:        4 of 4, 2 at once" in capsys.readouterr().out


def test_no_more_calls_wait_than_the_places(
    tmp_path: Path, inbox_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        (f"e{number:04}", f"Where is order {number}?") for number in range(30)
    ]
    many = write_inbox(tmp_path / "many.csv", rows)
    sent: list[dict[str, Any]] = []
    in_flight: list[int] = []
    answering(
        monkeypatch,
        sent,
        [with_answer(QUEUES[1])] * 30,
        waits=[0.01 * (number % 4 + 1) for number in range(30)],
        in_flight=in_flight,
    )
    main(["inbox", str(many), "--at-once", "4"])
    assert max(in_flight) == 4
    # Every email is sent, once.
    contents = sorted(request["messages"][0]["content"] for request in sent)
    assert contents == sorted(email for _, email in rows)


def test_ten_at_once_by_default(
    inbox_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []
    in_flight: list[int] = []
    answering(
        monkeypatch,
        sent,
        [with_answer(queue) for queue in QUEUES],
        waits=[0.05, 0.05, 0.05],
        in_flight=in_flight,
    )
    main(["inbox", str(inbox_file)])
    assert in_flight == [1, 2, 3]


def test_a_line_prints_the_moment_its_answer_comes(
    inbox_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The first email's answer comes last, so its line prints last; the
    # queues are still counted from every answer.
    sent: list[dict[str, Any]] = []
    answering(
        monkeypatch,
        sent,
        [with_answer(queue) for queue in QUEUES],
        waits=[0.15, 0.05, 0.0],
    )
    main(["inbox", str(inbox_file)])
    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == [
        "e0003   other",
        "e0002   delivery",
        "e0001   damaged",
    ]
    assert lines[5] == "Queues:        damaged 1, delivery 1, other 1"


def test_a_rejected_request_sends_no_more_emails(
    inbox_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two places: e0002 is rejected at once while e0001 still waits.
    # e0003 would take e0002's place, but is never sent.
    rejected = httpx2.Response(
        400,
        json={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "messages: text content blocks must be non-empty",
            },
        },
    )
    sent: list[dict[str, Any]] = []
    answering(
        monkeypatch,
        sent,
        [with_answer(QUEUES[0]), rejected, with_answer(QUEUES[2])],
        waits=[0.1, 0.0, 0.0],
    )
    with pytest.raises(SystemExit) as stopped:
        main(["inbox", str(inbox_file), "--at-once", "2"])
    assert str(stopped.value.code).startswith("askkit: e0002: ")
    assert len(sent) == 2


def test_a_rejected_request_among_many_names_its_own_email(
    inbox_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The second email is rejected at once while the first still waits:
    # the run names e0002, the email whose request was rejected.
    rejected = httpx2.Response(
        400,
        json={
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "messages: text content blocks must be non-empty",
            },
        },
    )
    sent: list[dict[str, Any]] = []
    answering(
        monkeypatch,
        sent,
        [with_answer(QUEUES[0]), rejected, with_answer(QUEUES[2])],
        waits=[0.1, 0.0, 0.0],
    )
    with pytest.raises(SystemExit) as stopped:
        main(["inbox", str(inbox_file)])
    assert str(stopped.value.code).startswith("askkit: e0002: ")


def test_at_once_must_be_a_number_of_emails(
    inbox_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []
    answering(monkeypatch, sent, [])
    with pytest.raises(SystemExit) as stopped:
        main(["inbox", str(inbox_file), "--at-once", "0"])
    assert stopped.value.code == (
        "askkit: --at-once 0 is not a number of emails; give 1 or more"
    )
    assert sent == []


def test_max_tokens_a_call_cannot_wait_for_is_refused_before_any_call(
    inbox_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []
    answering(monkeypatch, sent, [])
    with pytest.raises(SystemExit) as stopped:
        main(["inbox", str(inbox_file), "--max-tokens", "30000"])
    assert stopped.value.code == (
        "askkit: --max-tokens 30,000 may take up to 844 s, longer than the"
        " ten minutes a call that is not streamed may take; a queue needs"
        " far fewer"
    )
    assert sent == []


def test_each_call_is_logged_with_its_own_seconds(
    inbox_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The steady clock moves 0.42 s each reading: a call is timed from
    # its own start to its own answer, not over the whole run.
    sent: list[dict[str, Any]] = []
    answering(monkeypatch, sent, [with_answer(queue) for queue in QUEUES])
    main(["inbox", str(inbox_file)])
    folder: Path = vars(askkit.cli)["LOG_FOLDER"]
    calls = day_calls(folder, MORNING.date())
    assert [call.seconds for call in calls] == pytest.approx([1.26] * 3)
