import json
import math
from pathlib import Path

import httpx2
import pytest
from anthropic import Anthropic

from askkit.attacks import load_attacks
from askkit.clean import clean_email
from askkit.cli import main
from askkit.inbox import read_inbox
from askkit.redact import redacted
from askkit.release import RELEASE_FILE, Release, release_faults
from askkit.sorting_set import load_sorting_cases
from askkit.targets import Targets

PYTHON = Path(__file__).parents[1]
SET = PYTHON / "evals" / "classify"
TARGETS = Targets(
    task="Sort each customer email into its team's queue",
    emails_a_month=36000,
    cost_per_error=0.5,
    held_out_right=0.95,
    attacks_won=0,
    month_budget=5.0,
)


def queues_by_email() -> dict[str, tuple[str, str]]:
    # Each email as the sorting call sends it, with its id and the queue
    # its case expects.
    expected: dict[str, tuple[str, str]] = {}
    for case, email in load_sorting_cases(SET / "held_out"):
        expected[redacted(clean_email(email)[0])] = (case.id, case.queue)
    for attack, email in load_attacks(
        PYTHON / "evals" / "attacks" / "cases.json",
        PYTHON / "samples" / "attacks.csv",
    ):
        expected[redacted(clean_email(email)[0])] = (attack.id, attack.queue)
    return expected


def copy_of_the_set(tmp_path: Path) -> Path:
    # The set, the attacks and the targets in a folder of the test's own,
    # laid out as in python/, so release.json is written there.
    for part in ("main", "held_out"):
        target = tmp_path / "evals" / "classify" / part
        target.mkdir(parents=True)
        for name in ("cases.json", "emails.csv"):
            (target / name).write_bytes((SET / part / name).read_bytes())
    (tmp_path / "evals" / "classify" / "targets.json").write_bytes(
        (SET / "targets.json").read_bytes()
    )
    (tmp_path / "evals" / "attacks").mkdir()
    (tmp_path / "evals" / "attacks" / "cases.json").write_bytes(
        (PYTHON / "evals" / "attacks" / "cases.json").read_bytes()
    )
    (tmp_path / "samples").mkdir()
    (tmp_path / "samples" / "attacks.csv").write_bytes(
        (PYTHON / "samples" / "attacks.csv").read_bytes()
    )
    return tmp_path / "evals" / "classify"


def model_answering(
    monkeypatch: pytest.MonkeyPatch,
    sent: list[str],
    wrong: tuple[str, ...] = (),
    input_tokens: int = 280,
) -> None:
    # The right queue for every email but the ids in wrong, each call
    # billed input_tokens in and 12 out.
    expected = queues_by_email()

    def answer(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        email = json.loads(body["messages"][0]["content"])["content"]
        email_id, queue = expected[email]
        sent.append(email_id)
        if email_id in wrong:
            queue = "refund" if queue != "refund" else "other"
        return httpx2.Response(
            200,
            json={
                "id": "msg_01AskKitRelease",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [
                    {"type": "text", "text": json.dumps({"queue": queue})}
                ],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": input_tokens, "output_tokens": 12},
            },
        )

    def make(api_key: str) -> Anthropic:
        transport = httpx2.MockTransport(answer)
        return Anthropic(
            api_key=api_key, http_client=httpx2.Client(transport=transport)
        )

    monkeypatch.setattr("askkit.cli.Anthropic", make)


@pytest.fixture
def release_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    return copy_of_the_set(tmp_path)


def test_a_release_that_meets_every_bar_says_so(
    release_set: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sent: list[str] = []
    model_answering(monkeypatch, sent, wrong=("s38",))
    main(["release", "evals/classify"])
    lines = capsys.readouterr().out.splitlines()
    assert len(sent) == 26
    assert lines == [
        "Model:         claude-haiku-5-5, classify_email v2",
        "Held out:      20 of 21 right (0.95); 20 needed",
        "Misses:        1, not printed: scored, never read",
        "Attacks won:   0 of 5; 0 allowed",
        "Bill:          $0.000034 an email, $1.22 a month; budget $5.00",
        "Month:         $858.37 with the errors, the rules' $8,571.43",
        "Written:       evals/classify/release.json",
        "Release:       every measured bar met",
    ]


def test_the_misses_are_counted_never_named(
    release_set: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sent: list[str] = []
    model_answering(monkeypatch, sent, wrong=("s38",))
    main(["release", "evals/classify"])
    printed = capsys.readouterr().out
    written = (release_set / RELEASE_FILE).read_text(encoding="utf-8")
    assert "s38" not in printed
    assert "s38" not in written


def test_the_held_out_part_is_scored_once(
    release_set: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sent: list[str] = []
    model_answering(monkeypatch, sent)
    main(["release", "evals/classify"])
    with pytest.raises(SystemExit) as stopped:
        main(["release", "evals/classify"])
    assert len(sent) == 26
    assert str(stopped.value.code).startswith(
        "askkit: release.json exists already: the held-out part is scored"
        " once, and was on "
    )


def test_a_missed_bar_ends_with_exit_code_1_and_is_still_written(
    release_set: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sent: list[str] = []
    model_answering(monkeypatch, sent, wrong=("s38", "s45", "a0004"))
    with pytest.raises(SystemExit) as stopped:
        main(["release", "evals/classify"])
    assert stopped.value.code == (
        "askkit: not ready to ship: held out: 19 of 21 right, 20 needed;"
        " attacks: 1 won (a0004), 0 allowed"
    )
    release = Release.model_validate_json(
        (release_set / RELEASE_FILE).read_text(encoding="utf-8")
    )
    assert release.held_out_right == 19
    assert release.attacks_won == ["a0004"]
    assert len(release.faults) == 2


def test_a_bill_over_the_budget_is_a_missed_bar(
    release_set: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 1,500 input tokens a call: $5.62 a month at Haiku 5.5's prices.
    sent: list[str] = []
    model_answering(monkeypatch, sent, input_tokens=1500)
    with pytest.raises(SystemExit) as stopped:
        main(["release", "evals/classify"])
    assert stopped.value.code == (
        "askkit: not ready to ship: bill: $5.62 a month, over the budget"
        " of $5.00"
    )


def test_targets_below_the_minimum_stop_before_any_call(
    release_set: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    targets = json.loads((release_set / "targets.json").read_text())
    targets["held_out_right"] = 0.85
    (release_set / "targets.json").write_text(json.dumps(targets))
    sent: list[str] = []
    model_answering(monkeypatch, sent)
    with pytest.raises(SystemExit) as stopped:
        main(["release", "evals/classify"])
    assert sent == []
    assert not (release_set / RELEASE_FILE).exists()
    assert stopped.value.code == (
        "askkit: the targets are below the course minimum: held_out_right:"
        " 0.85 is below the course minimum of 0.90"
    )


def test_a_rejected_request_writes_nothing(
    release_set: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def rejected(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            401,
            json={
                "type": "error",
                "error": {
                    "type": "authentication_error",
                    "message": "invalid x-api-key",
                },
            },
        )

    def make(api_key: str) -> Anthropic:
        transport = httpx2.MockTransport(rejected)
        return Anthropic(
            api_key=api_key, http_client=httpx2.Client(transport=transport)
        )

    monkeypatch.setattr("askkit.cli.Anthropic", make)
    with pytest.raises(SystemExit) as stopped:
        main(["release", "evals/classify"])
    assert not (release_set / RELEASE_FILE).exists()
    assert str(stopped.value.code).endswith("; nothing was written")


def test_every_fault_is_named_at_once() -> None:
    faults = release_faults(
        TARGETS,
        right=17,
        total=21,
        won=["a0002", "a0004"],
        month_bill=6.0,
        rules_right=11 / 21,
    )
    assert faults == [
        "held out: 17 of 21 right, 20 needed",
        "attacks: 2 won (a0002, a0004), 0 allowed",
        "bill: $6.00 a month, over the budget of $5.00",
    ]


def test_a_month_no_cheaper_than_the_rules_is_a_fault() -> None:
    faults = release_faults(
        TARGETS,
        right=21,
        total=21,
        won=[],
        month_bill=4.0,
        rules_right=1.0,
    )
    assert faults == [
        "month: $4.00 with the errors, no less than the keyword rules' $0.00"
    ]


def test_the_bill_is_the_average_held_out_call_times_the_month() -> None:
    # 280 in and 12 out at $0.10 and $0.50 a million: $0.000034 a call.
    per_email = (280 * 0.10 + 12 * 0.50) / 1_000_000
    assert math.isclose(per_email * 36000, 1.224)


def test_the_demo_inbox_reads_as_an_inbox_each_named_once() -> None:
    emails = read_inbox(PYTHON / "samples" / "demo.csv")
    ids = [email.id for email in emails]
    assert ids == ["d01", "d02", "d03", "d04", "d05"]
