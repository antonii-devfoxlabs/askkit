import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic
from test_checks import RIGHT_PROBLEMS
from test_extract import (
    SHORT,
    TOO_LONG,
    TOO_LONG_FAULT,
    use_fake_answers,
    with_answer,
)
from test_label import printed_json

from askkit.checks import run_routed_case
from askkit.cli import main
from askkit.golden import Case, load_cases
from askkit.label import LABEL_KEY
from askkit.prices import CACHE_PRICES, PRICES
from askkit.routing import (
    LARGE_MODEL,
    SMALL_MODEL,
    call_model,
    escalation_fault,
)
from askkit.runs import add_run, diff_lines, load_results

PYTHON = Path(__file__).parent.parent
MAIN = PYTHON / "evals" / "extract" / "main"
TOASTER_EMAIL = (MAIN / "toaster.txt").read_text(encoding="utf-8")


def ticket(order: int | None, problem: str, wish: str) -> str:
    return json.dumps({"order": order, "problem": problem, "wish": wish})


def main_case(name: str) -> Case:
    for case in load_cases(MAIN):
        if case.email == name:
            return case
    raise AssertionError(f"no {name} case in the main part")


def test_the_large_model_costs_twenty_times_the_small_one() -> None:
    small = PRICES[SMALL_MODEL]
    large = PRICES[LARGE_MODEL]
    assert large["input"] == 20 * small["input"]
    assert large["output"] == 20 * small["output"]
    # Sonnet 5.5 reads its cache at 0.05 times its input price.
    assert CACHE_PRICES[LARGE_MODEL] == {"write": 2.50, "read": 0.10}


def test_the_first_call_goes_to_the_model_asked_for() -> None:
    assert call_model(0, SMALL_MODEL, LARGE_MODEL) == SMALL_MODEL


def test_a_retry_goes_to_the_model_escalated_to() -> None:
    assert call_model(1, SMALL_MODEL, LARGE_MODEL) == LARGE_MODEL


def test_without_an_escalation_a_retry_stays_on_the_same_model() -> None:
    assert call_model(1, SMALL_MODEL, None) == SMALL_MODEL


def test_a_ticket_that_passes_the_checks_is_not_escalated() -> None:
    answer = ticket(5566, "toaster sparks, cord melted", "refund")
    assert escalation_fault(answer, TOASTER_EMAIL) is None


def test_a_problem_over_40_characters_is_escalated() -> None:
    assert escalation_fault(TOO_LONG, "My desk lamp flickers.") == (
        TOO_LONG_FAULT
    )


def test_an_order_number_the_email_does_not_give_is_escalated() -> None:
    answer = ticket(5565, "toaster sparks", "refund")
    assert escalation_fault(answer, TOASTER_EMAIL) == (
        "order: 5565 is not a number in the email"
    )


def test_a_wrong_ticket_that_passes_the_checks_is_not_seen() -> None:
    # Lesson 5.8's toaster: the least serious of three problems. It is a
    # valid ticket for a number the email gives, so no check sends it on:
    # a cascade cannot fix what no code can see.
    answer = ticket(5566, "crumb tray missing", "refund")
    assert escalation_fault(answer, TOASTER_EMAIL) is None


@pytest.fixture
def lamp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    path = tmp_path / "lamp.txt"
    path.write_text("My desk lamp flickers when I switch it on.\n")
    return path


def test_extract_escalates_a_failed_ticket_to_the_large_model(
    lamp_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sent: list[dict[str, Any]] = []
    second = with_answer(SHORT, "end_turn")
    second["model"] = LARGE_MODEL
    second["usage"] = {"input_tokens": 312, "output_tokens": 29}
    use_fake_answers(
        monkeypatch, sent, [with_answer(TOO_LONG, "end_turn"), second]
    )
    main(["extract", str(lamp_file), "--escalate-to", LARGE_MODEL])
    assert [request["model"] for request in sent] == [
        SMALL_MODEL,
        LARGE_MODEL,
    ]
    ticket, lines = printed_json(capsys.readouterr().out)
    # The ticket's label names the model that wrote it, the larger one,
    # not the one --model asked for first.
    assert ticket[LABEL_KEY]["written_by"] == LARGE_MODEL
    # Each call at its own model's prices: 236 in and 31 out on Haiku
    # 5.5, $0.0000391, and 312 in and 29 out on Sonnet 5.5, $0.000914.
    assert lines[1:] == [
        "Model:         claude-haiku-5-5, then claude-sonnet-5-5",
        "Tokens:        548 in, 60 out",
        "Cost:          $0.000953",
        "Stop reason:   end_turn",
        "Time:          first token 0.42 s, total 0.84 s",
        "Time:          first token 0.42 s, total 0.84 s",
        f"Asked again:   {TOO_LONG_FAULT}",
        "Escalated:     to claude-sonnet-5-5",
        "Not stated:    order, wish",
    ]
    # The retry carries the failed answer and its faults, as before.
    assert sent[1]["messages"][1] == {
        "role": "assistant",
        "content": TOO_LONG,
    }


def test_a_ticket_that_passes_never_reaches_the_large_model(
    lamp_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sent: list[dict[str, Any]] = []
    use_fake_answers(monkeypatch, sent, [with_answer(SHORT, "end_turn")])
    main(["extract", str(lamp_file), "--escalate-to", LARGE_MODEL])
    assert len(sent) == 1
    output = capsys.readouterr().out
    assert "Model:         claude-haiku-5-5\n" in output
    assert "Cost:          $0.000039\n" in output
    assert "Escalated" not in output


def test_escalate_to_takes_only_a_claude_model(lamp_file: Path) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["extract", str(lamp_file), "--escalate-to", "gpt-6-luna"])
    assert stopped.value.code == 2


def answering_in_turn(
    sent: list[dict[str, Any]], texts: list[str]
) -> Anthropic:
    # One answer per request, in order, each from the model it was sent
    # to, so the eval's two calls can be told apart.
    def answer(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        sent.append(body)
        return httpx2.Response(
            200,
            json={
                "id": "msg_01AskKitRouted",
                "type": "message",
                "role": "assistant",
                "model": body["model"],
                "content": [{"type": "text", "text": texts[len(sent) - 1]}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 236, "output_tokens": 31},
            },
        )

    return Anthropic(
        api_key="sk-ant-api03-a-test-key-Ab12",
        http_client=httpx2.Client(transport=httpx2.MockTransport(answer)),
    )


def test_the_eval_scores_the_escalated_answer() -> None:
    # clock: three problems that do not fit in 40 characters. The first
    # answer breaks the contract, the larger model's passes the case.
    sent: list[dict[str, Any]] = []
    client = answering_in_turn(
        sent,
        [
            ticket(
                3391, "clock face cracked, hands bent, box crushed", "refund"
            ),
            ticket(3391, "clock face cracked, hands bent", "refund"),
        ],
    )
    faults, escalated = run_routed_case(
        client, MAIN, main_case("clock"), SMALL_MODEL, LARGE_MODEL, 1000
    )
    assert (faults, escalated) == ([], True)
    assert [request["model"] for request in sent] == [
        SMALL_MODEL,
        LARGE_MODEL,
    ]
    assert sent[1]["messages"][2]["role"] == "user"


def test_the_eval_does_not_escalate_a_ticket_that_passes_the_checks() -> None:
    # toaster's wrong ticket passes extract's checks: one call, and the
    # case still fails on its mentions.
    sent: list[dict[str, Any]] = []
    client = answering_in_turn(
        sent, [ticket(5566, "crumb tray missing", "refund")]
    )
    faults, escalated = run_routed_case(
        client, MAIN, main_case("toaster"), SMALL_MODEL, LARGE_MODEL, 1000
    )
    assert escalated is False
    assert len(sent) == 1
    assert faults == [
        "problem: no 'toaster' in 'crumb tray missing'",
        "problem: no 'spark' in 'crumb tray missing'",
    ]


def test_a_routed_run_is_kept_apart_from_a_plain_one(tmp_path: Path) -> None:
    path = tmp_path / "routed.json"
    add_run(path, SMALL_MODEL, 2, {"maria": []}, LARGE_MODEL)
    with pytest.raises(ValueError, match="escalated to claude-sonnet-5-5"):
        add_run(path, SMALL_MODEL, 2, {"maria": []})
    assert load_results(path).escalate_to == LARGE_MODEL


def test_a_plain_run_writes_its_file_as_before(tmp_path: Path) -> None:
    path = tmp_path / "v2.json"
    add_run(path, SMALL_MODEL, 2, {"maria": []})
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert sorted(saved) == ["cases", "model", "prompt_version"]


def test_diff_names_the_escalation_as_the_change(tmp_path: Path) -> None:
    plain = tmp_path / "v2.json"
    routed = tmp_path / "routed.json"
    for _ in range(2):
        add_run(plain, SMALL_MODEL, 2, {"clock": ["too long"], "maria": []})
        add_run(
            routed, SMALL_MODEL, 2, {"clock": [], "maria": []}, LARGE_MODEL
        )
    lines = diff_lines(
        "v2.json", load_results(plain), "routed.json", load_results(routed)
    )
    assert lines[:6] == [
        "              v2.json            routed.json",
        "Model         claude-haiku-5-5   claude-haiku-5-5",
        "Prompt        v2                 v2",
        "Escalate to   none               claude-sonnet-5-5",
        "Changed       escalation",
        "",
    ]
    assert (
        lines[6]
        == "clock         0 of 2             2 of 2             fixed"
    )


def test_the_eval_takes_escalate_to(tmp_path: Path) -> None:
    # Without a key every case is skipped, so this only proves the option
    # exists and the run reaches its summary without a call.
    environment = dict(os.environ)
    environment.pop("ANTHROPIC_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(PYTHON / "evals"),
            "-p",
            "no:cacheprovider",
            "--escalate-to",
            LARGE_MODEL,
            "-q",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "16 skipped, 13 deselected" in result.stdout


def test_a_replayed_routed_run_counts_the_escalated_cases(
    tmp_path: Path,
) -> None:
    # The eval and its settings copied beside a recording of a routed
    # run: every main case passes on the small model's first answer, but
    # clock, whose first answer is too long for the contract, is passed
    # by the larger model. The run ends by naming the share escalated.
    shutil.copytree(PYTHON / "evals", tmp_path / "evals")
    shutil.copytree(PYTHON / "samples", tmp_path / "samples")
    shutil.copy(PYTHON / "pyproject.toml", tmp_path / "pyproject.toml")
    exchanges: list[dict[str, Any]] = []
    for case in load_cases(MAIN):
        first = RIGHT_PROBLEMS[case.email]
        if case.email == "clock":
            first = "clock face cracked, hands bent, box crushed"
        texts = [ticket(case.order, first, case.wish)]
        texts.append(
            ticket(case.order, RIGHT_PROBLEMS[case.email], case.wish)
        )
        sent: list[dict[str, Any]] = []
        client = answering_in_turn(sent, texts)
        run_routed_case(client, MAIN, case, SMALL_MODEL, LARGE_MODEL, 1000)
        for number, request in enumerate(sent):
            exchanges.append(
                {
                    "request": request,
                    "response": {
                        "id": "msg_01AskKitRouted",
                        "type": "message",
                        "role": "assistant",
                        "model": request["model"],
                        "content": [{"type": "text", "text": texts[number]}],
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {"input_tokens": 236, "output_tokens": 31},
                    },
                }
            )
    recording = tmp_path / "evals" / "extract" / "recording.json"
    recording.write_text(json.dumps(exchanges), encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "evals",
            "-p",
            "no:cacheprovider",
            "--replay",
            "--escalate-to",
            LARGE_MODEL,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "Escalated to claude-sonnet-5-5: 1 of 16 cases (clock)"
        in result.stdout
    )
    assert "16 passed, 13 deselected" in result.stdout
