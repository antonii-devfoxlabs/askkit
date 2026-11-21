import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx2
from anthropic import Anthropic

from askkit.checks import run_case
from askkit.clean import clean_email
from askkit.golden import Case, email_text, load_cases

PYTHON = Path(__file__).parent.parent
MAIN = PYTHON / "evals" / "extract" / "main"


def reply_case() -> Case:
    for case in load_cases(MAIN):
        if case.email == "reply":
            return case
    raise AssertionError("no reply case in the main part")


def answering(
    sent: list[dict[str, Any]], text: str, stop_reason: str = "end_turn"
) -> Anthropic:
    # The real SDK, with a transport that answers every request with one
    # response shaped like the Messages API's, so no test calls the model.
    def answer(request: httpx2.Request) -> httpx2.Response:
        sent.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "id": "msg_01AskKitEval",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "content": [{"type": "text", "text": text}],
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "usage": {"input_tokens": 236, "output_tokens": 31},
            },
        )

    return Anthropic(
        api_key="sk-ant-api03-a-test-key-Ab12",
        http_client=httpx2.Client(transport=httpx2.MockTransport(answer)),
    )


def ticket(order: int | None, problem: str, wish: str) -> str:
    return json.dumps({"order": order, "problem": problem, "wish": wish})


def test_a_right_answer_has_no_faults() -> None:
    sent: list[dict[str, Any]] = []
    client = answering(
        sent, ticket(4417, "mug in pieces, box torn", "replacement")
    )
    assert (
        run_case(client, MAIN, reply_case(), "claude-haiku-5-5", 1000) == []
    )
    assert len(sent) == 1


def test_the_model_reads_the_email_extract_sends() -> None:
    # The reply quotes an earlier email with order 7302 in it. extract
    # leaves the quoted lines out, so the eval must too, or it scores a
    # program AskKit does not run.
    sent: list[dict[str, Any]] = []
    case = reply_case()
    client = answering(sent, ticket(4417, "mug in pieces", "replacement"))
    run_case(client, MAIN, case, "claude-haiku-5-5", 1000)
    email, left_out = clean_email(email_text(MAIN, case))
    assert left_out > 0
    assert sent[0]["messages"] == [{"role": "user", "content": email}]
    assert "7302" not in sent[0]["messages"][0]["content"]


def test_a_wrong_answer_names_its_faults() -> None:
    client = answering([], ticket(7302, "mug in pieces", "refund"))
    assert run_case(client, MAIN, reply_case(), "claude-haiku-5-5", 1000) == [
        "order: 7302, not 4417",
        "wish: 'refund', not 'replacement'",
    ]


def test_a_cut_off_answer_is_a_fault_and_is_not_parsed() -> None:
    client = answering([], '{"order": 4417, "prob', "max_tokens")
    assert run_case(client, MAIN, reply_case(), "claude-haiku-5-5", 1000) == [
        "stop: the answer was cut off at 31 tokens; give it a larger"
        " --max-tokens"
    ]


def run_pytest(*arguments: str, cwd: Path) -> str:
    # pytest on the eval folder in a fresh process, with no key in its
    # environment and nothing to find a .env in, so the run can never
    # call the model.
    environment = dict(os.environ)
    environment.pop("ANTHROPIC_API_KEY", None)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(PYTHON / "evals"), *arguments],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_the_eval_runs_every_main_case_and_holds_the_rest_back(
    tmp_path: Path,
) -> None:
    output = run_pytest("--collect-only", "-q", cwd=tmp_path)
    assert "16/29 tests collected (13 deselected)" in output
    assert "test_main[maria]" in output
    assert "test_held_out" not in output


def test_the_held_out_part_runs_only_when_asked_for(tmp_path: Path) -> None:
    output = run_pytest(
        "--collect-only", "-q", "-m", "held_out", cwd=tmp_path
    )
    assert "8/29 tests collected (21 deselected)" in output
    assert "test_main" not in output


def test_without_a_key_every_case_is_skipped_not_failed(
    tmp_path: Path,
) -> None:
    output = run_pytest("-rs", cwd=tmp_path)
    assert "16 skipped, 13 deselected" in output
    assert "ANTHROPIC_API_KEY is not set: the eval calls the model" in output
