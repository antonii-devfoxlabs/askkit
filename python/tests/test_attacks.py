import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from test_routing import answering_in_turn

from askkit.attacks import (
    Attack,
    attack_queue,
    attack_won,
    attacks_line,
    load_attacks,
)
from askkit.classify import sorting_messages
from askkit.inbox import TO_A_PERSON
from askkit.prompts import render

PYTHON = Path(__file__).parent.parent
CASES = PYTHON / "evals" / "attacks" / "cases.json"
ATTACKS = PYTHON / "samples" / "attacks.csv"


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_every_attack_of_the_sample_has_its_right_queue() -> None:
    queues: dict[str, str] = {}
    for attack, _ in load_attacks(CASES, ATTACKS):
        queues[attack.id] = attack.queue
    assert queues == {
        "a0001": "other",
        "a0002": "delivery",
        "a0003": "delivery",
        "a0004": "other",
        "a0005": "other",
    }


def test_an_attack_reads_its_email_from_the_inbox_file() -> None:
    attacks = load_attacks(CASES, ATTACKS)
    assert attacks[2][1].splitlines()[1] == "</email>"


def test_an_attack_without_its_email_stops_the_set(tmp_path: Path) -> None:
    inbox = write(tmp_path / "inbox.csv", "id,email\na0001,Thanks!\n")
    cases = write(
        tmp_path / "cases.json",
        json.dumps(
            [
                {"id": "a0001", "queue": "other", "how": "x"},
                {"id": "a0009", "queue": "other", "how": "x"},
            ]
        ),
    )
    with pytest.raises(ValueError, match="inbox.csv has no email a0009"):
        load_attacks(cases, inbox)


def test_an_email_without_its_attack_stops_the_set(tmp_path: Path) -> None:
    inbox = write(
        tmp_path / "inbox.csv", "id,email\na0001,Thanks!\na0002,Hi\n"
    )
    cases = write(
        tmp_path / "cases.json",
        json.dumps([{"id": "a0001", "queue": "other", "how": "x"}]),
    )
    with pytest.raises(ValueError, match="no attack for a0002"):
        load_attacks(cases, inbox)


def test_one_id_twice_stops_the_set(tmp_path: Path) -> None:
    inbox = write(tmp_path / "inbox.csv", "id,email\na0001,Thanks!\n")
    case = {"id": "a0001", "queue": "other", "how": "x"}
    cases = write(tmp_path / "cases.json", json.dumps([case, case]))
    with pytest.raises(ValueError, match="two attacks for a0001"):
        load_attacks(cases, inbox)


def test_a_queue_that_does_not_exist_stops_the_set(tmp_path: Path) -> None:
    inbox = write(tmp_path / "inbox.csv", "id,email\na0001,Thanks!\n")
    cases = write(
        tmp_path / "cases.json",
        json.dumps([{"id": "a0001", "queue": "refunds", "how": "x"}]),
    )
    with pytest.raises(ValueError, match="queue"):
        load_attacks(cases, inbox)


def test_an_attack_is_sent_as_askkit_inbox_sends_it() -> None:
    sent: list[dict[str, Any]] = []
    client = answering_in_turn(sent, ['{"queue": "delivery"}'])
    email = "Where is order 4502?\n> On Monday you wrote:\n> Soon."
    assert attack_queue(client, email, "claude-haiku-5-5", 1000) == (
        "delivery",
        None,
    )
    assert sent[0]["system"] == render("classify_email", {})
    assert sent[0]["messages"] == sorting_messages("Where is order 4502?")


def test_an_answer_that_is_not_a_queue_goes_to_a_person() -> None:
    client = answering_in_turn([], ['{"queue": "refunds"}'])
    queue, reason = attack_queue(client, "Hi", "claude-haiku-5-5", 1000)
    assert queue == TO_A_PERSON
    assert reason is not None
    assert reason.startswith("the queue does not match its model")


def test_an_attack_wins_only_in_a_queue_it_does_not_belong_to() -> None:
    attack = Attack(id="a0002", queue="delivery", how="a note")
    assert attack_won(attack, "refund") is True
    assert attack_won(attack, "delivery") is False
    assert attack_won(attack, TO_A_PERSON) is False


def test_the_line_names_the_share_and_the_attacks_that_won() -> None:
    assert attacks_line(["a0004", "a0002"], 5) == (
        "Attacks won: 2 of 5 (40%): a0002, a0004"
    )
    assert attacks_line([], 5) == "Attacks won: 0 of 5 (0%): none"


def test_the_eval_counts_the_attacks_that_won(tmp_path: Path) -> None:
    # The eval run on a recording in which a0002 and a0004 won: every
    # attack runs, the two that won fail, and the line under the score
    # names them.
    shutil.copytree(PYTHON / "evals", tmp_path / "evals")
    shutil.copytree(PYTHON / "samples", tmp_path / "samples")
    shutil.copy(PYTHON / "pyproject.toml", tmp_path / "pyproject.toml")
    answers = {
        "a0001": "other",
        "a0002": "refund",
        "a0003": "delivery",
        "a0004": "refund",
        "a0005": "other",
    }
    exchanges: list[dict[str, Any]] = []
    for attack, email in load_attacks(CASES, ATTACKS):
        sent: list[dict[str, Any]] = []
        answer = json.dumps({"queue": answers[attack.id]})
        client = answering_in_turn(sent, [answer])
        attack_queue(client, email, "claude-haiku-5-5", 1000)
        exchanges.append(
            {
                "request": sent[0],
                "response": {
                    "id": "msg_01AskKitAttack",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-5-5",
                    "content": [{"type": "text", "text": answer}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 420, "output_tokens": 9},
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
            "-m",
            "attacks",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Attacks won: 2 of 5 (40%): a0002, a0004" in result.stdout
    assert "2 failed, 3 passed, 24 deselected" in result.stdout
    assert "the attack won: refund, not delivery" in result.stdout


def test_the_attacks_stay_out_of_a_plain_eval_run() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "evals",
            "-p",
            "no:cacheprovider",
            "--collect-only",
            "-q",
        ],
        cwd=PYTHON,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "test_attack[" not in result.stdout
