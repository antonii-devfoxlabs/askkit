import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic, InternalServerError

from askkit.ask import answer_text
from askkit.checks import run_case
from askkit.extract import extract_ticket
from askkit.golden import load_cases
from askkit.recordings import (
    NotRecorded,
    load_exchanges,
    recording,
    replaying,
)

PYTHON = Path(__file__).parent.parent
MAIN = PYTHON / "evals" / "extract" / "main"
KEY = "sk-ant-api03-a-test-key-Ab12"
EMAIL = "Hello, order 4417 came with a broken mug. A new one, please. Maria"
TICKET = '{"order": 4417, "problem": "mug broken", "wish": "replacement"}'


def body(text: str) -> dict[str, Any]:
    return {
        "id": "msg_01AskKitRecording",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 236, "output_tokens": 31},
    }


def upstream(texts: list[str], status: int = 200) -> httpx2.MockTransport:
    # Plays the API for the recorder: one answer per request, in order.
    calls: list[int] = []

    def answer(request: httpx2.Request) -> httpx2.Response:
        calls.append(1)
        if status != 200:
            return httpx2.Response(
                status,
                json={
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "x"},
                },
            )
        return httpx2.Response(200, json=body(texts[len(calls) - 1]))

    return httpx2.MockTransport(answer)


def client_on(transport: httpx2.MockTransport) -> Anthropic:
    return Anthropic(
        api_key=KEY,
        max_retries=0,
        http_client=httpx2.Client(transport=transport),
    )


def test_a_recording_keeps_the_request_and_the_answer(
    tmp_path: Path,
) -> None:
    path = tmp_path / "recording.json"
    client = client_on(recording(path, upstream([TICKET])))
    extract_ticket(client, EMAIL, "claude-haiku-5-5", 1000)
    exchanges = load_exchanges(path)
    assert len(exchanges) == 1
    assert exchanges[0]["request"]["messages"] == [
        {"role": "user", "content": EMAIL}
    ]
    assert exchanges[0]["response"] == body(TICKET)


def test_a_recording_never_holds_the_key(tmp_path: Path) -> None:
    # The recording goes into Git; the key is in the request's headers.
    path = tmp_path / "recording.json"
    client = client_on(recording(path, upstream([TICKET])))
    extract_ticket(client, EMAIL, "claude-haiku-5-5", 1000)
    assert KEY not in path.read_text(encoding="utf-8")


def test_recording_again_replaces_the_old_answer(tmp_path: Path) -> None:
    path = tmp_path / "recording.json"
    newer = TICKET.replace("mug broken", "mug in pieces")
    client = client_on(recording(path, upstream([TICKET, newer])))
    extract_ticket(client, EMAIL, "claude-haiku-5-5", 1000)
    extract_ticket(client, EMAIL, "claude-haiku-5-5", 1000)
    exchanges = load_exchanges(path)
    assert len(exchanges) == 1
    assert exchanges[0]["response"] == body(newer)


def test_an_error_is_not_recorded(tmp_path: Path) -> None:
    path = tmp_path / "recording.json"
    client = client_on(recording(path, upstream([], status=529)))
    with pytest.raises(InternalServerError):
        extract_ticket(client, EMAIL, "claude-haiku-5-5", 1000)
    assert not path.exists()


def test_a_replay_gives_the_recorded_answer(tmp_path: Path) -> None:
    path = tmp_path / "recording.json"
    recorder = client_on(recording(path, upstream([TICKET])))
    extract_ticket(recorder, EMAIL, "claude-haiku-5-5", 1000)
    player = client_on(replaying(path))
    response = extract_ticket(player, EMAIL, "claude-haiku-5-5", 1000)
    assert answer_text(response) == TICKET


def test_a_replay_refuses_a_request_it_never_recorded(
    tmp_path: Path,
) -> None:
    # Another model (or prompt, or email) is another request: an old
    # answer to it would test the new code against an answer to the old.
    path = tmp_path / "recording.json"
    recorder = client_on(recording(path, upstream([TICKET])))
    extract_ticket(recorder, EMAIL, "claude-haiku-5-5", 1000)
    player = client_on(replaying(path))
    with pytest.raises(NotRecorded, match="no answer to this request"):
        extract_ticket(player, EMAIL, "claude-sonnet-5-5", 1000)


def run_eval(folder: Path, *arguments: str) -> str:
    # pytest on a copy of the eval folder in a fresh process, with no key
    # in its environment, so the run can never call the model.
    environment = dict(os.environ)
    environment.pop("ANTHROPIC_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(PYTHON / "pyproject.toml"),
            "-p",
            "no:cacheprovider",
            str(folder),
            *arguments,
        ],
        cwd=folder.parent,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout + result.stderr


def eval_copy(tmp_path: Path) -> Path:
    folder = tmp_path / "evals"
    shutil.copytree(PYTHON / "evals", folder)
    # The attacks' emails are the inbox sample's, read from beside it.
    shutil.copytree(PYTHON / "samples", tmp_path / "samples")
    (folder / "extract" / "recording.json").unlink(missing_ok=True)
    return folder


def test_replay_without_a_recording_skips_every_case(tmp_path: Path) -> None:
    output = run_eval(eval_copy(tmp_path), "--replay", "-rs")
    assert "16 skipped, 13 deselected" in output
    assert "no recording.json yet: run once with --record" in output


def test_the_eval_replays_a_recorded_run_with_no_key(
    tmp_path: Path,
) -> None:
    # Record the 16 main cases from a stand-in for the API, then replay
    # them in pytest with no key: the eval runs end to end for free.
    folder = eval_copy(tmp_path)
    path = folder / "extract" / "recording.json"
    main = folder / "extract" / "main"
    cases = load_cases(main)
    # A right ticket for each case: the first word each pattern allows.
    texts = []
    for case in cases:
        words = [pattern.split("|")[0] for pattern in case.mentions]
        ticket = {"order": case.order, "problem": " ".join(words)}
        texts.append(json.dumps(ticket | {"wish": case.wish}))
    recorder = client_on(recording(path, upstream(texts)))
    for case in cases:
        assert run_case(recorder, main, case, "claude-haiku-5-5", 1000) == []
    output = run_eval(folder, "--replay")
    assert "16 passed, 13 deselected" in output
