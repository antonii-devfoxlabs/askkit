import json
import logging
from pathlib import Path
from typing import Any

import httpx2
import pytest
from test_classify import use_fake_client, with_answer
from test_inbox import answering, write_inbox
from test_tools import ANSWERED, asked_for, use_answers

import askkit.cli
from askkit.cli import main

# The yes/no bars of the course minimum, each proved at the command a
# person runs, with the API played by a stand-in, so CI holds every one
# on every push and no key is needed. uv run pytest -m release runs them
# alone. The measured bars (the held-out score, the attacks, the bill)
# need real calls: askkit release runs those once.
pytestmark = pytest.mark.release

KEY = "sk-ant-api03-a-test-key-123456789-Ab12"
EMAIL = "Order 4417 came with a broken mug. I'd like a new one."
OVERLOADED = httpx2.Response(
    529,
    json={
        "type": "error",
        "error": {"type": "overloaded_error", "message": "Overloaded"},
    },
)


@pytest.fixture
def one_email(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    return write_inbox(tmp_path / "inbox.csv", [("e0001", EMAIL)])


def test_a_cut_off_answer_is_never_sorted(
    one_email: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A queue name cut off at max_tokens can still read as a queue.
    sent: list[dict[str, Any]] = []
    answering(monkeypatch, sent, [with_answer('{"queue":"dam', "max_tokens")])
    main(["inbox", str(one_email)])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("e0001   to a person: the answer was cut off")
    assert "Queues:        to a person 1" in lines


def test_a_declined_answer_is_never_sorted(
    one_email: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A refused answer may hold a queue the model wrote before it stopped.
    sent: list[dict[str, Any]] = []
    answering(
        monkeypatch, sent, [with_answer('{"queue":"refund"}', "refusal")]
    )
    main(["inbox", str(one_email)])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("e0001   to a person: the model declined")
    assert "Queues:        to a person 1" in lines


def test_a_failing_request_is_tried_three_times_and_no_more(
    one_email: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The SDK is the one owner here: two retries, then the email is left
    # for the next run. No loop of AskKit's own adds to them.
    sent: list[dict[str, Any]] = []
    answering(monkeypatch, sent, [OVERLOADED] * 4)
    main(["inbox", str(one_email)])
    lines = capsys.readouterr().out.splitlines()
    assert len(sent) == 3
    assert lines[0] == "e0001   not sorted: 529 overloaded_error"


def test_a_denied_tool_never_runs_even_when_the_model_asks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never_run(tool_input: dict[str, object]) -> str:
        raise AssertionError("the calculator ran")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("askkit.tools.calculate", never_run)
    sent: list[dict[str, Any]] = []
    asked = asked_for(
        "calculate", {"operation": "multiply", "a": 1284, "b": 0.5}
    )
    use_answers(monkeypatch, sent, [asked, ANSWERED])
    main(["ask", "What is 1,284 x 0.5?", "--deny", "calculate"])
    result = sent[1]["messages"][2]["content"][0]
    assert all(tool["name"] != "calculate" for tool in sent[0]["tools"])
    assert result["is_error"] is True
    assert result["content"].startswith("the user has not allowed calculate")


def test_a_fetch_outside_the_allowed_sites_never_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def never_reached(*arguments: object) -> Any:
        raise AssertionError("the site was reached")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("askkit.fetch.addresses_of", never_reached)
    monkeypatch.setattr("askkit.fetch.open_client", never_reached)
    sent: list[dict[str, Any]] = []
    asked = asked_for("fetch_page", {"url": "https://example.com/notes"})
    use_answers(monkeypatch, sent, [asked, ANSWERED])
    main(["ask", "Read my wiki", "--allow-site", "wiki.example.org"])
    result = sent[1]["messages"][2]["content"][0]
    assert result["is_error"] is True
    assert result["content"].startswith(
        "fetch_page fetches only from the sites the user allowed"
    )


def test_the_key_never_reaches_the_call_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pasted into a question by mistake. Its digits would pass for a
    # phone number, so the key must go before the redaction runs.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    main(["--log-content", "ask", f"Is {KEY} a valid key?"])
    assert KEY in json.dumps(sent[0]["messages"])
    log_folder: Path = vars(askkit.cli)["LOG_FOLDER"]
    logged = ""
    for path in log_folder.glob("*.jsonl"):
        logged += path.read_text(encoding="utf-8")
    assert "Is [KEY] a valid key?" in logged
    assert "123456789" not in logged
    assert KEY not in logged


def test_the_key_never_reaches_the_sdk_debug_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # ANTHROPIC_LOG=debug prints the SDK's requests; a later SDK that
    # printed its headers would print the key.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    headers: list[str] = []
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    make = vars(askkit.cli)["Anthropic"]

    def recording(api_key: str) -> Any:
        client = make(api_key)
        headers.append(client.auth_headers["Authorization"])
        return client

    monkeypatch.setattr("askkit.cli.Anthropic", recording)
    with caplog.at_level(logging.DEBUG):
        main(["ask", "Where is order 4417?"])
    assert headers == [f"Bearer {KEY}"]
    assert "Request options" in caplog.text
    assert KEY not in caplog.text


def test_a_rejected_key_is_never_printed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)

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

    def make(api_key: str) -> Any:
        from anthropic import Anthropic

        transport = httpx2.MockTransport(rejected)
        return Anthropic(
            api_key=api_key, http_client=httpx2.Client(transport=transport)
        )

    monkeypatch.setattr("askkit.cli.Anthropic", make)
    with pytest.raises(SystemExit) as stopped:
        main(["ask", "Where is order 4417?"])
    printed = capsys.readouterr()
    assert KEY not in str(stopped.value.code)
    assert KEY not in printed.out + printed.err
    assert json.dumps(KEY) not in str(stopped.value.code)
