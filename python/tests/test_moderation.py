import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx2
import pytest
from openai import OpenAI
from openai.types import Moderation
from test_batch import BatchesAPI, use_batches
from test_classify import use_fake_client, with_answer
from test_inbox import EMAILS, QUEUES, answering, email_in, write_inbox

from askkit.checkpoint import read_checkpoint
from askkit.cli import main
from askkit.config import OPENAI_KEY_NAME
from askkit.inbox import read_inbox
from askkit.moderation import (
    MODERATION_MODEL,
    PER_REQUEST,
    flagged_categories,
    moderate,
    person_reason,
)

SAMPLE = Path(__file__).parent.parent / "samples" / "moderation.csv"

# Every category of the endpoint, under the names its guide writes.
CATEGORIES = [
    "harassment",
    "harassment/threatening",
    "hate",
    "hate/threatening",
    "illicit",
    "illicit/violent",
    "self-harm",
    "self-harm/intent",
    "self-harm/instructions",
    "sexual",
    "sexual/minors",
    "violence",
    "violence/graphic",
]


def result(*flagged: str) -> dict[str, Any]:
    # One text's result, shaped as the endpoint returns it; the scores
    # are made up: 0.9 for a flagged category, 0.01 for the others.
    return {
        "flagged": bool(flagged),
        "categories": {name: name in flagged for name in CATEGORIES},
        "category_scores": {
            name: 0.9 if name in flagged else 0.01 for name in CATEGORIES
        },
        "category_applied_input_types": {
            name: ["text"] for name in CATEGORIES
        },
    }


def moderation_server(
    sent: list[dict[str, Any]],
    flags: dict[str, Sequence[str]] | None = None,
    status: int = 200,
) -> httpx2.MockTransport:
    # Answers each moderation request with one result per text: a text
    # that holds a key of flags gets that key's categories flagged.
    chosen = flags or {}

    def answer(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        sent.append(body)
        if status != 200:
            return httpx2.Response(
                status,
                json={"error": {"message": "Incorrect API key provided"}},
            )
        results = []
        for text in body["input"]:
            categories: Sequence[str] = ()
            for words, named in chosen.items():
                if words in text:
                    categories = named
            results.append(result(*categories))
        return httpx2.Response(
            200,
            json={
                "id": "modr-1",
                "model": "omni-moderation-2024-09-26",
                "results": results,
            },
        )

    return httpx2.MockTransport(answer)


def fake_openai(
    sent: list[dict[str, Any]],
    flags: dict[str, Sequence[str]] | None = None,
    status: int = 200,
) -> OpenAI:
    return OpenAI(
        api_key="sk-proj-a-test-key-Cd34",
        max_retries=0,
        http_client=httpx2.Client(
            transport=moderation_server(sent, flags, status)
        ),
    )


def use_fake_openai(
    monkeypatch: pytest.MonkeyPatch,
    sent: list[dict[str, Any]],
    flags: dict[str, Sequence[str]] | None = None,
    status: int = 200,
) -> None:
    monkeypatch.setenv(OPENAI_KEY_NAME, "sk-proj-a-test-key-Cd34")

    def make(api_key: str) -> OpenAI:
        return fake_openai(sent, flags, status)

    monkeypatch.setattr("askkit.cli.OpenAI", make)


def test_a_flagged_text_names_its_categories_as_the_guide_writes_them() -> (
    None
):
    flagged = Moderation.model_validate(
        result("harassment/threatening", "violence")
    )
    assert flagged_categories(flagged) == [
        "harassment/threatening",
        "violence",
    ]


def test_a_text_not_flagged_has_no_categories() -> None:
    assert flagged_categories(Moderation.model_validate(result())) == []


def test_flagged_with_no_category_still_reads_as_flagged() -> None:
    odd = result()
    odd["flagged"] = True
    assert flagged_categories(Moderation.model_validate(odd)) != []


def test_the_check_sends_the_email_without_its_card_or_phone() -> None:
    sent: list[dict[str, Any]] = []
    moderate(
        fake_openai(sent),
        ["Card 4242 4242 4242 4242, call +44 7700 900123. Order 4417."],
    )
    assert sent[0]["model"] == MODERATION_MODEL
    assert sent[0]["input"] == ["Card [CARD], call [PHONE]. Order 4417."]


def test_each_text_gets_its_own_result_in_order_over_several_requests() -> (
    None
):
    sent: list[dict[str, Any]] = []
    texts = [f"email {number}" for number in range(PER_REQUEST + 5)]
    texts[PER_REQUEST + 2] = "email with a threat"
    flags = moderate(fake_openai(sent, {"threat": ["violence"]}), texts)
    assert [len(body["input"]) for body in sent] == [PER_REQUEST, 5]
    assert len(flags) == len(texts)
    assert flags[PER_REQUEST + 2] == ["violence"]
    assert flags.count([]) == len(texts) - 1


def test_the_person_is_told_what_was_flagged() -> None:
    assert person_reason(["harassment", "violence"]) == (
        "flagged harassment, violence"
    )


def test_the_sample_reads_as_an_inbox_each_named_once() -> None:
    emails = read_inbox(SAMPLE)
    assert [email.id for email in emails] == ["m001", "m002", "m003", "m004"]


@pytest.fixture
def email_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    path = tmp_path / "sam.txt"
    path.write_text(
        "Order 5120 is late. Whoever I find there will get hurt.",
        encoding="utf-8",
    )
    return path


def test_classify_sends_a_flagged_email_to_a_person_not_the_model(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checked: list[dict[str, Any]] = []
    sorted_: list[dict[str, Any]] = []
    use_fake_openai(monkeypatch, checked, {"hurt": ["violence"]})
    use_fake_client(monkeypatch, sorted_, with_answer('{"queue":"other"}'))
    with pytest.raises(SystemExit) as stopped:
        main(["classify", str(email_file), "--moderate"])
    assert stopped.value.code == (
        "askkit: to a person: flagged violence; the email was not sent to"
        " the model"
    )
    assert len(checked) == 1
    assert sorted_ == []


def test_classify_sorts_an_email_the_check_lets_through(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checked: list[dict[str, Any]] = []
    sorted_: list[dict[str, Any]] = []
    use_fake_openai(monkeypatch, checked)
    use_fake_client(monkeypatch, sorted_, with_answer('{"queue":"delivery"}'))
    main(["classify", str(email_file), "--moderate", "--fresh"])
    assert '"queue": "delivery"' in capsys.readouterr().out
    assert len(checked) == 1
    assert len(sorted_) == 1


def test_classify_without_moderate_sends_nothing_to_openai(
    email_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checked: list[dict[str, Any]] = []
    sorted_: list[dict[str, Any]] = []
    use_fake_openai(monkeypatch, checked)
    use_fake_client(monkeypatch, sorted_, with_answer('{"queue":"delivery"}'))
    main(["classify", str(email_file), "--fresh"])
    assert checked == []
    assert len(sorted_) == 1


def test_moderate_without_an_openai_key_stops_before_any_call(
    email_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(OPENAI_KEY_NAME, raising=False)
    sorted_: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sorted_)
    with pytest.raises(SystemExit) as stopped:
        main(["classify", str(email_file), "--moderate"])
    assert stopped.value.code == (
        f"askkit: {OPENAI_KEY_NAME} is not set (--moderate runs at OpenAI);"
        " put your OpenAI key after the = in .env"
    )
    assert sorted_ == []


def test_a_failed_check_sorts_nothing_unchecked(
    email_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checked: list[dict[str, Any]] = []
    sorted_: list[dict[str, Any]] = []
    use_fake_openai(monkeypatch, checked, status=401)
    use_fake_client(monkeypatch, sorted_)
    with pytest.raises(SystemExit) as stopped:
        main(["classify", str(email_file), "--moderate"])
    assert stopped.value.code == (
        "askkit: OpenAI rejected the key; check OPENAI_API_KEY in .env"
    )
    assert sorted_ == []


@pytest.fixture
def inbox_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    return write_inbox(tmp_path / "inbox.csv", EMAILS)


def test_inbox_sorts_only_the_emails_the_check_lets_through(
    inbox_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checked: list[dict[str, Any]] = []
    sorted_: list[dict[str, Any]] = []
    use_fake_openai(monkeypatch, checked, {"gift": ["harassment"]})
    answering(
        monkeypatch, sorted_, [with_answer(QUEUES[0]), with_answer(QUEUES[2])]
    )
    main(["inbox", str(inbox_file), "--moderate", "--at-once", "1"])
    out = capsys.readouterr().out
    assert "e0002   to a person: flagged harassment" in out
    assert len(checked) == 1
    assert len(checked[0]["input"]) == 3
    sent_emails = [email_in(request) for request in sorted_]
    assert sent_emails == [EMAILS[0][1], "Thank you, the vase is lovely!"]
    assert "Emails:        2 of 3, one at a time" in out
    assert "Flagged:       1, to a person: e0002" in out
    assert "Queues:        damaged 1, other 1, to a person 1" in out


def test_a_flagged_email_goes_into_the_checkpoint_to_a_person(
    inbox_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checked: list[dict[str, Any]] = []
    sorted_: list[dict[str, Any]] = []
    use_fake_openai(monkeypatch, checked, {"gift": ["harassment"]})
    answering(
        monkeypatch, sorted_, [with_answer(QUEUES[0]), with_answer(QUEUES[2])]
    )
    checkpoint = inbox_file.parent / "sorted.jsonl"
    main(
        [
            "inbox",
            str(inbox_file),
            "--moderate",
            "--checkpoint",
            str(checkpoint),
        ]
    )
    assert read_checkpoint(checkpoint)["e0002"] == "to a person"
    # A next run sends nothing again: not to the check, not to the model.
    capsys.readouterr()
    main(
        [
            "inbox",
            str(inbox_file),
            "--moderate",
            "--checkpoint",
            str(checkpoint),
        ]
    )
    assert len(checked) == 1
    assert "Emails:        none left to send" in capsys.readouterr().out


def test_an_inbox_all_flagged_makes_no_sorting_call(
    inbox_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checked: list[dict[str, Any]] = []
    sorted_: list[dict[str, Any]] = []
    use_fake_openai(
        monkeypatch,
        checked,
        {"Order": ["violence"], "gift": ["violence"], "vase": ["violence"]},
    )
    answering(monkeypatch, sorted_, [])
    main(["inbox", str(inbox_file), "--moderate"])
    assert sorted_ == []
    out = capsys.readouterr().out
    assert "Flagged:       3, to a person: e0001, e0002, e0003" in out


def test_a_failed_check_stops_the_inbox_before_any_sorting_call(
    inbox_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checked: list[dict[str, Any]] = []
    sorted_: list[dict[str, Any]] = []
    use_fake_openai(monkeypatch, checked, status=401)
    answering(monkeypatch, sorted_, [with_answer(queue) for queue in QUEUES])
    with pytest.raises(SystemExit) as stopped:
        main(["inbox", str(inbox_file), "--moderate"])
    assert stopped.value.code == (
        "askkit: OpenAI rejected the key; check OPENAI_API_KEY in .env"
    )
    assert sorted_ == []


def test_a_batch_holds_only_the_emails_the_check_lets_through(
    inbox_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checked: list[dict[str, Any]] = []
    use_fake_openai(monkeypatch, checked, {"gift": ["harassment"]})
    api = BatchesAPI([], [])
    use_batches(monkeypatch, api)
    main(["inbox", str(inbox_file), "--moderate", "--batch"])
    requests = api.created[0]["requests"]
    assert [request["custom_id"] for request in requests] == [
        "e0001",
        "e0003",
    ]
    out = capsys.readouterr().out
    assert out.startswith("e0002   to a person: flagged harassment\n")
