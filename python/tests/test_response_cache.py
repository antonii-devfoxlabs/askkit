import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from anthropic.types import Message
from test_classify import ANSWER, use_fake_client, with_answer
from test_invoices import FIELDS, INVOICE

from askkit.cli import main
from askkit.response_cache import (
    MAX_AGE,
    keep_answer,
    kept_answer,
    remove_expired,
    request_key,
)

# The moment of the first run, and the cache folder every test uses.
FIRST_RUN = datetime(2026, 11, 20, 9, 14, tzinfo=UTC)

REQUEST: dict[str, Any] = {
    "model": "claude-haiku-5-5",
    "max_tokens": 1000,
    "system": "Sort the email into a queue.",
    "messages": [{"role": "user", "content": "Order 4417 came broken."}],
}


def answer_message() -> Message:
    return Message.model_validate(with_answer('{"queue":"damaged"}'))


@pytest.fixture
def folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    # The CLI's cache folder, and its clock, both set by the test.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    cache = tmp_path / ".askkit-cache"
    monkeypatch.setattr("askkit.cli.CACHE_FOLDER", cache)
    set_clock(monkeypatch, FIRST_RUN)
    return cache


def set_clock(monkeypatch: pytest.MonkeyPatch, moment: datetime) -> None:
    monkeypatch.setattr("askkit.cli.now", lambda: moment)


def test_the_same_request_gets_the_same_key_in_any_field_order() -> None:
    reordered = dict(reversed(list(REQUEST.items())))
    assert request_key(reordered) == request_key(REQUEST)
    assert len(request_key(REQUEST)) == 64


@pytest.mark.parametrize(
    "field, value",
    [
        ("model", "claude-sonnet-5-5"),
        ("max_tokens", 1001),
        ("system", "Sort the email into a queue!"),
        ("messages", [{"role": "user", "content": "Order 4418, broken."}]),
    ],
)
def test_any_field_that_differs_gives_another_key(
    field: str, value: Any
) -> None:
    assert request_key({**REQUEST, field: value}) != request_key(REQUEST)


def test_nothing_is_kept_before_an_answer_is_stored(tmp_path: Path) -> None:
    assert kept_answer(tmp_path, request_key(REQUEST), FIRST_RUN) is None


def test_a_stored_answer_comes_back_whole(tmp_path: Path) -> None:
    key = request_key(REQUEST)
    keep_answer(tmp_path, key, answer_message(), FIRST_RUN)
    kept = kept_answer(tmp_path, key, FIRST_RUN + timedelta(hours=1))
    assert kept is not None
    assert kept.response == answer_message()
    assert kept.stored == FIRST_RUN


def test_an_answer_is_kept_for_30_days_and_not_a_moment_more(
    tmp_path: Path,
) -> None:
    key = request_key(REQUEST)
    keep_answer(tmp_path, key, answer_message(), FIRST_RUN)
    assert MAX_AGE == timedelta(days=30)
    assert kept_answer(tmp_path, key, FIRST_RUN + MAX_AGE) is not None
    later = FIRST_RUN + MAX_AGE + timedelta(seconds=1)
    assert kept_answer(tmp_path, key, later) is None


def test_the_file_holds_the_answer_but_not_the_request(
    tmp_path: Path,
) -> None:
    key = request_key(REQUEST)
    keep_answer(tmp_path, key, answer_message(), FIRST_RUN)
    text = (tmp_path / f"{key}.json").read_text(encoding="utf-8")
    assert "Order 4417" not in text
    assert json.loads(text)["stored"] == "2026-11-20T09:14:00+00:00"


@pytest.mark.parametrize(
    "content", ['{"stored": "2026-11-20T09:14:00+00:00", "resp', "[]", "{}"]
)
def test_a_file_that_cannot_be_read_back_is_not_kept(
    tmp_path: Path, content: str
) -> None:
    # A run stopped while writing, or a file of another shape: the model
    # is asked again rather than the command failing.
    key = request_key(REQUEST)
    (tmp_path / f"{key}.json").write_text(content, encoding="utf-8")
    assert kept_answer(tmp_path, key, FIRST_RUN) is None


def test_the_key_is_made_from_exactly_what_was_sent(
    folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent, with_answer(json.dumps(FIELDS)))
    main(["read", str(INVOICE)])
    [request] = sent
    # The SDK adds stream itself; every other field is the cache's.
    del request["stream"]
    assert [path.stem for path in folder.iterdir()] == [request_key(request)]


def test_read_twice_sends_one_request(
    folder: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent, with_answer(json.dumps(FIELDS)))
    main(["read", str(INVOICE)])
    first = capsys.readouterr().out.splitlines()
    set_clock(monkeypatch, FIRST_RUN + timedelta(minutes=5))
    main(["read", str(INVOICE)])
    second = capsys.readouterr().out.splitlines()
    assert len(sent) == 1
    # The same invoice both times, labelled as written when it was first
    # asked for; only the lines under it differ.
    assert second[:22] == first[:22]
    assert '    "written": "2026-11-20T09:14:00Z",' in second[:22]
    assert second[22:] == [
        "Model:         claude-haiku-5-5",
        "Answer:        kept on 2026-11-20 09:14 UTC; nothing was sent",
        "Cost:          $0.000000, $0.000027 saved",
    ]


def test_a_kept_answer_needs_no_key_and_no_network(
    folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_fake_client(monkeypatch, [], with_answer(json.dumps(FIELDS)))
    main(["read", str(INVOICE)])
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setattr("askkit.cli.load_api_key", lambda: None)

    def no_client(api_key: str) -> None:
        raise AssertionError("a kept answer made a client")

    monkeypatch.setattr("askkit.cli.Anthropic", no_client)
    main(["read", str(INVOICE)])


def test_fresh_asks_again_and_keeps_the_new_answer(
    folder: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent, with_answer(json.dumps(FIELDS)))
    main(["read", str(INVOICE)])
    later = FIRST_RUN + timedelta(days=2)
    set_clock(monkeypatch, later)
    main(["read", str(INVOICE), "--fresh"])
    assert len(sent) == 2
    [path] = folder.iterdir()
    stored = json.loads(path.read_text(encoding="utf-8"))["stored"]
    assert datetime.fromisoformat(stored) == later
    capsys.readouterr()


def test_an_answer_older_than_30_days_is_asked_for_again(
    folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent, with_answer(json.dumps(FIELDS)))
    main(["read", str(INVOICE)])
    set_clock(monkeypatch, FIRST_RUN + timedelta(days=31))
    main(["read", str(INVOICE)])
    assert len(sent) == 2


def test_another_model_is_another_request(
    folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent, ANSWER)
    email = Path("gift.txt")
    email.write_text("Do you sell gift cards?", encoding="utf-8")
    main(["classify", str(email)])
    main(["classify", str(email), "-m", "claude-sonnet-5-5"])
    main(["classify", str(email)])
    assert [request["model"] for request in sent] == [
        "claude-haiku-5-5",
        "claude-sonnet-5-5",
    ]
    assert len(list(folder.iterdir())) == 2


def test_an_invoice_that_does_not_add_up_is_not_kept(
    folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []
    wrong = with_answer(json.dumps({**FIELDS, "total": 819.96}))
    use_fake_client(monkeypatch, sent, wrong)
    for _ in range(2):
        with pytest.raises(SystemExit):
            main(["read", str(INVOICE)])
    assert len(sent) == 2
    assert not folder.exists()


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_an_answer_that_stopped_early_is_not_kept(
    folder: Path, monkeypatch: pytest.MonkeyPatch, stop_reason: str
) -> None:
    sent: list[dict[str, Any]] = []
    use_fake_client(
        monkeypatch, sent, with_answer('{"queue":"oth', stop_reason)
    )
    email = Path("gift.txt")
    email.write_text("Do you sell gift cards?", encoding="utf-8")
    for _ in range(2):
        with pytest.raises(SystemExit):
            main(["classify", str(email)])
    assert len(sent) == 2
    assert not folder.exists()


def test_a_kept_answer_meets_the_checks_of_today(
    folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stored when the checks were looser: the kept answer is checked again
    # and goes to a person, as a new one would.
    use_fake_client(monkeypatch, [], with_answer(json.dumps(FIELDS)))
    main(["read", str(INVOICE)])
    [path] = folder.iterdir()
    data = json.loads(path.read_text(encoding="utf-8"))
    data["response"]["content"][0]["text"] = json.dumps(
        {**FIELDS, "total": 819.96}
    )
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(SystemExit) as stopped:
        main(["read", str(INVOICE)])
    assert "a person checks the invoice" in str(stopped.value)


def test_keeping_an_answer_removes_the_expired_ones(tmp_path: Path) -> None:
    old_key = request_key(REQUEST)
    keep_answer(tmp_path, old_key, answer_message(), FIRST_RUN)
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    later = FIRST_RUN + MAX_AGE + timedelta(days=1)
    new_key = request_key({**REQUEST, "max_tokens": 10})
    keep_answer(tmp_path, new_key, answer_message(), later)
    assert [path.stem for path in tmp_path.iterdir()] == [new_key]


def test_answers_within_their_age_are_left_alone(tmp_path: Path) -> None:
    key = request_key(REQUEST)
    keep_answer(tmp_path, key, answer_message(), FIRST_RUN)
    assert remove_expired(tmp_path, FIRST_RUN + MAX_AGE) == 0
    assert remove_expired(tmp_path, FIRST_RUN + MAX_AGE * 2) == 1
    assert list(tmp_path.iterdir()) == []
