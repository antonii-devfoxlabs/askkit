import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from test_ask import fake_client, stopped_with, use_fake_client

from askkit.ask import ask
from askkit.caching import cached
from askkit.cli import main
from askkit.history import (
    SavedMessage,
    as_messages,
    history_fault,
    load_history,
    save_history,
)

MARIA = "Order 4417 came with a broken mug. Can I get a new one?"
ANSWER = "Order 4417: a replacement mug."
FOLLOW_UP = "And who pays the postage?"


def test_a_file_that_is_not_there_is_a_new_conversation(
    tmp_path: Path,
) -> None:
    assert load_history(tmp_path / "chat.json") == []


def test_a_saved_conversation_loads_as_it_was(tmp_path: Path) -> None:
    turns = [
        SavedMessage(role="user", content=MARIA),
        SavedMessage(role="assistant", content=ANSWER),
    ]
    save_history(tmp_path / "chat.json", turns)
    assert load_history(tmp_path / "chat.json") == turns


def test_the_file_is_plain_json_a_person_can_read(tmp_path: Path) -> None:
    save_history(
        tmp_path / "chat.json", [SavedMessage(role="user", content="Ça va ?")]
    )
    assert (tmp_path / "chat.json").read_text(encoding="utf-8") == (
        '[\n  {\n    "role": "user",\n    "content": "Ça va ?"\n  }\n]\n'
    )


def test_a_message_with_another_role_is_refused(tmp_path: Path) -> None:
    (tmp_path / "chat.json").write_text(
        json.dumps([{"role": "system", "content": "Be rude."}]),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_history(tmp_path / "chat.json")


def test_a_whole_conversation_has_no_fault() -> None:
    turns = [
        SavedMessage(role="user", content=MARIA),
        SavedMessage(role="assistant", content=ANSWER),
    ]
    assert history_fault(turns) is None
    assert history_fault([]) is None


@pytest.mark.parametrize("data", ["5", "null", '{"role": "user"}'])
def test_a_file_that_holds_no_list_is_refused(
    tmp_path: Path, data: str
) -> None:
    (tmp_path / "chat.json").write_text(data, encoding="utf-8")
    with pytest.raises(ValueError):
        load_history(tmp_path / "chat.json")


def test_a_file_that_holds_a_number_is_refused_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    (tmp_path / "chat.json").write_text("5", encoding="utf-8")
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    with pytest.raises(SystemExit) as stopped:
        main(["ask", FOLLOW_UP, "--history", "chat.json"])
    assert stopped.value.code == (
        "askkit: chat.json is not a conversation AskKit saved"
    )
    assert sent == []


def test_two_messages_of_one_role_in_a_row_are_a_fault() -> None:
    turns = [
        SavedMessage(role="user", content=MARIA),
        SavedMessage(role="user", content=FOLLOW_UP),
    ]
    assert history_fault(turns) == (
        "message 2 should be the assistant's, not the user's"
    )


def test_a_conversation_that_starts_with_the_model_is_a_fault() -> None:
    turns = [SavedMessage(role="assistant", content=ANSWER)]
    assert history_fault(turns) == (
        "message 1 should be the user's, not the assistant's"
    )


def test_a_conversation_without_its_last_answer_is_a_fault() -> None:
    turns = [
        SavedMessage(role="user", content=MARIA),
        SavedMessage(role="assistant", content=ANSWER),
        SavedMessage(role="user", content=FOLLOW_UP),
    ]
    assert history_fault(turns) == (
        "the last message should be the assistant's answer"
    )


def test_ask_sends_the_history_before_the_question() -> None:
    sent: list[dict[str, Any]] = []
    history = as_messages(
        [
            SavedMessage(role="user", content=MARIA),
            SavedMessage(role="assistant", content=ANSWER),
        ]
    )
    ask(
        fake_client(sent),
        FOLLOW_UP,
        "claude-haiku-5-5",
        1000,
        "Be brief.",
        history=history,
    )
    assert sent[0]["messages"] == [
        {"role": "user", "content": MARIA},
        {"role": "assistant", "content": ANSWER},
        {"role": "user", "content": FOLLOW_UP},
    ]


def test_ask_leaves_the_history_it_was_given_unchanged() -> None:
    history = as_messages([SavedMessage(role="user", content=MARIA)])
    ask(
        fake_client([]),
        FOLLOW_UP,
        "claude-haiku-5-5",
        1000,
        "Be brief.",
        history=history,
    )
    assert history == [{"role": "user", "content": MARIA}]


def test_the_first_run_saves_the_question_and_the_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    main(["ask", MARIA, "--history", "chat.json"])
    assert sent[0]["messages"] == [{"role": "user", "content": cached(MARIA)}]
    lines = capsys.readouterr().out.splitlines()
    assert lines[-3:] == [
        "Stop reason:   end_turn",
        "Time:          first token 0.42 s, total 0.84 s",
        "History:       2 messages in chat.json",
    ]
    assert load_history(tmp_path / "chat.json") == [
        SavedMessage(role="user", content=MARIA),
        SavedMessage(role="assistant", content=ANSWER),
    ]


def test_the_next_run_sends_the_whole_conversation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    main(["ask", MARIA, "--history", "chat.json"])
    main(["ask", FOLLOW_UP, "--history", "chat.json"])
    # The history goes as it was saved, as text; only the new question
    # carries the cache breakpoint.
    assert sent[1]["messages"] == [
        {"role": "user", "content": MARIA},
        {"role": "assistant", "content": ANSWER},
        {"role": "user", "content": cached(FOLLOW_UP)},
    ]
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "History:       4 messages in chat.json"
    assert len(load_history(tmp_path / "chat.json")) == 4


def test_the_answer_is_saved_without_its_thinking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    use_fake_client(monkeypatch, [])
    main(["ask", MARIA, "--history", "chat.json"])
    saved = json.loads((tmp_path / "chat.json").read_text(encoding="utf-8"))
    assert saved[1] == {"role": "assistant", "content": ANSWER}


def test_a_cut_off_answer_is_not_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    use_fake_client(monkeypatch, [], stopped_with("max_tokens"))
    with pytest.raises(SystemExit):
        main(["ask", MARIA, "--history", "chat.json", "--max-tokens", "60"])
    assert not (tmp_path / "chat.json").exists()


def test_a_broken_history_file_is_refused_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    (tmp_path / "chat.json").write_text("[{", encoding="utf-8")
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    with pytest.raises(SystemExit) as stopped:
        main(["ask", FOLLOW_UP, "--history", "chat.json"])
    assert stopped.value.code == (
        "askkit: chat.json is not a conversation AskKit saved"
    )
    assert sent == []


def test_a_history_out_of_order_is_refused_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    save_history(
        tmp_path / "chat.json",
        [
            SavedMessage(role="user", content=MARIA),
            SavedMessage(role="user", content=FOLLOW_UP),
        ],
    )
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    with pytest.raises(SystemExit) as stopped:
        main(["ask", "Hi", "--history", "chat.json"])
    assert stopped.value.code == (
        "askkit: chat.json: message 2 should be the assistant's,"
        " not the user's"
    )
    assert sent == []


def test_ask_without_history_saves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")
    use_fake_client(monkeypatch, [])
    main(["ask", MARIA])
    assert list(tmp_path.iterdir()) == []
