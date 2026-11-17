import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx2
import pytest
from anthropic import Anthropic
from test_ask import WRITTEN, stopped_with

from askkit.budget import DEFAULT_BUDGET
from askkit.chat import NOT_KEPT, NOT_SENT, chat
from askkit.cli import main
from askkit.history import SavedMessage, load_history, save_history

MARIA = "Order 4417 came with a broken mug. Can I get a new one?"
FOLLOW_UP = "And who pays the postage?"
ANSWER = "Order 4417: a replacement mug."
OVERLOADED = {
    "type": "error",
    "error": {"type": "overloaded_error", "message": "Overloaded"},
}


def answering(
    sent: list[dict[str, Any]],
    replies: list[int | dict[str, Any]],
    counted: list[dict[str, Any]] | None = None,
    needed: int | None = 118,
) -> Anthropic:
    # Each call gets the next reply: a response body, or a status code
    # the API answers with an error. A count before a call gets needed
    # tokens, or a 529 when needed is None; it is no call, so it is kept
    # in counted, not in sent.
    def answer(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        if request.url.path == "/v1/messages/count_tokens":
            if counted is not None:
                counted.append(body)
            if needed is None:
                return httpx2.Response(529, json=OVERLOADED)
            return httpx2.Response(200, json={"input_tokens": needed})
        sent.append(body)
        reply = replies[len(sent) - 1]
        if isinstance(reply, int):
            return httpx2.Response(reply, json=OVERLOADED)
        return httpx2.Response(200, json=reply)

    return Anthropic(
        api_key="sk-ant-api03-a-test-key-Ab12",
        http_client=httpx2.Client(transport=httpx2.MockTransport(answer)),
        max_retries=0,
    )


def typed(*lines: str) -> Callable[[str], str]:
    # Stands in for input(): returns the lines one by one, then reports
    # the end of input, as Ctrl-D does.
    left = list(lines)

    def read(prompt: str) -> str:
        if not left:
            raise EOFError
        return left.pop(0)

    return read


def run_chat(
    tmp_path: Path,
    replies: list[int | dict[str, Any]],
    read: Callable[[str], str],
    history: list[SavedMessage] | None = None,
    budget: int = DEFAULT_BUDGET,
    needed: int | None = 118,
    counted: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[SavedMessage]]:
    sent: list[dict[str, Any]] = []
    if history is None:
        history = []
    chat(
        answering(sent, replies, counted, needed),
        tmp_path / "chat.json",
        history,
        "claude-haiku-5-5",
        1000,
        "Be brief.",
        None,
        budget,
        read,
    )
    return sent, history


def test_every_question_carries_the_whole_conversation(
    tmp_path: Path,
) -> None:
    sent, _ = run_chat(tmp_path, [WRITTEN, WRITTEN], typed(MARIA, FOLLOW_UP))
    assert sent[0]["messages"] == [{"role": "user", "content": MARIA}]
    assert sent[1]["messages"] == [
        {"role": "user", "content": MARIA},
        {"role": "assistant", "content": ANSWER},
        {"role": "user", "content": FOLLOW_UP},
    ]


def test_the_file_is_saved_after_every_whole_answer(tmp_path: Path) -> None:
    saved_after: list[int] = []

    def read(prompt: str) -> str:
        path = tmp_path / "chat.json"
        saved_after.append(len(load_history(path)))
        if len(saved_after) == 3:
            return "/exit"
        return MARIA

    run_chat(tmp_path, [WRITTEN, WRITTEN], read)
    assert saved_after == [0, 2, 4]


def test_exit_stops_the_chat_without_a_call(tmp_path: Path) -> None:
    sent, history = run_chat(tmp_path, [], typed("/exit", MARIA))
    assert sent == []
    assert history == []
    assert not (tmp_path / "chat.json").exists()


def test_the_end_of_input_stops_the_chat(tmp_path: Path) -> None:
    sent, history = run_chat(tmp_path, [WRITTEN], typed(MARIA))
    assert len(sent) == 1
    assert len(history) == 2


def test_an_empty_line_asks_nothing(tmp_path: Path) -> None:
    sent, _ = run_chat(tmp_path, [WRITTEN], typed("", "   ", MARIA))
    assert len(sent) == 1


def test_a_failed_call_keeps_nothing_and_the_chat_goes_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sent, history = run_chat(
        tmp_path, [WRITTEN, 529, WRITTEN], typed(MARIA, FOLLOW_UP, FOLLOW_UP)
    )
    # The question asked again goes after the last whole turn, once.
    assert sent[2]["messages"] == sent[1]["messages"]
    assert len(history) == 4
    assert history_roles(tmp_path) == ["user", "assistant"] * 2
    captured = capsys.readouterr()
    assert captured.err.startswith("askkit: Anthropic's servers are")
    assert NOT_KEPT in captured.out


def history_roles(tmp_path: Path) -> list[str]:
    roles: list[str] = []
    for message in load_history(tmp_path / "chat.json"):
        roles.append(message.role)
    return roles


def test_a_cut_off_answer_is_shown_warned_and_not_kept(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, history = run_chat(
        tmp_path, [stopped_with("max_tokens")], typed(MARIA)
    )
    assert history == []
    assert not (tmp_path / "chat.json").exists()
    captured = capsys.readouterr()
    assert captured.out.startswith(ANSWER + "\n" + NOT_KEPT)
    assert captured.err == (
        "askkit: the answer was cut off at 60 tokens;"
        " give it a larger --max-tokens\n"
    )


def test_a_declined_answer_is_not_shown_or_kept(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, history = run_chat(tmp_path, [stopped_with("refusal")], typed(MARIA))
    assert history == []
    captured = capsys.readouterr()
    assert ANSWER not in captured.out
    assert captured.err == (
        "askkit: the model declined to answer this question\n"
    )


def test_each_turn_prints_its_tokens_and_cost(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_chat(
        tmp_path, [WRITTEN, WRITTEN], typed(MARIA, FOLLOW_UP), budget=400
    )
    assert capsys.readouterr().out.splitlines() == [
        ANSWER,
        "Turn 1:        118 in (30% of budget), 27 out, $0.000025",
        "",
        ANSWER,
        "Turn 2:        118 in (30% of budget), 27 out, $0.000025",
        "",
        "",
    ]


def test_a_saved_chat_goes_on_where_it_stopped(tmp_path: Path) -> None:
    earlier = [
        SavedMessage(role="user", content=MARIA),
        SavedMessage(role="assistant", content=ANSWER),
    ]
    sent, history = run_chat(
        tmp_path, [WRITTEN], typed(FOLLOW_UP), list(earlier)
    )
    assert sent[0]["messages"][:2] == [
        {"role": "user", "content": MARIA},
        {"role": "assistant", "content": ANSWER},
    ]
    assert len(history) == 4


def use_answers(
    monkeypatch: pytest.MonkeyPatch,
    sent: list[dict[str, Any]],
    replies: list[int | dict[str, Any]],
    lines: list[str],
) -> None:
    def make(**options: Any) -> Anthropic:
        return answering(sent, replies)

    monkeypatch.setattr("askkit.cli.Anthropic", make)
    monkeypatch.setattr("askkit.cli.input", typed(*lines), raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")


def test_askkit_chat_reads_and_saves_chat_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [WRITTEN], [MARIA, "/exit"])
    main(["chat"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[:4] == [
        "History:       0 messages in chat.json",
        "Budget:        90,000 input tokens a turn",
        "Type /exit to stop.",
        "",
    ]
    assert lines[-1] == "History:       2 messages in chat.json"
    assert len(load_history(tmp_path / "chat.json")) == 2


def test_askkit_chat_takes_another_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    use_answers(monkeypatch, [], [WRITTEN], [MARIA])
    main(["chat", "--history", "work.json"])
    assert len(load_history(tmp_path / "work.json")) == 2
    assert not (tmp_path / "chat.json").exists()


def test_ctrl_c_keeps_every_whole_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    use_answers(monkeypatch, [], [WRITTEN], [])
    asked: list[str] = [MARIA]

    def read(prompt: str) -> str:
        if not asked:
            raise KeyboardInterrupt
        return asked.pop()

    monkeypatch.setattr("askkit.cli.input", read, raising=False)
    main(["chat"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[-1] == "History:       2 messages in chat.json"
    assert len(load_history(tmp_path / "chat.json")) == 2


def test_askkit_chat_refuses_a_broken_file_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    save_history(
        tmp_path / "chat.json",
        [
            SavedMessage(role="user", content=MARIA),
            SavedMessage(role="user", content=FOLLOW_UP),
        ],
    )
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [WRITTEN], [MARIA])
    with pytest.raises(SystemExit) as stopped:
        main(["chat"])
    assert stopped.value.code == (
        "askkit: chat.json: message 2 should be the assistant's,"
        " not the user's"
    )
    assert sent == []


def test_the_count_carries_what_the_call_sends(tmp_path: Path) -> None:
    counted: list[dict[str, Any]] = []
    sent, _ = run_chat(
        tmp_path,
        [WRITTEN, WRITTEN],
        typed(MARIA, FOLLOW_UP),
        counted=counted,
    )
    assert len(counted) == 2
    for count, call in zip(counted, sent, strict=True):
        assert count["model"] == call["model"]
        assert count["system"] == call["system"]
        assert count["messages"] == call["messages"]


def test_a_turn_over_the_budget_is_not_sent(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sent, history = run_chat(
        tmp_path, [WRITTEN], typed(MARIA), budget=100, needed=118
    )
    assert sent == []
    assert history == []
    assert not (tmp_path / "chat.json").exists()
    captured = capsys.readouterr()
    assert captured.err == (
        "askkit: this turn would send 118 tokens,"
        " over the chat's budget of 100\n"
    )
    assert NOT_SENT in captured.out


def test_a_turn_at_the_budget_is_sent(tmp_path: Path) -> None:
    sent, history = run_chat(
        tmp_path, [WRITTEN], typed(MARIA), budget=118, needed=118
    )
    assert len(sent) == 1
    assert len(history) == 2


def test_the_chat_goes_on_after_a_turn_over_the_budget(
    tmp_path: Path,
) -> None:
    # The refused turn leaves nothing behind, so /exit still ends the
    # chat as usual.
    sent, history = run_chat(
        tmp_path, [], typed(MARIA, FOLLOW_UP, "/exit"), budget=100
    )
    assert sent == []
    assert history == []


def test_a_failed_count_keeps_nothing_and_sends_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sent, history = run_chat(tmp_path, [WRITTEN], typed(MARIA), needed=None)
    assert sent == []
    assert history == []
    captured = capsys.readouterr()
    assert captured.err.startswith("askkit: Anthropic's servers are")
    assert NOT_KEPT in captured.out


def test_askkit_chat_takes_a_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [WRITTEN], [MARIA])
    main(["chat", "--budget", "100"])
    assert sent == []
    lines = capsys.readouterr().out.splitlines()
    assert lines[1] == "Budget:        100 input tokens a turn"


def test_askkit_chat_refuses_a_budget_the_window_cannot_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [WRITTEN], [MARIA])
    with pytest.raises(SystemExit) as stopped:
        main(["chat", "--budget", "999500"])
    assert stopped.value.code == (
        "askkit: --budget 999,500 and --max-tokens 1,000 come to 1,000,500"
        " tokens, more than the 1,000,000 of claude-haiku-5-5's window"
    )
    assert sent == []
    assert not (tmp_path / "chat.json").exists()
