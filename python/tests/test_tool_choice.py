from copy import deepcopy
from typing import Any

import pytest
from test_loop import PRODUCT, QUESTION, asked, two_sums
from test_tools import ANSWERED, asked_for, use_answers

from askkit.cli import main
from askkit.loop import MAX_ROUNDS, TOOL_LIMIT, tool_choice_for

# The calculator's request as a forced call gives it: no thinking and no
# text first, only the request.
FORCED = asked_for("calculate", PRODUCT)
FORCED["content"] = FORCED["content"][1:]


def test_no_choice_leaves_it_to_the_model() -> None:
    assert tool_choice_for(0, None) is None
    assert tool_choice_for(3, None) is None


def test_use_tool_forces_only_the_first_call() -> None:
    assert tool_choice_for(0, "calculate") == {
        "type": "tool",
        "name": "calculate",
    }
    assert tool_choice_for(1, "calculate") is None


def test_the_call_after_the_last_round_may_not_ask() -> None:
    assert tool_choice_for(MAX_ROUNDS, None) == {"type": "none"}
    assert tool_choice_for(MAX_ROUNDS, "calculate") == {"type": "none"}
    assert TOOL_LIMIT == (
        "5 rounds ran; the next call may not ask for a tool"
    )


def test_askkit_ask_sends_no_choice_by_default(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, two_sums())
    main(["ask", QUESTION])
    assert len(sent) == 3
    for request in sent:
        assert "tool_choice" not in request


def test_use_tool_forces_the_first_call_and_frees_the_next(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [FORCED, ANSWERED])
    main(["ask", QUESTION, "--use-tool", "calculate"])
    assert sent[0]["tool_choice"] == {"type": "tool", "name": "calculate"}
    assert "tool_choice" not in sent[1]
    printed = capsys.readouterr().out.splitlines()
    assert printed[0] == "Tool:          calculate: 0.175908"
    # The forced call wrote no text, so its first Time: line says so.
    times = [line for line in printed if line.startswith("Time:")]
    assert times[0].startswith("Time:          no text, total")


@pytest.mark.parametrize(
    "options",
    [
        ["--use-tool", "fetch_page"],
        ["--use-tool", "calculate", "--deny", "calculate"],
    ],
    ids=["fetch-without-a-site", "denied"],
)
def test_a_tool_that_is_not_offered_cannot_be_forced(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    options: list[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [ANSWERED])
    with pytest.raises(SystemExit) as stopped:
        main(["ask", QUESTION, *options])
    name = options[1]
    assert stopped.value.code == (
        f"askkit: --use-tool {name} names a tool that is not offered;"
        " take it out of --deny, or, for fetch_page, allow a site with"
        " --allow-site"
    )
    assert sent == []


def test_fetch_page_can_be_forced_once_a_site_is_allowed(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [ANSWERED])
    main(
        [
            "ask",
            "What is this page about?",
            "--allow-site",
            "docs.python.org",
            "--use-tool",
            "fetch_page",
        ]
    )
    assert sent[0]["tool_choice"] == {"type": "tool", "name": "fetch_page"}


def test_the_last_call_answers_without_tools(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    answered = deepcopy(ANSWERED)
    answered["content"][0]["text"] = "I could not finish the sum."
    bodies = [asked(number, PRODUCT, 618) for number in range(1, 6)]
    use_answers(monkeypatch, sent, [*bodies, answered])
    main(["ask", QUESTION])
    assert len(sent) == MAX_ROUNDS + 1
    for request in sent[:MAX_ROUNDS]:
        assert "tool_choice" not in request
    assert sent[MAX_ROUNDS]["tool_choice"] == {"type": "none"}
    # The tools are still offered on the last call: only the choice
    # changes, so the model still reads what each earlier request was.
    assert sent[MAX_ROUNDS]["tools"] == sent[0]["tools"]
    printed = capsys.readouterr().out.splitlines()
    limit = printed.index(f"Tool limit:    {TOOL_LIMIT}")
    assert printed[limit - 1].startswith("Tool:          calculate:")
    assert printed[limit + 1] == "I could not finish the sum."
    assert "Tool requests: 5 in 5 rounds" in printed


def test_no_choice_is_sent_when_no_tool_is_offered(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [ANSWERED])
    main(["ask", QUESTION, "--deny", "calculate", "--deny", "current_time"])
    assert "tools" not in sent[0]
    assert "tool_choice" not in sent[0]
