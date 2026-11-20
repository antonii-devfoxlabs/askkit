from copy import deepcopy
from typing import Any

import pytest
from anthropic.types import Message
from test_tools import use_answers

from askkit.caching import cached
from askkit.cli import main
from askkit.loop import MAX_PAUSES
from askkit.server_tools import (
    MAX_SEARCHES,
    cited_urls,
    searches,
    server_line,
    web_search,
)

QUESTION = "What is new in Python 3.15?"
PAGE = "https://docs.python.org/3/whatsnew/3.15.html"
CITED = (
    "The biggest changes include lazy imports, frozendict and sentinel"
    " builtins, UTF-8 as the default encoding, unpacking in comprehensions,"
)
SEARCH = {
    "type": "server_tool_use",
    "id": "srvtoolu_01AskKitSearch",
    "name": "web_search",
    "input": {"query": "what's new in Python 3.15"},
}
RESULTS = {
    "type": "web_search_tool_result",
    "tool_use_id": "srvtoolu_01AskKitSearch",
    "content": [
        {
            "type": "web_search_result",
            "url": PAGE,
            "title": "What's new in Python 3.15",
            "encrypted_content": "test",
            "page_age": None,
        }
    ],
}
CITATION = {
    "type": "web_search_result_location",
    "url": PAGE,
    "title": "What's new in Python 3.15",
    "encrypted_index": "test",
    "cited_text": CITED,
}


def message(
    content: list[dict[str, Any]],
    stop_reason: str = "end_turn",
    searched: int = 1,
) -> dict[str, Any]:
    # A reply in which the API searched by itself: the request, its
    # results and the cited answer all come back in this one reply.
    return {
        "id": "msg_01AskKitSearched",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 9240,
            "output_tokens": 120,
            "server_tool_use": {
                "web_search_requests": searched,
                "web_fetch_requests": 0,
            },
        },
    }


def searched() -> dict[str, Any]:
    return message(
        [
            {"type": "text", "text": "I'll search the Python docs."},
            SEARCH,
            RESULTS,
            {
                "type": "text",
                "text": (
                    "Python 3.15 adds lazy imports, the frozendict and"
                    " sentinel built-ins, and UTF-8 as the default encoding."
                ),
                "citations": [CITATION],
            },
        ]
    )


def parsed(body: dict[str, Any]) -> Message:
    return Message.model_validate(body)


def test_the_search_is_limited_to_your_sites_and_a_few_searches() -> None:
    assert web_search(("docs.python.org",)) == {
        "type": "web_search_20250305",
        "name": "web_search",
        "max_uses": MAX_SEARCHES,
        "allowed_domains": ["docs.python.org"],
    }
    assert MAX_SEARCHES == 3


def test_without_a_search_site_no_search_is_offered(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [searched()])
    main(["ask", QUESTION])
    names = [tool["name"] for tool in sent[0]["tools"]]
    assert "web_search" not in names


def test_a_search_site_offers_the_search_beside_your_tools(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [searched()])
    main(
        [
            "ask",
            QUESTION,
            "--search-site",
            "docs.python.org",
            "--search-site",
            "peps.python.org",
        ]
    )
    names = [tool["name"] for tool in sent[0]["tools"]]
    assert names == ["current_time", "calculate", "web_search"]
    assert sent[0]["tools"][-1]["allowed_domains"] == [
        "docs.python.org",
        "peps.python.org",
    ]


def test_ask_shows_the_search_its_source_and_what_it_cost(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [searched()])
    main(["ask", QUESTION, "--search-site", "docs.python.org"])
    # One call: the API ran the search inside it, so no second call
    # carries results back.
    assert len(sent) == 1
    assert capsys.readouterr().out.splitlines() == [
        "I'll search the Python docs.",
        "Search:        what's new in Python 3.15",
        "Python 3.15 adds lazy imports, the frozendict and sentinel"
        " built-ins, and UTF-8 as the default encoding.",
        "",
        f"Source:        {PAGE}",
        "Model:         claude-haiku-5-5",
        "Tokens:        9,240 in, 120 out",
        "Searches:      1, $0.01",
        "Cost:          $0.010984",
        "Stop reason:   end_turn",
        "Time:          first token 0.42 s, total 0.84 s",
    ]


def test_a_failed_search_is_a_line_not_an_exception() -> None:
    failed: dict[str, Any] = deepcopy(RESULTS)
    failed["content"] = {
        "type": "web_search_tool_result_error",
        "error_code": "max_uses_exceeded",
    }
    reply = parsed(message([SEARCH, failed], searched=0))
    assert server_line(reply.content[1]) == "Search error:  max_uses_exceeded"
    assert searches([reply]) == 0


def test_only_searches_and_failed_searches_get_a_line() -> None:
    reply = parsed(searched())
    lines = [server_line(block) for block in reply.content]
    assert lines == [
        None,
        "Search:        what's new in Python 3.15",
        None,
        None,
    ]


def test_each_cited_page_is_listed_once() -> None:
    twice = searched()
    twice["content"].append(
        {"type": "text", "text": " Again.", "citations": [CITATION]}
    )
    assert cited_urls([parsed(twice)]) == [PAGE]


def test_searches_add_up_over_every_reply() -> None:
    replies = [parsed(message([], searched=2)), parsed(searched())]
    assert searches(replies) == 3


def paused() -> dict[str, Any]:
    # The API paused its own loop after the search: the turn has begun,
    # and it goes on when the reply is sent back as it came.
    return message(
        [{"type": "text", "text": "I'll search the Python docs."}, SEARCH],
        stop_reason="pause_turn",
    )


def test_a_paused_turn_goes_back_as_it_came(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    rest = message(
        [
            RESULTS,
            {
                "type": "text",
                "text": " Python 3.15 adds lazy imports.",
                "citations": [CITATION],
            },
        ],
        searched=0,
    )
    use_answers(monkeypatch, sent, [paused(), rest])
    main(["ask", QUESTION, "--search-site", "docs.python.org"])
    assert len(sent) == 2
    last = sent[1]["messages"][-1]
    # No new message: the paused turn is the last one, as it came.
    assert last["role"] == "assistant"
    assert last["content"][-1]["id"] == SEARCH["id"]
    assert sent[1]["messages"][-2] == {
        "role": "user",
        "content": cached(QUESTION),
    }
    # The same tools, so the search that waits can still run.
    assert sent[1]["tools"] == sent[0]["tools"]
    out = capsys.readouterr().out.splitlines()
    assert out[:3] == [
        "I'll search the Python docs.",
        "Search:        what's new in Python 3.15",
        " Python 3.15 adds lazy imports.",
    ]
    assert "Tool requests:" not in "\n".join(out)
    assert "Searches:      1, $0.01" in out


def test_a_turn_that_keeps_pausing_stops(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [paused()] * (MAX_PAUSES + 2))
    with pytest.raises(SystemExit) as stopped:
        main(["ask", QUESTION, "--search-site", "docs.python.org"])
    assert len(sent) == MAX_PAUSES + 1
    assert (
        stopped.value.code == "askkit: the answer stopped early: pause_turn"
    )


def test_a_continued_first_call_is_not_forced_again(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [paused(), searched()])
    main(
        [
            "ask",
            QUESTION,
            "--search-site",
            "docs.python.org",
            "--use-tool",
            "current_time",
        ]
    )
    assert sent[0]["tool_choice"] == {"type": "tool", "name": "current_time"}
    assert "tool_choice" not in sent[1]


def test_a_search_waits_while_your_tool_runs(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The model asked for a search and the clock in one turn. The API ran
    # neither: it waits for the clock's result, then runs the search.
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    both = message(
        [
            SEARCH,
            {
                "type": "tool_use",
                "id": "toolu_01AskKitClock",
                "name": "current_time",
                "input": {},
            },
        ],
        stop_reason="tool_use",
        searched=0,
    )
    use_answers(monkeypatch, sent, [both, searched()])
    main(["ask", QUESTION, "--search-site", "docs.python.org"])
    last = sent[1]["messages"][-1]
    # Only your result goes back; the search's result is the API's.
    assert last["role"] == "user"
    assert [block["type"] for block in last["content"]] == ["tool_result"]
    assert last["content"][0]["tool_use_id"] == "toolu_01AskKitClock"
    assert sent[1]["tools"] == sent[0]["tools"]
