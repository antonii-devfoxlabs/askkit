import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from anthropic.types import Usage
from test_ask import WRITTEN
from test_chat import run_chat, typed
from test_documents import POLICY
from test_loop import QUESTION, asked
from test_tools import ANSWERED, use_answers

from askkit.caching import CACHE, cached, cached_system
from askkit.cli import main
from askkit.cost import prompt_tokens, usage_cost
from askkit.documents import documents_block, with_documents
from askkit.prices import CACHE_PRICES, PRICES
from askkit.prompts import render

SUM = {"operation": "multiply", "a": 1284, "b": 0.000137}


def usage(
    plain: int, written: int | None, read: int | None, output: int
) -> Usage:
    return Usage(
        input_tokens=plain,
        cache_creation_input_tokens=written,
        cache_read_input_tokens=read,
        output_tokens=output,
    )


def a_sum(
    first: tuple[int, int, int], second: tuple[int, int, int]
) -> list[dict[str, Any]]:
    # Lesson 8.2's sum: the model asks for the calculator, then answers.
    # Each call's input is (plain, written to the cache, read from it).
    asking = asked(1, SUM, first[0])
    asking["usage"] = {
        "input_tokens": first[0],
        "cache_creation_input_tokens": first[1],
        "cache_read_input_tokens": first[2],
        "output_tokens": 36,
    }
    answer = deepcopy(ANSWERED)
    answer["content"][0]["text"] = "About 18 cents."
    answer["usage"] = {
        "input_tokens": second[0],
        "cache_creation_input_tokens": second[1],
        "cache_read_input_tokens": second[2],
        "output_tokens": 32,
    }
    return [asking, answer]


def run_ask(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    bodies: list[dict[str, Any]],
    question: str = QUESTION,
) -> list[dict[str, Any]]:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, bodies)
    main(["ask", question])
    return sent


def breakpoints(request: dict[str, Any]) -> int:
    # Every cache_control in the request, wherever it stands.
    return json.dumps(request).count('"cache_control"')


def test_each_kind_of_input_token_has_its_own_price() -> None:
    # 100 plain at $0.10, 1,000 written at $0.125, 10,000 read at $0.01
    # and 50 out at $0.50, per million tokens.
    cost = usage_cost("claude-haiku-5-5", usage(100, 1_000, 10_000, 50))
    assert cost == pytest.approx(0.000010 + 0.000125 + 0.000100 + 0.000025)


def test_a_write_costs_more_than_plain_input_and_a_read_a_tenth() -> None:
    plain = usage_cost("claude-haiku-5-5", usage(1_000, 0, 0, 0))
    written = usage_cost("claude-haiku-5-5", usage(0, 1_000, 0, 0))
    read = usage_cost("claude-haiku-5-5", usage(0, 0, 1_000, 0))
    assert written == pytest.approx(1.25 * plain)
    assert read == pytest.approx(0.1 * plain)


def test_a_missing_cache_count_counts_as_none() -> None:
    assert prompt_tokens(usage(118, None, None, 27)) == 118
    assert usage_cost("claude-haiku-5-5", usage(118, None, None, 27)) == (
        pytest.approx(0.0000253)
    )


def test_the_price_step_counts_what_the_cache_read() -> None:
    # 10 plain tokens, but 100,000 read: the prompt is 100,010 tokens, so
    # the whole call is billed at Claude Haiku 5.5's higher prices.
    cost = usage_cost("claude-haiku-5-5", usage(10, 0, 100_000, 0))
    assert prompt_tokens(usage(10, 0, 100_000, 0)) == 100_010
    assert cost == pytest.approx((10 * 0.50 + 100_000 * 0.05) / 1_000_000)


def test_every_claude_model_has_cache_prices() -> None:
    for model in PRICES:
        if model.startswith("claude-"):
            assert model in CACHE_PRICES


def test_ask_prints_what_the_cache_read_and_wrote(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The first call writes its 628 tokens; the second reads them back
    # and sends only the tool's request and result in full.
    run_ask(monkeypatch, tmp_path, a_sum((0, 628, 0), (51, 0, 628)))
    lines = capsys.readouterr().out.splitlines()
    assert lines[4:8] == [
        "Tokens:        1,307 in, 68 out",
        "Cache:         628 read, 628 written",
        "Cost:          $0.000124",
        "Stop reason:   end_turn",
    ]


def test_a_command_with_nothing_cached_prints_no_cache_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_ask(monkeypatch, tmp_path, a_sum((628, 0, 0), (679, 0, 0)))
    lines = capsys.readouterr().out.splitlines()
    assert lines[4:6] == [
        "Tokens:        1,307 in, 68 out",
        "Cost:          $0.000165",
    ]


def test_the_price_step_goes_by_each_calls_prompt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Two calls of 60,000 tokens: 120,000 in all, but no call is over
    # Haiku 5.5's 100,000, so neither pays the higher prices. AskKit
    # used to add the calls up first and price the total at the step.
    run_ask(monkeypatch, tmp_path, a_sum((60_000, 0, 0), (60_000, 0, 0)))
    lines = capsys.readouterr().out.splitlines()
    assert lines[4:6] == [
        "Tokens:        120,000 in, 68 out",
        "Cost:          $0.012034",
    ]


def test_ask_marks_the_system_prompt_and_the_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent = run_ask(monkeypatch, tmp_path, a_sum((0, 628, 0), (51, 0, 628)))
    first, second = sent
    assert first["system"][0]["cache_control"] == CACHE
    assert first["messages"] == [
        {"role": "user", "content": cached(QUESTION)}
    ]
    # Two breakpoints in each call: the system prompt, which keeps the
    # tools before it, and the question. The tool's result has none.
    assert breakpoints(first) == 2
    assert breakpoints(second) == 2
    assert "cache_control" not in json.dumps(second["messages"][2])


def test_the_second_call_starts_with_the_whole_first_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A cache hit needs the same bytes up to the breakpoint, so the second
    # call must repeat the first call's tools, system prompt and messages
    # exactly, and only add to them.
    first, second = run_ask(
        monkeypatch, tmp_path, a_sum((0, 628, 0), (51, 0, 628))
    )
    assert second["tools"] == first["tools"]
    assert second["system"] == first["system"]
    count = len(first["messages"])
    assert second["messages"][:count] == first["messages"]


def test_two_questions_share_the_tools_and_the_system_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two runs a minute apart: everything before the question is the
    # same, byte for byte, so the second run reads it from the cache.
    one = run_ask(monkeypatch, tmp_path, [WRITTEN], "Where is my order?")
    two = run_ask(monkeypatch, tmp_path, [WRITTEN], "Who pays postage?")
    for part in ("model", "tools", "system"):
        assert json.dumps(one[0][part]) == json.dumps(two[0][part])


def test_the_system_breakpoint_keeps_the_prompt_as_it_was() -> None:
    assert cached_system("Be brief.") == [
        {"type": "text", "text": "Be brief.", "cache_control": CACHE}
    ]


def test_cached_marks_only_the_last_block() -> None:
    blocks = cached(
        [
            {"type": "text", "text": "Image 1:"},
            {"type": "text", "text": "What changed?"},
        ]
    )
    assert blocks == [
        {"type": "text", "text": "Image 1:"},
        {"type": "text", "text": "What changed?", "cache_control": CACHE},
    ]


def test_documents_end_in_a_breakpoint_before_the_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "returns-policy.txt").write_text(POLICY, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [WRITTEN])
    # The answer quotes nothing, so AskKit stops after the call; the
    # request is what this test reads.
    with pytest.raises(SystemExit):
        main(["ask", "Who pays?", "--document", "returns-policy.txt"])
    documents, question = sent[0]["messages"][0]["content"]
    assert documents["cache_control"] == CACHE
    assert question["cache_control"] == CACHE
    assert "Who pays?" not in documents["text"]
    # The system prompt, the documents and the question: three of the
    # four breakpoints a request may have.
    assert breakpoints(sent[0]) == 3


def test_the_documents_request_keeps_the_words_of_v2() -> None:
    # The eval sends with_documents as one text: splitting it into two
    # blocks changed no word, so its scores still compare with v2's.
    documents = {"returns-policy.txt": POLICY}
    values = {"documents": documents_block(documents), "question": "Who?"}
    old = render("quote_request", values, 2)
    assert with_documents("Who?", documents) == old


def test_a_chat_asks_the_api_to_move_a_breakpoint_along(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reply = deepcopy(WRITTEN)
    reply["usage"] = {
        "input_tokens": 40,
        "cache_creation_input_tokens": 160,
        "cache_read_input_tokens": 600,
        "output_tokens": 27,
    }
    sent, _ = run_chat(tmp_path, [reply], typed("Where is my order?"))
    assert sent[0]["cache_control"] == CACHE
    lines = capsys.readouterr().out.splitlines()
    assert lines[-5:-3] == [
        "Turn 1:        800 in (1% of budget), 27 out, $0.000044",
        "Cache:         600 read, 160 written",
    ]
