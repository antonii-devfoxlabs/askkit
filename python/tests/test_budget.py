import json
from typing import Any

import httpx2
from anthropic import Anthropic

from askkit.budget import (
    DEFAULT_BUDGET,
    budget_fault,
    input_tokens,
    over_budget,
)
from askkit.limits import LIMITS
from askkit.prices import LONG_PROMPT_PRICES


def counting(sent: list[dict[str, Any]], needed: int) -> Anthropic:
    def answer(request: httpx2.Request) -> httpx2.Response:
        sent.append({"path": request.url.path, **json.loads(request.content)})
        return httpx2.Response(200, json={"input_tokens": needed})

    return Anthropic(
        api_key="sk-ant-api03-a-test-key-Ab12",
        http_client=httpx2.Client(transport=httpx2.MockTransport(answer)),
    )


def test_the_default_budget_stays_under_the_long_prompt_price() -> None:
    assert DEFAULT_BUDGET < LONG_PROMPT_PRICES["claude-haiku-5-5"]["over"]


def test_the_default_budget_fits_the_window_with_the_largest_answer() -> None:
    limits = LIMITS["claude-haiku-5-5"]
    assert (
        budget_fault("claude-haiku-5-5", DEFAULT_BUDGET, limits["max_output"])
        is None
    )


def test_a_budget_and_answer_the_window_holds_exactly_pass() -> None:
    assert budget_fault("claude-haiku-5-5", 999_000, 1_000) is None


def test_a_budget_and_answer_over_the_window_are_refused() -> None:
    assert budget_fault("claude-haiku-5-5", 999_001, 1_000) == (
        "--budget 999,001 and --max-tokens 1,000 come to 1,000,001 tokens,"
        " more than the 1,000,000 of claude-haiku-5-5's window"
    )


def test_a_budget_below_one_is_refused() -> None:
    assert budget_fault("claude-haiku-5-5", 0, 1_000) == (
        "--budget 0 leaves no room for a question"
    )


def test_input_tokens_counts_what_the_call_would_send() -> None:
    sent: list[dict[str, Any]] = []
    history: list[Any] = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    needed = input_tokens(
        counting(sent, 131),
        "claude-haiku-5-5",
        "Be brief.",
        "How are you?",
        history,
    )
    assert needed == 131
    assert sent == [
        {
            "path": "/v1/messages/count_tokens",
            "model": "claude-haiku-5-5",
            "system": "Be brief.",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
            ],
        }
    ]
    # The count leaves the history as it was.
    assert len(history) == 2


def test_a_turn_within_the_budget_passes() -> None:
    assert over_budget(400, 400) is None


def test_a_turn_over_the_budget_says_by_how_much() -> None:
    assert over_budget(1_212, 1_000) == (
        "this turn would send 1,212 tokens, over the chat's budget of 1,000"
    )
