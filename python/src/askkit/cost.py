from anthropic.types import Usage

from askkit.prices import (
    CACHE_PRICES,
    LONG_PROMPT_CACHE_PRICES,
    LONG_PROMPT_PRICES,
    PRICES,
)


def call_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = PRICES[model]
    long_prompt = LONG_PROMPT_PRICES.get(model)
    if long_prompt is not None and input_tokens > long_prompt["over"]:
        prices = long_prompt
    dollars = (
        input_tokens * prices["input"] + output_tokens * prices["output"]
    )
    return dollars / 1_000_000


def prompt_tokens(usage: Usage) -> int:
    # With caching on, input_tokens counts only what was neither written
    # to the cache nor read from it, so the whole prompt is all three.
    written = usage.cache_creation_input_tokens or 0
    read = usage.cache_read_input_tokens or 0
    return usage.input_tokens + written + read


def usage_cost(model: str, usage: Usage) -> float:
    # One call, from its response's usage: each kind of input token at
    # its own price. Lesson 1.2's price step goes by the whole prompt,
    # what the cache wrote and read included.
    prices = PRICES[model]
    cache = CACHE_PRICES[model]
    long_prompt = LONG_PROMPT_PRICES.get(model)
    if long_prompt is not None and prompt_tokens(usage) > long_prompt["over"]:
        prices = long_prompt
        cache = LONG_PROMPT_CACHE_PRICES[model]
    dollars = (
        usage.input_tokens * prices["input"]
        + (usage.cache_creation_input_tokens or 0) * cache["write"]
        + (usage.cache_read_input_tokens or 0) * cache["read"]
        + usage.output_tokens * prices["output"]
    )
    return dollars / 1_000_000


def error_cost(calls: int, error_rate: float, cost_per_error: float) -> float:
    # What the wrong answers cost: someone's time, a lost sale, a refund.
    return calls * error_rate * cost_per_error
