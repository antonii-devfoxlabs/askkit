from askkit.prices import LONG_PROMPT_PRICES, PRICES


def call_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = PRICES[model]
    long_prompt = LONG_PROMPT_PRICES.get(model)
    if long_prompt is not None and input_tokens > long_prompt["over"]:
        prices = long_prompt
    dollars = (
        input_tokens * prices["input"] + output_tokens * prices["output"]
    )
    return dollars / 1_000_000


def error_cost(calls: int, error_rate: float, cost_per_error: float) -> float:
    # What the wrong answers cost: someone's time, a lost sale, a refund.
    return calls * error_rate * cost_per_error
