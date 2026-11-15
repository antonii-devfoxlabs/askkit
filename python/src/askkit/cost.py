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
