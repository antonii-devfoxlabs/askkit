from askkit.compare import COMPARED
from askkit.limits import LIMITS
from askkit.prices import PRICES
from askkit.routing import LARGE_MODEL, SMALL_MODEL


def test_one_comparable_model_from_each_provider() -> None:
    # Anthropic, Google, OpenAI and Qwen, an open-weight family: compare
    # gives none of them more models than another.
    assert list(COMPARED) == [
        "claude-haiku-5-5",
        "gemini-3.5-flash-lite",
        "gpt-6-luna",
        "qwen3.8-27b",
    ]
    assert list(LIMITS) == list(PRICES)


def test_prices_hold_the_compared_models_and_the_routing_pair() -> None:
    # The compared four first, in their order, then the larger Claude
    # model a request is routed to: nothing else.
    assert list(PRICES) == [*COMPARED, LARGE_MODEL]
    assert SMALL_MODEL in COMPARED
