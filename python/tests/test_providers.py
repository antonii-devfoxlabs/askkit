from askkit.limits import LIMITS
from askkit.prices import PRICES


def test_one_comparable_model_from_each_provider() -> None:
    # Anthropic, Google, OpenAI and Qwen, an open-weight family: none of
    # them gets more models than another.
    assert list(PRICES) == [
        "claude-haiku-5-5",
        "gemini-3.5-flash-lite",
        "gpt-6-luna",
        "qwen3.8-27b",
    ]
    assert list(LIMITS) == list(PRICES)
