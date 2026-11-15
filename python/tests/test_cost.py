import pytest

from askkit.cost import call_cost


def test_input_and_output_have_their_own_prices() -> None:
    # 1,500 x $0.10 + 200 x $0.50, per million tokens
    assert call_cost("claude-haiku-5-5", 1_500, 200) == pytest.approx(0.00025)


def test_output_costs_five_times_input() -> None:
    input_only = call_cost("claude-sonnet-5-5", 1_000, 0)
    output_only = call_cost("claude-sonnet-5-5", 0, 1_000)
    assert output_only == pytest.approx(5 * input_only)


def test_long_haiku_prompt_pays_the_higher_price() -> None:
    assert call_cost("claude-haiku-5-5", 100_000, 0) == pytest.approx(0.01)
    assert call_cost("claude-haiku-5-5", 100_001, 0) == pytest.approx(
        0.0500005
    )


def test_long_prompt_price_is_haiku_only() -> None:
    assert call_cost("claude-opus-5-5", 200_000, 0) == pytest.approx(0.8)


def test_unknown_model_is_an_error() -> None:
    with pytest.raises(KeyError):
        call_cost("claude-haiku-5.5", 100, 100)
