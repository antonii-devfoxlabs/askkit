import pytest

from askkit.cost import call_cost, error_cost


def test_input_and_output_have_their_own_prices() -> None:
    # 1,500 x $0.10 + 200 x $0.50, per million tokens
    assert call_cost("claude-haiku-5-5", 1_500, 200) == pytest.approx(0.00025)


def test_output_costs_six_times_input_on_qwen() -> None:
    input_only = call_cost("qwen3.8-27b", 1_000, 0)
    output_only = call_cost("qwen3.8-27b", 0, 1_000)
    assert output_only == pytest.approx(6 * input_only)


def test_long_haiku_prompt_pays_the_higher_price() -> None:
    assert call_cost("claude-haiku-5-5", 100_000, 0) == pytest.approx(0.01)
    assert call_cost("claude-haiku-5-5", 100_001, 0) == pytest.approx(
        0.0500005
    )


def test_a_model_without_a_long_prompt_price_keeps_its_own() -> None:
    assert call_cost("qwen3.8-27b", 200_000, 0) == pytest.approx(0.1)


def test_unknown_model_is_an_error() -> None:
    with pytest.raises(KeyError):
        call_cost("claude-haiku-5.5", 100, 100)


def test_error_cost_is_the_wrong_calls_times_the_price_of_one() -> None:
    # 3 in 100 of 36,000 calls wrong, $0.50 each
    assert error_cost(36_000, 0.03, 0.50) == pytest.approx(540.0)


def test_no_errors_cost_nothing() -> None:
    assert error_cost(36_000, 0.0, 0.50) == 0.0
