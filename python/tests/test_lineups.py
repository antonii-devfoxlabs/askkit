from askkit.cost import call_cost
from askkit.limits import LIMITS
from askkit.prices import LONG_PROMPT_PRICES, PRICES
from askkit.tokens import TOKENIZERS

# Each provider's lineup, from the highest output price down.
LINEUPS = {
    "Anthropic": [
        "claude-fable-5-2",
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
    ],
    "Google": [
        "gemini-4-carbon",
        "gemini-4-argon",
        "gemini-4-hydrogen",
        "gemini-3.5-flash-lite",
    ],
    "OpenAI": [
        "gpt-6-astra",
        "gpt-6.1-sol",
        "gpt-6.1-terra",
        "gpt-6-luna",
    ],
    "Qwen": ["qwen3.8-max", "qwen3.8-27b", "qwen3.8-flash"],
}


def test_the_price_table_holds_every_lineup_in_order() -> None:
    expected = [model for lineup in LINEUPS.values() for model in lineup]
    assert list(PRICES) == expected


def test_each_lineup_runs_from_the_highest_output_price_down() -> None:
    for lineup in LINEUPS.values():
        outputs = [PRICES[model]["output"] for model in lineup]
        assert outputs == sorted(outputs, reverse=True)


def test_gpt_6_prompts_over_272k_cost_more_on_every_model() -> None:
    for model in LINEUPS["OpenAI"]:
        long_prompt = LONG_PROMPT_PRICES[model]
        assert long_prompt["over"] == 272_000
        assert long_prompt["input"] == PRICES[model]["input"] * 2
        assert long_prompt["output"] == PRICES[model]["output"] * 1.5


def test_a_long_prompt_on_gpt_6_astra_is_priced_at_the_long_rate() -> None:
    # 300,000 in at $20, 1,000 out at $75 per million tokens.
    assert call_cost("gpt-6-astra", 300_000, 1_000) == 6.075


def test_every_gpt_6_model_reads_at_most_922k_tokens() -> None:
    for model in LINEUPS["OpenAI"]:
        assert LIMITS[model] == {
            "context_window": 1_050_000,
            "max_input": 922_000,
            "max_output": 128_000,
        }


def test_gemini_4_answers_up_to_65_536_tokens_in_a_1m_window() -> None:
    for model in ["gemini-4-carbon", "gemini-4-argon", "gemini-4-hydrogen"]:
        assert LIMITS[model] == {
            "context_window": 1_048_576,
            "max_input": 1_048_576,
            "max_output": 65_536,
        }


def test_qwen_cloud_models_share_the_published_tokenizer() -> None:
    for model in LINEUPS["Qwen"]:
        assert TOKENIZERS[model] == "Qwen/Qwen3.8-27B"
