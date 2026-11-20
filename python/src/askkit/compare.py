from askkit.cost import call_cost
from askkit.limits import LIMITS
from askkit.tokens import count_tokens

# The high-volume model of each of the four providers: the comparison
# AskKit is for. The larger models in prices.py are for routing a request
# that the small one cannot do, not one more row here.
COMPARED: tuple[str, ...] = (
    "claude-haiku-5-5",
    "gemini-3.5-flash-lite",
    "gpt-6-luna",
    "qwen3.8-27b",
)


def compare_models(text: str, output_tokens: int, calls: int) -> list[str]:
    rows: list[str] = []
    for model in COMPARED:
        # Each model counts the text with its own tokenizer.
        input_tokens, exact = count_tokens(model, text)
        limits = LIMITS[model]
        fits = (
            output_tokens <= limits["max_output"]
            and input_tokens <= limits["max_input"]
            and input_tokens + output_tokens <= limits["context_window"]
        )
        if not fits:
            rows.append(f"{model:<22} does not fit")
            continue
        one_call = call_cost(model, input_tokens, output_tokens)
        # ~ marks an estimate.
        tokens = f"{input_tokens:,}" if exact else f"~{input_tokens:,}"
        rows.append(
            f"{model:<22} {tokens:>8} in"
            f"   ${one_call:.6f} a call   ${one_call * calls:,.2f} in all"
        )
    return rows
