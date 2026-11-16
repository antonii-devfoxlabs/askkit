from askkit.cost import call_cost
from askkit.limits import LIMITS
from askkit.prices import PRICES


def compare_models(
    input_tokens: int, output_tokens: int, calls: int
) -> list[str]:
    rows: list[str] = []
    for model in PRICES:
        limits = LIMITS[model]
        fits = (
            output_tokens <= limits["max_output"]
            and input_tokens + output_tokens <= limits["context_window"]
        )
        if not fits:
            rows.append(f"{model:<18} does not fit")
            continue
        one_call = call_cost(model, input_tokens, output_tokens)
        rows.append(
            f"{model:<18} ${one_call:.6f} a call"
            f"   ${one_call * calls:,.2f} in all"
        )
    return rows
