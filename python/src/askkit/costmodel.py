from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from askkit.prices import LONG_PROMPT_PRICES, PRICES


# One feature of the app and what one use of it costs. A use is what a
# person does once, such as classify one email or ask one question; it
# can take several calls, and its tokens are those calls added up, read
# from the usage of real runs. Fees are what a use costs on top of this
# model's tokens: a web search, or a recording written down at another
# provider.
class Feature(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    model: str
    uses: int
    calls: int
    input_tokens: int
    output_tokens: int
    fees: float


class CostModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: list[Feature]


@dataclass
class FeatureCost:
    name: str
    uses: int
    calls: int
    one_use: float
    month: float
    share: float


def feature_fault(feature: Feature) -> str | None:
    if feature.model not in PRICES:
        return f"{feature.name}: AskKit has no prices for {feature.model}"
    if feature.calls < 1:
        return f"{feature.name}: a use makes at least one call"
    numbers = [
        feature.uses,
        feature.input_tokens,
        feature.output_tokens,
        feature.fees,
    ]
    if min(numbers) < 0:
        return f"{feature.name}: uses, tokens and fees cannot be below zero"
    return None


def load_cost_model(path: Path) -> CostModel:
    model = CostModel.model_validate_json(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for feature in model.features:
        fault = feature_fault(feature)
        if fault is not None:
            raise ValueError(f"{path.name}: {fault}")
        if feature.name in names:
            raise ValueError(
                f"{path.name}: two features named {feature.name}"
            )
        names.append(feature.name)
    return model


def use_cost(feature: Feature) -> float:
    # Lesson 1.2's price step goes by the prompt of each call, not by the
    # tokens of a whole use, so the step is checked on an average call.
    # A feature whose calls differ a lot in size is better split into two.
    prices = PRICES[feature.model]
    long_prompt = LONG_PROMPT_PRICES.get(feature.model)
    average_input = feature.input_tokens / feature.calls
    if long_prompt is not None and average_input > long_prompt["over"]:
        prices = long_prompt
    tokens = (
        feature.input_tokens * prices["input"]
        + feature.output_tokens * prices["output"]
    )
    return tokens / 1_000_000 + feature.fees


def month_costs(model: CostModel) -> list[FeatureCost]:
    # The features with what they cost a month, the dearest first.
    rows: list[FeatureCost] = []
    for feature in model.features:
        one_use = use_cost(feature)
        rows.append(
            FeatureCost(
                feature.name,
                feature.uses,
                feature.calls,
                one_use,
                one_use * feature.uses,
                0.0,
            )
        )
    total = sum([row.month for row in rows])
    for row in rows:
        row.share = row.month / total if total > 0 else 0.0
    return sorted(rows, key=lambda row: row.month, reverse=True)


def report_lines(rows: list[FeatureCost]) -> list[str]:
    def line(
        name: str, uses: str, calls: str, one: str, month: str, share: str
    ) -> str:
        return (
            f"{name:<20}{uses:>7}{calls:>7}{one:>12}{month:>10}{share:>8}"
        ).rstrip()

    lines = [line("Feature", "Uses", "Calls", "One use", "A month", "Share")]
    for row in rows:
        lines.append(
            line(
                row.name,
                f"{row.uses:,}",
                f"{row.calls:,}",
                f"${row.one_use:.6f}",
                f"${row.month:,.2f}",
                f"{row.share:.1%}",
            )
        )
    uses = sum([row.uses for row in rows])
    month = sum([row.month for row in rows])
    lines.append(
        line("All features", f"{uses:,}", "", "", f"${month:,.2f}", "")
    )
    return lines
