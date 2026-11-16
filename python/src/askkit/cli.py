import argparse
import sys
from pathlib import Path

from askkit.compare import compare_models
from askkit.cost import call_cost, error_cost
from askkit.limits import LIMITS
from askkit.prices import PRICES
from askkit.tokens import count_tokens


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="askkit")
    commands = parser.add_subparsers(dest="command", required=True)
    estimate = commands.add_parser(
        "estimate", help="estimate what sending a file would cost"
    )
    estimate.add_argument("file", type=Path)
    estimate.add_argument(
        "-m", "--model", choices=sorted(PRICES), default="claude-haiku-5-5"
    )
    estimate.add_argument("--output", type=int, default=500)
    estimate.add_argument("--calls", type=int, default=1)
    estimate.add_argument("--error-rate", type=float, default=0.0)
    estimate.add_argument("--cost-per-error", type=float, default=0.0)
    compare = commands.add_parser(
        "compare", help="price one file on every model"
    )
    compare.add_argument("file", type=Path)
    compare.add_argument("--output", type=int, default=500)
    compare.add_argument("--calls", type=int, default=1)
    options = parser.parse_args(arguments)

    text = options.file.read_text(encoding="utf-8")
    if options.command == "compare":
        print(f"Characters:    {len(text):,}")
        print(f"Output tokens: {options.output:,} (your guess)")
        print(f"Calls:         {options.calls:,}")
        for row in compare_models(text, options.output, options.calls):
            print(row)
        return

    if not 0 <= options.error_rate <= 1:
        sys.exit(
            f"askkit: --error-rate {options.error_rate:g} is not a share"
            " of the calls; give one from 0 to 1, such as 0.03 for 3 in 100"
        )
    if options.cost_per_error < 0:
        sys.exit(
            f"askkit: --cost-per-error {options.cost_per_error:g} is below"
            " zero; a wrong answer costs something or nothing"
        )
    input_tokens, exact = count_tokens(options.model, text)
    window = LIMITS[options.model]["context_window"]
    max_input = LIMITS[options.model]["max_input"]
    max_output = LIMITS[options.model]["max_output"]
    if options.output > max_output:
        sys.exit(
            f"askkit: --output {options.output:,} is over the most"
            f" {options.model} writes in one call, {max_output:,} tokens"
        )
    if input_tokens > max_input:
        sys.exit(
            f"askkit: {input_tokens:,} input tokens are over the most"
            f" {options.model} reads in one call, {max_input:,} tokens"
        )
    needed = input_tokens + options.output
    if needed > window:
        sys.exit(
            f"askkit: {needed:,} tokens do not fit the context window"
            f" of {options.model}, {window:,} tokens"
        )

    one_call = call_cost(options.model, input_tokens, options.output)
    if exact:
        how = "counted by the model's tokenizer"
    else:
        how = f"estimated from {len(text):,} characters"
    print(f"Model:         {options.model}")
    print(f"Input tokens:  {input_tokens:,} ({how})")
    print(f"Output tokens: {options.output:,} (your guess)")
    print(f"Window:        {needed:,} of {window:,} tokens")
    print(f"One call:      ${one_call:.6f}")
    if options.calls > 1:
        print(f"{options.calls:,} calls: ${one_call * options.calls:,.2f}")
        if options.error_rate > 0:
            wrong = options.calls * options.error_rate
            errors = error_cost(
                options.calls, options.error_rate, options.cost_per_error
            )
            print(f"Errors:        {wrong:,.0f} wrong, ${errors:,.2f}")
            print(f"With errors:   ${one_call * options.calls + errors:,.2f}")
