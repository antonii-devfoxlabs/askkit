import argparse
import sys
from pathlib import Path

from askkit.cost import call_cost
from askkit.limits import LIMITS
from askkit.prices import PRICES
from askkit.tokens import estimate_tokens


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
    options = parser.parse_args(arguments)

    text = options.file.read_text(encoding="utf-8")
    input_tokens = estimate_tokens(text)
    window = LIMITS[options.model]["context_window"]
    max_output = LIMITS[options.model]["max_output"]
    if options.output > max_output:
        sys.exit(
            f"askkit: --output {options.output:,} is over the most"
            f" {options.model} writes in one call, {max_output:,} tokens"
        )
    context = input_tokens + options.output
    if context > window:
        sys.exit(
            f"askkit: {context:,} tokens do not fit the context window"
            f" of {options.model}, {window:,} tokens"
        )

    one_call = call_cost(options.model, input_tokens, options.output)
    print(f"Model:         {options.model}")
    print(f"Input tokens:  {input_tokens:,} (from {len(text):,} characters)")
    print(f"Output tokens: {options.output:,} (your guess)")
    print(f"Context:       {context:,} of {window:,} tokens")
    print(f"One call:      ${one_call:.6f}")
    if options.calls > 1:
        print(f"{options.calls:,} calls: ${one_call * options.calls:,.2f}")
