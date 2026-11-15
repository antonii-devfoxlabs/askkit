import argparse
from pathlib import Path

from askkit.cost import call_cost
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
        "--model", choices=sorted(PRICES), default="claude-haiku-5-5"
    )
    estimate.add_argument("--output-tokens", type=int, default=500)
    estimate.add_argument("--calls", type=int, default=1)
    options = parser.parse_args(arguments)

    text = options.file.read_text(encoding="utf-8")
    input_tokens = estimate_tokens(text)
    one_call = call_cost(options.model, input_tokens, options.output_tokens)
    print(f"Model:         {options.model}")
    print(f"Input tokens:  {input_tokens:,} (from {len(text):,} characters)")
    print(f"Output tokens: {options.output_tokens:,} (your guess)")
    print(f"One call:      ${one_call:.6f}")
    print(f"{options.calls:,} calls: ${one_call * options.calls:,.2f}")
