import argparse
import sys
from pathlib import Path

from anthropic import Anthropic, APIError

from askkit.ask import MODELS, answer_text, ask, stop_warning, thinking_tokens
from askkit.compare import compare_models
from askkit.config import KEY_NAME, load_api_key, masked
from askkit.cost import call_cost, error_cost
from askkit.errors import error_message
from askkit.limits import LIMITS
from askkit.prices import PRICES
from askkit.prompts import ASK_SYSTEM
from askkit.tokens import count_tokens


def require_api_key() -> str:
    key = load_api_key()
    if key is None:
        sys.exit(
            f"askkit: {KEY_NAME} is not set; copy .env.example to .env"
            " and put your key after the ="
        )
    return key


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
    commands.add_parser("key", help="check that AskKit finds your API key")
    ask_command = commands.add_parser("ask", help="ask a model a question")
    ask_command.add_argument("question")
    ask_command.add_argument(
        "-m", "--model", choices=MODELS, default="claude-haiku-5-5"
    )
    ask_command.add_argument("--max-tokens", type=int, default=1000)
    ask_command.add_argument("--system-file", type=Path)
    options = parser.parse_args(arguments)

    if options.command == "key":
        print(f"API key: {masked(require_api_key())}")
        return

    if options.command == "ask":
        system = ASK_SYSTEM
        if options.system_file is not None:
            system = options.system_file.read_text(encoding="utf-8").strip()
        client = Anthropic(api_key=require_api_key())
        try:
            response = ask(
                client,
                options.question,
                options.model,
                options.max_tokens,
                system,
            )
        except APIError as error:
            sys.exit(f"askkit: {error_message(error)}")
        usage = response.usage
        cost = call_cost(
            options.model, usage.input_tokens, usage.output_tokens
        )
        # Anthropic advises discarding the part of a declined answer.
        if response.stop_reason != "refusal":
            print(answer_text(response))
            print()
        print(f"Model:         {response.model}")
        tokens = f"{usage.input_tokens:,} in, {usage.output_tokens:,} out"
        thinking = thinking_tokens(response)
        if thinking is not None:
            tokens += f" ({thinking:,} thinking)"
        print(f"Tokens:        {tokens}")
        print(f"Cost:          ${cost:.6f}")
        print(f"Stop reason:   {response.stop_reason}")
        warning = stop_warning(response)
        if warning is not None:
            sys.exit(f"askkit: {warning}")
        return

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
