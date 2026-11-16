import argparse
import sys
from pathlib import Path

from anthropic import Anthropic, APIError
from anthropic.types import Message
from pydantic import ValidationError

from askkit.ask import (
    EFFORTS,
    MODELS,
    answer_text,
    ask,
    stop_warning,
    thinking_tokens,
)
from askkit.compare import compare_models
from askkit.config import KEY_NAME, load_api_key, masked
from askkit.cost import call_cost, error_cost
from askkit.documents import (
    missing_quotes,
    quotes_in,
    unchecked,
    with_documents,
)
from askkit.errors import error_message
from askkit.extract import (
    check_ticket,
    extract_ticket,
    retry_ticket,
    ticket_error,
    ticket_problems,
)
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


def print_calls(responses: list[Message], model: str) -> None:
    # One summary for every call a command made: the tokens and the cost
    # are added up, so a ticket asked for twice shows what both cost.
    input_tokens = 0
    output_tokens = 0
    thinking: int | None = None
    for response in responses:
        input_tokens += response.usage.input_tokens
        output_tokens += response.usage.output_tokens
        thought = thinking_tokens(response)
        if thought is not None:
            thinking = (thinking or 0) + thought
    last = responses[-1]
    cost = call_cost(model, input_tokens, output_tokens)
    print(f"Model:         {last.model}")
    tokens = f"{input_tokens:,} in, {output_tokens:,} out"
    if thinking is not None:
        tokens += f" ({thinking:,} thinking)"
    print(f"Tokens:        {tokens}")
    print(f"Cost:          ${cost:.6f}")
    print(f"Stop reason:   {last.stop_reason}")


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
    ask_command.add_argument("--effort", choices=EFFORTS)
    ask_command.add_argument("--system-file", type=Path)
    ask_command.add_argument(
        "--document", type=Path, action="append", default=[]
    )
    extract = commands.add_parser(
        "extract", help="turn an email into a ticket, as JSON"
    )
    extract.add_argument("file", type=Path)
    extract.add_argument(
        "-m", "--model", choices=MODELS, default="claude-haiku-5-5"
    )
    extract.add_argument("--max-tokens", type=int, default=1000)
    extract.add_argument("--retries", type=int, default=1)
    options = parser.parse_args(arguments)

    if options.command == "key":
        print(f"API key: {masked(require_api_key())}")
        return

    if options.command == "extract":
        if options.retries < 0:
            sys.exit(
                f"askkit: --retries {options.retries} is below zero;"
                " give 0 to never ask again"
            )
        email = options.file.read_text(encoding="utf-8")
        client = Anthropic(api_key=require_api_key())
        responses: list[Message] = []
        notes: list[str] = []
        answer = ""
        problems = ""
        rejection = ""
        for attempt in range(options.retries + 1):
            try:
                if attempt == 0:
                    response = extract_ticket(
                        client, email, options.model, options.max_tokens
                    )
                else:
                    notes.append(f"Asked again:   {problems}")
                    response = retry_ticket(
                        client,
                        email,
                        answer,
                        problems,
                        options.model,
                        options.max_tokens,
                    )
            except APIError as error:
                sys.exit(f"askkit: {error_message(error)}")
            responses.append(response)
            # A cut-off or declined answer may not match the schema, so the
            # stop reason is read before the JSON is. Neither is asked for
            # again: the same room or the same email gives the same stop.
            warning = stop_warning(response)
            if warning is not None:
                print_calls(responses, options.model)
                sys.exit(f"askkit: {warning}")
            answer = answer_text(response)
            # Ticket checks every field again here, in our own code: the
            # API holds the shape, but not the capitals of an allowed value
            # or the length of a text, and nothing reaches the warehouse
            # unchecked.
            try:
                ticket, change = check_ticket(answer)
            except ValidationError as error:
                problems = ticket_problems(error)
                rejection = ticket_error(error)
                continue
            if change is not None:
                notes.append(f"Repaired:      {change}")
            print(ticket.model_dump_json(indent=2))
            print()
            print_calls(responses, options.model)
            for note in notes:
                print(note)
            return
        # Every answer broke the contract: no ticket, and exit code 1, so
        # whatever runs AskKit hands the email to a person.
        print_calls(responses, options.model)
        for note in notes:
            print(note)
        sys.exit(f"askkit: {rejection}")

    if options.command == "ask":
        system = ASK_SYSTEM
        if options.system_file is not None:
            system = options.system_file.read_text(encoding="utf-8").strip()
        documents: dict[str, str] = {}
        for path in options.document:
            documents[str(path)] = path.read_text(encoding="utf-8")
        question = options.question
        if documents:
            question = with_documents(options.question, documents)
        client = Anthropic(api_key=require_api_key())
        try:
            response = ask(
                client,
                question,
                options.model,
                options.max_tokens,
                system,
                options.effort,
            )
        except APIError as error:
            sys.exit(f"askkit: {error_message(error)}")
        text = answer_text(response)
        # Anthropic advises discarding the part of a declined answer.
        if response.stop_reason != "refusal":
            print(text)
            print()
        print_calls([response], options.model)
        warning = stop_warning(response)
        if warning is not None:
            sys.exit(f"askkit: {warning}")
        if documents:
            quotes = quotes_in(text)
            missing = missing_quotes(text, documents)
            if not quotes:
                print("Quotes:        none in the answer")
            else:
                found = len(quotes) - len(missing)
                print(
                    f"Quotes:        {found} of {len(quotes)}"
                    " found in the documents"
                )
            if missing:
                sys.exit(f'askkit: not in the documents: "{missing[0]}"')
            if unchecked(text):
                sys.exit(
                    "askkit: the answer quotes nothing from the documents,"
                    " so it cannot be checked"
                )
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
