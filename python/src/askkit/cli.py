import argparse
import json
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
from askkit.classify import (
    check_classification,
    classification_error,
    classify_email,
)
from askkit.clean import clean_email
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
    not_stated,
    retry_ticket,
    ticket_error,
    ticket_faults,
)
from askkit.golden import KINDS, kind_counts, load_cases
from askkit.limits import LIMITS
from askkit.prices import PRICES
from askkit.prompts import ASK_SYSTEM
from askkit.rubric import (
    Answer,
    Grade,
    answer_faults,
    answers_passed,
    criterion_scores,
    grade_answer,
    load_answers,
    load_questions,
    load_rubric,
)
from askkit.rules import email_fault, load_orders, order_fault
from askkit.tokens import count_tokens


def require_api_key() -> str:
    key = load_api_key()
    if key is None:
        sys.exit(
            f"askkit: {KEY_NAME} is not set; copy .env.example to .env"
            " and put your key after the ="
        )
    return key


def print_calls(
    responses: list[Message], model: str, stop_reason: bool = True
) -> None:
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
    # One stop reason says how one answer ended; a command that asked
    # several questions names each answer's own on its line instead.
    if stop_reason:
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
    extract.add_argument("--orders", type=Path)
    classify = commands.add_parser(
        "classify", help="put an email in one of the shop's queues"
    )
    classify.add_argument("file", type=Path)
    classify.add_argument(
        "-m", "--model", choices=MODELS, default="claude-haiku-5-5"
    )
    classify.add_argument("--max-tokens", type=int, default=1000)
    cases_command = commands.add_parser(
        "cases", help="count a golden set's cases by kind"
    )
    cases_command.add_argument("folder", type=Path)
    answer_command = commands.add_parser(
        "answer", help="answer every question of an ask set, to grade"
    )
    answer_command.add_argument("folder", type=Path)
    answer_command.add_argument("--out", type=Path, required=True)
    answer_command.add_argument(
        "-m", "--model", choices=MODELS, default="claude-haiku-5-5"
    )
    answer_command.add_argument("--max-tokens", type=int, default=1000)
    grade_command = commands.add_parser(
        "grade", help="grade saved answers by hand against a rubric"
    )
    grade_command.add_argument("answers", type=Path)
    grade_command.add_argument("--cases", type=Path, required=True)
    grade_command.add_argument("--rubric", type=Path, required=True)
    grade_command.add_argument("--out", type=Path, required=True)
    options = parser.parse_args(arguments)

    if options.command == "key":
        print(f"API key: {masked(require_api_key())}")
        return

    if options.command == "cases":
        # Counts from each part's cases.json. The held-out emails are
        # never opened here: checking the set's shape is not reading the
        # emails kept apart.
        try:
            main_cases = load_cases(options.folder / "main")
            held_out = load_cases(options.folder / "held_out")
        except (OSError, ValueError) as error:
            sys.exit(f"askkit: {error}")
        main_counts = kind_counts(main_cases)
        held_counts = kind_counts(held_out)
        print(f"{'Kind':<14}{'Main':>5}{'Held out':>10}")
        empty: list[str] = []
        for kind in KINDS:
            print(f"{kind:<14}{main_counts[kind]:>5}{held_counts[kind]:>10}")
            if main_counts[kind] == 0 or held_counts[kind] == 0:
                empty.append(kind)
        print(f"{'Total':<14}{len(main_cases):>5}{len(held_out):>10}")
        if empty:
            sys.exit("askkit: no case in one part for " + ", ".join(empty))
        return

    if options.command == "answer":
        # Every question of one part, sent as askkit ask sends it, and the
        # answers saved for a person to grade. What code can check is
        # checked here, so no one spends a minute on a cut-off answer.
        try:
            questions = load_questions(options.folder)
        except (OSError, ValueError) as error:
            sys.exit(f"askkit: {error}")
        client = Anthropic(api_key=require_api_key())
        answers: list[Answer] = []
        calls: list[Message] = []
        for item in questions:
            sources: dict[str, str] = {}
            for name in item.documents:
                path = options.folder.parent / "documents" / name
                sources[name] = path.read_text(encoding="utf-8")
            try:
                response = ask(
                    client,
                    with_documents(item.question, sources),
                    options.model,
                    options.max_tokens,
                    ASK_SYSTEM,
                )
            except APIError as error:
                sys.exit(f"askkit: {error_message(error)}")
            calls.append(response)
            code_faults = answer_faults(response, sources)
            # Of a declined answer no part is kept, as askkit ask prints
            # none of it.
            text = ""
            if response.stop_reason != "refusal":
                text = answer_text(response)
            answers.append(
                Answer(case=item.name, text=text, faults=code_faults)
            )
            state = "to grade"
            if code_faults:
                state = "; ".join(code_faults)
            print(f"{item.name:<14}{state}")
        saved = [one.model_dump() for one in answers]
        options.out.write_text(
            json.dumps(saved, indent=2) + "\n", encoding="utf-8"
        )
        print()
        print_calls(calls, options.model, stop_reason=False)
        print(f"Saved:         {options.out}")
        return

    if options.command == "grade":
        try:
            rubric = load_rubric(options.rubric)
            questions = load_questions(options.cases)
            answers = load_answers(options.answers)
        except (OSError, ValueError) as error:
            sys.exit(f"askkit: {error}")
        asked = {}
        for item in questions:
            asked[item.name] = item
        for one in answers:
            if one.case not in asked:
                sys.exit(f"askkit: no case {one.case!r} in {options.cases}")
        print("Rubric")
        for criterion in rubric:
            print(f"  {criterion.name}: {criterion.asks}")
            print(f"    Passes: {criterion.passes}")
            print(f"    Fails:  {criterion.fails}")
        grades: list[Grade] = []
        number = 0
        for one in answers:
            number += 1
            item = asked[one.case]
            print()
            print(f"Answer {number} of {len(answers)}: {item.name}")
            print(f"Question: {item.question}")
            for fact in item.must:
                print(f"Must:     {fact}")
            print()
            if one.faults:
                print("Failed by code: " + "; ".join(one.faults))
            else:
                print(one.text)
                print()
            grades.extend(grade_answer(one, rubric, input))
        saved = [grade.model_dump() for grade in grades]
        options.out.write_text(
            json.dumps(saved, indent=2) + "\n", encoding="utf-8"
        )
        print()
        scores = criterion_scores(grades, rubric)
        for name in scores:
            passed, graded = scores[name]
            print(f"{name:<14}{passed} of {graded}")
        passed, graded = answers_passed(grades)
        print(f"{'All criteria':<14}{passed} of {graded} answers")
        print(f"Saved:         {options.out}")
        return

    if options.command == "extract":
        if options.retries < 0:
            sys.exit(
                f"askkit: --retries {options.retries} is below zero;"
                " give 0 to never ask again"
            )
        # The model reads the cleaned email, and the rules check against
        # it: a number only in a quoted earlier email is not this email's.
        raw = options.file.read_text(encoding="utf-8")
        email, left_out = clean_email(raw)
        orders: set[int] | None = None
        if options.orders is not None:
            orders = load_orders(options.orders)
        client = Anthropic(api_key=require_api_key())
        responses: list[Message] = []
        notes: list[str] = []
        if left_out:
            notes.append(f"Left out:      {left_out} quoted lines")
        answer = ""
        faults = ""
        rejection = ""
        for attempt in range(options.retries + 1):
            try:
                if attempt == 0:
                    response = extract_ticket(
                        client, email, options.model, options.max_tokens
                    )
                else:
                    notes.append(f"Asked again:   {faults}")
                    response = retry_ticket(
                        client,
                        email,
                        answer,
                        faults,
                        options.model,
                        options.max_tokens,
                    )
            except APIError as error:
                sys.exit(f"askkit: {error_message(error)}")
            responses.append(response)
            # A cut-off or declined answer may not match the schema, so the
            # stop reason is read before the JSON is. Neither is asked for
            # again: the same room or the same email gives the same stop.
            warning = stop_warning(response, "turn this email into a ticket")
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
                faults = ticket_faults(error)
                rejection = ticket_error(error)
                continue
            # The business rules, on a ticket Ticket has passed. A number
            # the email does not give is the model's misreading: asked
            # again, as a broken field is.
            fault = email_fault(ticket, email)
            if fault is not None:
                faults = fault
                rejection = f"the ticket does not match the email: {fault}"
                continue
            # A number the email gives but the shop does not have: asking
            # again returns the same number, so a person sees it at once.
            if orders is not None:
                fault = order_fault(ticket, orders)
                if fault is not None:
                    print_calls(responses, options.model)
                    for note in notes:
                        print(note)
                    sys.exit(
                        f"askkit: {fault}; the email gives this number,"
                        " so a person checks it"
                    )
            if change is not None:
                notes.append(f"Repaired:      {change}")
            missing = not_stated(ticket)
            if missing:
                notes.append(f"Not stated:    {', '.join(missing)}")
            print(ticket.model_dump_json(indent=2))
            print()
            print_calls(responses, options.model)
            for note in notes:
                print(note)
            return
        # Every answer broke the contract or a rule: no ticket, and exit
        # code 1, so whatever runs AskKit hands the email to a person.
        print_calls(responses, options.model)
        for note in notes:
            print(note)
        sys.exit(f"askkit: {rejection}")

    if options.command == "classify":
        # The same cleaning as extract: a quoted earlier email is not this
        # one, and its words could pull the email into the wrong queue.
        email, left_out = clean_email(
            options.file.read_text(encoding="utf-8")
        )
        client = Anthropic(api_key=require_api_key())
        try:
            response = classify_email(
                client, email, options.model, options.max_tokens
            )
        except APIError as error:
            sys.exit(f"askkit: {error_message(error)}")
        warning = stop_warning(response, "sort this email into a queue")
        if warning is not None:
            print_calls([response], options.model)
            sys.exit(f"askkit: {warning}")
        try:
            classification, change = check_classification(
                answer_text(response)
            )
        except ValidationError as error:
            print_calls([response], options.model)
            sys.exit(f"askkit: {classification_error(error)}")
        print(classification.model_dump_json(indent=2))
        print()
        print_calls([response], options.model)
        if left_out:
            print(f"Left out:      {left_out} quoted lines")
        if change is not None:
            print(f"Repaired:      {change}")
        return

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
