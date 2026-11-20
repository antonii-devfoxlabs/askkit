import argparse
import json
import sys
from dataclasses import replace
from datetime import date
from functools import partial
from pathlib import Path

import openai
from anthropic import Anthropic, APIError, Timeout
from anthropic.types import Message, ToolUnionParam
from openai import OpenAI
from pydantic import ValidationError

from askkit.analysis import fault_lines
from askkit.ask import (
    CANCELLED,
    DECLINED_PART,
    EFFORTS,
    MODELS,
    Content,
    Timing,
    answer_text,
    ask,
    stop_warning,
    thinking_tokens,
    timing_line,
)
from askkit.audio import (
    TRANSCRIBE_MODEL,
    audio_fault,
    audio_format,
    audio_seconds,
    openai_error,
    transcribe,
    transcription_cost,
)
from askkit.budget import DEFAULT_BUDGET, budget_fault
from askkit.chat import STOP, chat
from askkit.classify import (
    CLASSIFICATION_SCHEMA,
    check_classification,
    classification_error,
)
from askkit.clean import clean_email
from askkit.compare import compare_models
from askkit.config import KEY_NAME, OPENAI_KEY_NAME, load_api_key, masked
from askkit.cost import call_cost, error_cost, prompt_tokens, usage_cost
from askkit.costmodel import load_cost_model, month_costs, report_lines
from askkit.documents import (
    document_question,
    missing_quotes,
    quotes_in,
    unchecked,
    with_documents,
)
from askkit.errors import error_message
from askkit.extract import (
    TICKET_SCHEMA,
    check_ticket,
    not_stated,
    ticket_error,
    ticket_faults,
    ticket_messages,
)
from askkit.fallback import (
    OPENAI_MODELS,
    FirstAnswer,
    ask_openai,
    fallbacks_line,
    first_answer,
)
from askkit.golden import KINDS, Case, kind_counts, load_cases
from askkit.history import (
    SavedMessage,
    as_messages,
    history_fault,
    load_history,
    save_history,
)
from askkit.images import is_heic, media_type_of, with_images
from askkit.inputs import image_input, pdf_input
from askkit.invoices import (
    INVOICE_SCHEMA,
    Invoice,
    invoice_error,
    invoice_faults,
)
from askkit.limits import LIMITS
from askkit.loop import (
    TOO_MANY_ROUNDS,
    final_text,
    requests_line,
    tool_loop,
)
from askkit.memory import load_memory
from askkit.models import (
    CHECKED,
    DEFAULT_MODEL,
    LIFECYCLES,
    claude_lifecycle,
    lifecycle_change,
    openai_lifecycle,
    retirement_fault,
    retirement_warning,
    retires_text,
    table_lines,
)
from askkit.pdfs import document_block, is_pdf
from askkit.permissions import offered, permissions_for
from askkit.prices import PRICES, SEARCH_PRICE
from askkit.prompts import ASK_SYSTEM, render
from askkit.retry import (
    ANSWER_POLICY,
    ANSWER_TIMEOUT,
    CONNECT_TIMEOUT,
    Retry,
    RetryPolicy,
    after_tries,
    retries_line,
    timeout_fault,
    with_retries,
)
from askkit.routing import call_model
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
from askkit.runs import Results, diff_lines, load_results
from askkit.server_tools import cited_urls, searches, web_search
from askkit.structured import stream_json
from askkit.tokens import count_tokens
from askkit.tools import TOOLS
from askkit.trim import OLD_TURNS


def require_api_key() -> str:
    key = load_api_key()
    if key is None:
        sys.exit(
            f"askkit: {KEY_NAME} is not set; copy .env.example to .env"
            " and put your key after the ="
        )
    return key


def require_history(path: Path) -> list[SavedMessage]:
    # Read and checked before the client is made, so a broken file costs
    # nothing.
    try:
        history = load_history(path)
    except ValueError:
        # Not JSON (JSONDecodeError), not a list, or not AskKit's
        # messages (ValidationError): each is a ValueError.
        sys.exit(f"askkit: {path} is not a conversation AskKit saved")
    fault = history_fault(history)
    if fault is not None:
        sys.exit(f"askkit: {path}: {fault}")
    return history


def require_memory(path: Path) -> list[str]:
    # Read before the client is made, as the history is, so a broken file
    # costs nothing.
    try:
        return load_memory(path)
    except ValueError:
        # Not JSON, or not AskKit's notes: both are ValueErrors.
        sys.exit(f"askkit: {path} is not a memory AskKit saved")


def send_to(
    model: str,
    *,
    client: Anthropic,
    openai_client: OpenAI | None,
    question: str,
    max_tokens: int,
    policy: RetryPolicy,
    tried: list[Retry],
    tries_made: list[int],
) -> Message:
    # One question to one model of askkit answer's chain. An OpenAI model
    # gets the same question through OpenAI's SDK, whose own retries own
    # its tries. A Claude model goes through AskKit's retry policy: each
    # retry is added to tried, and the number of tries this model made to
    # tries_made, so a message names the last model's tries, not all.
    if model in OPENAI_MODELS and openai_client is not None:
        return ask_openai(
            openai_client, question, model, max_tokens, ASK_SYSTEM
        )
    # One question's call is retried, never the run: the answers before
    # it are not asked for, or paid for, again. partial(ask, ...) is ask
    # with its arguments filled in, a call with_retries can make as often
    # as it needs.
    call = partial(ask, client, question, model, max_tokens, ASK_SYSTEM)
    this_model: list[Retry] = []
    try:
        return with_retries(call, policy, this_model)
    finally:
        tried.extend(this_model)
        tries_made.append(len(this_model) + 1)


def print_calls(
    responses: list[Message],
    model: str,
    stop_reason: bool = True,
    models: list[str] | None = None,
) -> None:
    # One summary for every call a command made: the tokens and the cost
    # are added up, so a ticket asked for twice shows what both cost.
    # models names each call's model when the calls went to more than
    # one (extract --escalate-to); every call is priced at its own.
    if models is None:
        models = [model] * len(responses)
    input_tokens = 0
    output_tokens = 0
    written = 0
    read = 0
    cost = 0.0
    thinking: int | None = None
    for response, priced_as in zip(responses, models, strict=True):
        usage = response.usage
        input_tokens += prompt_tokens(usage)
        output_tokens += usage.output_tokens
        written += usage.cache_creation_input_tokens or 0
        read += usage.cache_read_input_tokens or 0
        # Call by call: each call's prompt decides its prices.
        cost += usage_cost(priced_as, usage)
        thought = thinking_tokens(response)
        if thought is not None:
            thinking = (thinking or 0) + thought
    last = responses[-1]
    searched = searches(responses)
    # Searches are billed on top of the tokens their results add.
    cost += searched * SEARCH_PRICE
    names: list[str] = []
    for response in responses:
        if response.model not in names:
            names.append(response.model)
    print(f"Model:         {', then '.join(names)}")
    tokens = f"{input_tokens:,} in, {output_tokens:,} out"
    if thinking is not None:
        tokens += f" ({thinking:,} thinking)"
    print(f"Tokens:        {tokens}")
    if read or written:
        # How many of the input tokens the cache served, and how many
        # this command wrote to it for the next calls.
        print(f"Cache:         {read:,} read, {written:,} written")
    if searched:
        print(f"Searches:      {searched}, ${searched * SEARCH_PRICE:.2f}")
    print(f"Cost:          ${cost:.6f}")
    # One stop reason says how one answer ended; a command that asked
    # several questions names each answer's own on its line instead.
    if stop_reason:
        print(f"Stop reason:   {last.stop_reason}")


# What input() raises when no one answers: no line left to read (input
# from a pipe or a file), or Ctrl-C (Control-C on a Mac).
NO_ANSWER = (EOFError, KeyboardInterrupt)


def confirm_on_terminal(request: str) -> bool:
    # Asked on stderr, so a program reading AskKit's answer never reads
    # the question. Only y or yes runs the tool; anything else, or no one
    # at the keyboard to answer, is a no.
    print(f"Run {request}? [y/N] ", end="", file=sys.stderr, flush=True)
    try:
        answer = input()
    except NO_ANSWER:
        print(file=sys.stderr)
        return False
    return answer.strip().lower() in ("y", "yes")


def print_times(timings: list[Timing]) -> None:
    # One line per call, in order: the times of two calls do not add up
    # the way their tokens do.
    for timing in timings:
        print(f"Time:          {timing_line(timing)}")


def today() -> date:
    # The one place the CLI reads the date, so a test can set it.
    return date.today()


def models_called(options: argparse.Namespace) -> list[str]:
    # The models a command will call, known before its first call.
    if options.command == "transcribe":
        return [TRANSCRIBE_MODEL]
    if options.command == "answer":
        return [options.model, *options.fallback]
    if options.command == "extract" and options.escalate_to is not None:
        return [options.model, options.escalate_to]
    if options.command in ("ask", "chat", "extract", "classify", "read"):
        return [options.model]
    return []


def require_current(models: list[str], on: date) -> None:
    # Every model of the command is checked before any call, the last
    # one of a fallback chain too: a retired model would fail every
    # request with a 404, and a fallback does not hide that (lesson
    # 10.5). A model that retires soon still runs, and is named.
    for model in models:
        fault = retirement_fault(model, on)
        if fault is not None:
            sys.exit(f"askkit: {fault}")
    for model in models:
        warning = retirement_warning(model, on)
        if warning is not None:
            print(f"askkit: warning: {warning}", file=sys.stderr)


def live_changes() -> list[str]:
    # What each provider says today, against the dates AskKit acts on.
    # The table is not changed: a person reads the notice and updates
    # its row.
    client = Anthropic(api_key=require_api_key())
    openai_key = load_api_key(OPENAI_KEY_NAME)
    openai_client = None if openai_key is None else OpenAI(api_key=openai_key)
    changes: list[str] = []
    for model, pinned in LIFECYCLES.items():
        try:
            if pinned.provider == "Anthropic":
                live = claude_lifecycle(client, model)
            elif openai_client is not None:
                live = openai_lifecycle(openai_client, model)
            else:
                print(
                    f"Not checked:   {model} ({OPENAI_KEY_NAME} is not set)"
                )
                continue
        except APIError as error:
            sys.exit(f"askkit: {model}: {error_message(error)}")
        except openai.APIError as error:
            sys.exit(f"askkit: {model}: {openai_error(error)}")
        print(f"Live:          {model}, {retires_text(live)}")
        change = lifecycle_change(model, pinned, live)
        if change is not None:
            changes.append(change)
    return changes


def main(arguments: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="askkit")
    commands = parser.add_subparsers(dest="command", required=True)
    estimate = commands.add_parser(
        "estimate", help="estimate what sending a file would cost"
    )
    estimate.add_argument("file", type=Path)
    estimate.add_argument(
        "-m", "--model", choices=sorted(PRICES), default=DEFAULT_MODEL
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
    costs_command = commands.add_parser(
        "costs", help="what each feature costs a month, from a cost model"
    )
    costs_command.add_argument("file", type=Path)
    commands.add_parser("key", help="check that AskKit finds your API key")
    models_command = commands.add_parser(
        "models", help="list the models AskKit calls and when each retires"
    )
    # Also ask each provider, to find a date models.py does not have yet.
    models_command.add_argument("--live", action="store_true")
    ask_command = commands.add_parser("ask", help="ask a model a question")
    ask_command.add_argument("question")
    ask_command.add_argument(
        "-m", "--model", choices=MODELS, default=DEFAULT_MODEL
    )
    ask_command.add_argument("--max-tokens", type=int, default=1000)
    ask_command.add_argument("--effort", choices=EFFORTS)
    ask_command.add_argument("--system-file", type=Path)
    ask_command.add_argument(
        "--document", type=Path, action="append", default=[]
    )
    ask_command.add_argument(
        "--image", type=Path, action="append", default=[], metavar="FILE"
    )
    ask_command.add_argument("--history", type=Path)
    tool_names = [tool["name"] for tool in TOOLS]
    ask_command.add_argument(
        "--confirm", choices=tool_names, action="append", default=[]
    )
    ask_command.add_argument(
        "--deny", choices=tool_names, action="append", default=[]
    )
    ask_command.add_argument(
        "--allow-site", action="append", default=[], metavar="HOST"
    )
    ask_command.add_argument("--use-tool", choices=tool_names)
    ask_command.add_argument(
        "--search-site", action="append", default=[], metavar="DOMAIN"
    )
    chat_command = commands.add_parser(
        "chat", help="chat with a model until you type /exit"
    )
    chat_command.add_argument(
        "-m", "--model", choices=MODELS, default=DEFAULT_MODEL
    )
    chat_command.add_argument("--max-tokens", type=int, default=1000)
    chat_command.add_argument("--effort", choices=EFFORTS)
    chat_command.add_argument("--system-file", type=Path)
    chat_command.add_argument(
        "--history", type=Path, default=Path("chat.json")
    )
    chat_command.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    chat_command.add_argument(
        "--old-turns", choices=OLD_TURNS, default="summarise"
    )
    chat_command.add_argument(
        "--memory", type=Path, default=Path("memory.json")
    )
    extract = commands.add_parser(
        "extract", help="turn an email into a ticket, as JSON"
    )
    extract.add_argument("file", type=Path)
    extract.add_argument(
        "-m", "--model", choices=MODELS, default=DEFAULT_MODEL
    )
    extract.add_argument("--max-tokens", type=int, default=1000)
    extract.add_argument("--retries", type=int, default=1)
    extract.add_argument("--escalate-to", choices=MODELS)
    extract.add_argument("--orders", type=Path)
    classify = commands.add_parser(
        "classify", help="put an email in one of the shop's queues"
    )
    classify.add_argument("file", type=Path)
    classify.add_argument(
        "-m", "--model", choices=MODELS, default=DEFAULT_MODEL
    )
    classify.add_argument("--max-tokens", type=int, default=1000)
    read_command = commands.add_parser(
        "read", help="take the fields out of an invoice PDF, as JSON"
    )
    read_command.add_argument("file", type=Path)
    read_command.add_argument(
        "-m", "--model", choices=MODELS, default=DEFAULT_MODEL
    )
    read_command.add_argument("--max-tokens", type=int, default=1000)
    transcribe_command = commands.add_parser(
        "transcribe", help="turn a voice message into text"
    )
    transcribe_command.add_argument("file", type=Path)
    transcribe_command.add_argument("--out", type=Path)
    cases_command = commands.add_parser(
        "cases", help="count a golden set's cases by kind"
    )
    cases_command.add_argument("folder", type=Path)
    answer_command = commands.add_parser(
        "answer", help="answer every question of an ask set, to grade"
    )
    answer_command.add_argument("folder", type=Path)
    answer_command.add_argument("--out", type=Path, required=True)
    # A fallback model can be the first one too, to answer the whole set
    # with it and grade it before you rely on it.
    answer_command.add_argument(
        "-m",
        "--model",
        choices=[*MODELS, *OPENAI_MODELS],
        default=DEFAULT_MODEL,
    )
    answer_command.add_argument("--max-tokens", type=int, default=1000)
    answer_command.add_argument(
        "--tries", type=int, default=ANSWER_POLICY.tries
    )
    answer_command.add_argument(
        "--timeout", type=float, default=ANSWER_TIMEOUT
    )
    # Models to ask, in order, when the one before them cannot answer.
    answer_command.add_argument(
        "--fallback",
        action="append",
        default=[],
        choices=[*MODELS, *OPENAI_MODELS],
    )
    grade_command = commands.add_parser(
        "grade", help="grade saved answers by hand against a rubric"
    )
    grade_command.add_argument("answers", type=Path)
    grade_command.add_argument("--cases", type=Path, required=True)
    grade_command.add_argument("--rubric", type=Path, required=True)
    grade_command.add_argument("--out", type=Path, required=True)
    diff_command = commands.add_parser(
        "diff", help="compare two results files of the eval, case by case"
    )
    diff_command.add_argument("old", type=Path)
    diff_command.add_argument("new", type=Path)
    faults_command = commands.add_parser(
        "faults", help="sort the failed answers of eval runs, to read them"
    )
    faults_command.add_argument("results", type=Path, nargs="+")
    faults_command.add_argument(
        "--cases", type=Path, default=Path("evals/extract/main")
    )
    options = parser.parse_args(arguments)
    require_current(models_called(options), today())

    if options.command == "models":
        # The dates AskKit acts on, from models.py; no call unless --live.
        for line in table_lines():
            print(line)
        print()
        print(f"Default:       {DEFAULT_MODEL}")
        print(
            f"Checked:       {CHECKED}, on each provider's deprecations page"
        )
        on = today()
        problems: list[str] = []
        for model in LIFECYCLES:
            fault = retirement_fault(model, on)
            if fault is not None:
                problems.append(fault)
            warning = retirement_warning(model, on)
            if warning is not None:
                print(f"Warning:       {warning}")
        if options.live:
            problems.extend(live_changes())
        if problems:
            sys.exit("askkit: " + "\naskkit: ".join(problems))
        return

    if options.command == "costs":
        # No model call: a file of features, each with its uses a month and
        # what one use sends, priced from prices.py.
        try:
            cost_model = load_cost_model(options.file)
        except (OSError, ValueError) as error:
            sys.exit(f"askkit: {error}")
        for line in report_lines(month_costs(cost_model)):
            print(line)
        return

    if options.command == "key":
        print(f"API key: {masked(require_api_key())}")
        return

    if options.command == "transcribe":
        try:
            data = options.file.read_bytes()
        except OSError as error:
            reason = f"{error.filename}: {error.strerror}"
            sys.exit(f"askkit: cannot read {reason}")
        fault = audio_fault(options.file, data)
        if fault is not None:
            sys.exit(f"askkit: {fault}")
        openai_key = load_api_key(OPENAI_KEY_NAME)
        if openai_key is None:
            sys.exit(
                f"askkit: {OPENAI_KEY_NAME} is not set (transcribe runs at"
                " OpenAI); put your OpenAI key after the = in .env"
            )
        openai_client = OpenAI(api_key=openai_key)
        try:
            transcript = transcribe(
                openai_client, options.file, data, TRANSCRIBE_MODEL
            )
        except openai.APIError as error:
            sys.exit(f"askkit: {openai_error(error)}")
        print(transcript.text)
        print()
        print(f"Model:         {TRANSCRIBE_MODEL}")
        usage = transcript.usage
        if usage is not None and usage.type == "tokens":
            tokens = f"{usage.input_tokens:,} in, {usage.output_tokens:,} out"
            print(f"Tokens:        {tokens}")
        seconds = audio_seconds(data)
        kind = audio_format(data)
        if seconds is None:
            print(f"Audio:         {kind}, length unknown")
            print("Cost:          unknown: only a WAV gives its length")
        else:
            print(f"Audio:         {kind}, {seconds:.1f} s")
            print(f"Cost:          ${transcription_cost(seconds):.6f}")
        if options.out is not None:
            options.out.write_text(transcript.text + "\n", encoding="utf-8")
            print(f"Saved:         {options.out}")
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

    if options.command == "diff":
        # No model call: two files the eval wrote, read side by side.
        try:
            old = load_results(options.old)
            new = load_results(options.new)
            lines = diff_lines(options.old.name, old, options.new.name, new)
        except (OSError, ValueError) as error:
            sys.exit(f"askkit: {error}")
        for line in lines:
            print(line)
        return

    if options.command == "faults":
        # No model call either: the failures the eval stored, sorted by
        # what code can tell, for a person to read.
        try:
            cases: dict[str, Case] = {}
            for case in load_cases(options.cases):
                cases[case.email] = case
            files: dict[str, Results] = {}
            for path in options.results:
                files[path.name] = load_results(path)
            lines = fault_lines(files, cases)
        except (OSError, ValueError) as error:
            sys.exit(f"askkit: {error}")
        for line in lines:
            print(line)
        return

    if options.command == "answer":
        # Every question of one part, sent as askkit ask sends it, and the
        # answers saved for a person to grade. What code can check is
        # checked here, so no one spends a minute on a cut-off answer.
        fault = timeout_fault(
            options.tries, options.timeout, options.max_tokens
        )
        if fault is not None:
            sys.exit(f"askkit: {fault}")
        chain = [options.model, *options.fallback]
        for model in options.fallback:
            if chain.count(model) > 1:
                sys.exit(
                    f"askkit: --fallback {model} is already in the chain;"
                    " fall back to another model"
                )
        try:
            questions = load_questions(options.folder)
        except (OSError, ValueError) as error:
            sys.exit(f"askkit: {error}")
        policy = replace(ANSWER_POLICY, tries=options.tries)
        timeout = Timeout(options.timeout, connect=CONNECT_TIMEOUT)
        # AskKit retries these calls itself, so the SDK does not: its own
        # two retries inside each of AskKit's tries would send up to
        # three times as many requests. connect: seconds to reach the
        # API; the timeout: the longest wait for the answer.
        client = Anthropic(
            api_key=require_api_key(), max_retries=0, timeout=timeout
        )
        fallback_client: OpenAI | None = None
        if any(model in OPENAI_MODELS for model in chain):
            # Checked before any call, so a missing key costs nothing.
            fallback_key = load_api_key(OPENAI_KEY_NAME)
            if fallback_key is None:
                sys.exit(
                    f"askkit: {OPENAI_KEY_NAME} is not set (a fallback in"
                    " the chain runs at OpenAI); put your OpenAI key after"
                    " the = in .env"
                )
            # The SDK's own two retries own this provider's tries.
            fallback_client = OpenAI(api_key=fallback_key, timeout=timeout)
        answers: list[Answer] = []
        calls: list[Message] = []
        priced_as: list[str] = []
        retries: list[Retry] = []
        outcomes: list[tuple[str, FirstAnswer]] = []
        # When each model that failed may be asked again, for the run.
        down_until: dict[str, float] = {}
        for item in questions:
            sources: dict[str, str] = {}
            for name in item.documents:
                path = options.folder.parent / "documents" / name
                sources[name] = path.read_text(encoding="utf-8")
            question = with_documents(item.question, sources)
            tried: list[Retry] = []
            tries_made: list[int] = []
            send = partial(
                send_to,
                client=client,
                openai_client=fallback_client,
                question=question,
                max_tokens=options.max_tokens,
                policy=policy,
                tried=tried,
                tries_made=tries_made,
            )
            try:
                outcome = first_answer(chain, send, down_until)
            except APIError as error:
                tries = after_tries(tries_made[-1])
                sys.exit(
                    f"askkit: {item.name}: {error_message(error, tries)}"
                )
            except openai.APIError as error:
                sys.exit(f"askkit: {item.name}: {openai_error(error)}")
            except ValueError as error:
                sys.exit(f"askkit: {item.name}: {error}")
            retries.extend(tried)
            outcomes.append((item.name, outcome))
            # A declined answer that handed the question on may be billed
            # too, so it is priced with the rest.
            for declined, model in outcome.declined:
                calls.append(declined)
                priced_as.append(model)
            response = outcome.response
            calls.append(response)
            priced_as.append(outcome.model)
            code_faults = answer_faults(response, sources)
            # Of a declined answer no part is kept, as askkit ask prints
            # none of it.
            text = ""
            if response.stop_reason != "refusal":
                text = answer_text(response)
            answers.append(
                Answer(
                    case=item.name,
                    model=outcome.model,
                    text=text,
                    faults=code_faults,
                )
            )
            state = "to grade"
            if code_faults:
                state = "; ".join(code_faults)
            if outcome.model != options.model:
                state += f" ({outcome.model})"
            print(f"{item.name:<14}{state}")
        saved = [one.model_dump() for one in answers]
        options.out.write_text(
            json.dumps(saved, indent=2) + "\n", encoding="utf-8"
        )
        print()
        print_calls(calls, options.model, stop_reason=False, models=priced_as)
        if retries:
            print(retries_line(retries))
        handed_on = fallbacks_line(outcomes)
        if handed_on is not None:
            print(handed_on)
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
        timings: list[Timing] = []
        # Each call's model: the first goes to --model, a retry to
        # --escalate-to when it is given (lesson 10.3).
        models: list[str] = []
        answer = ""
        faults = ""
        rejection = ""
        for attempt in range(options.retries + 1):
            model = call_model(attempt, options.model, options.escalate_to)
            messages = ticket_messages(email)
            if attempt > 0:
                notes.append(f"Asked again:   {faults}")
                if model != options.model:
                    notes.append(f"Escalated:     to {model}")
                messages = ticket_messages(email, answer, faults)
            try:
                reply, timing = stream_json(
                    client,
                    model,
                    options.max_tokens,
                    render("extract_ticket", {}),
                    messages,
                    TICKET_SCHEMA,
                )
            except APIError as error:
                sys.exit(f"askkit: {error_message(error)}")
            if reply is None:
                # Cancelled with Ctrl-C: the fields shown are a preview of
                # a ticket that never came, and nothing is used.
                if responses:
                    print_calls(responses, options.model, models=models)
                print_times(timings)
                print(f"Time:          {timing_line(timing, cancelled=True)}")
                sys.exit(f"askkit: {CANCELLED}")
            response = reply
            responses.append(response)
            models.append(model)
            timings.append(timing)
            # A cut-off or declined answer may not match the schema, so the
            # stop reason is read before the JSON is. Neither is asked for
            # again: the same room or the same email gives the same stop.
            warning = stop_warning(response, "turn this email into a ticket")
            if warning is not None:
                print_calls(responses, options.model, models=models)
                print_times(timings)
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
                    print_calls(responses, options.model, models=models)
                    print_times(timings)
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
            print_calls(responses, options.model, models=models)
            print_times(timings)
            for note in notes:
                print(note)
            return
        # Every answer broke the contract or a rule: no ticket, and exit
        # code 1, so whatever runs AskKit hands the email to a person.
        print_calls(responses, options.model, models=models)
        print_times(timings)
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
            reply, timing = stream_json(
                client,
                options.model,
                options.max_tokens,
                render("classify_email", {}),
                [{"role": "user", "content": email}],
                CLASSIFICATION_SCHEMA,
            )
        except APIError as error:
            sys.exit(f"askkit: {error_message(error)}")
        if reply is None:
            print(f"Time:          {timing_line(timing, cancelled=True)}")
            sys.exit(f"askkit: {CANCELLED}")
        response = reply
        warning = stop_warning(response, "sort this email into a queue")
        if warning is not None:
            print_calls([response], options.model)
            print_times([timing])
            sys.exit(f"askkit: {warning}")
        try:
            classification, change = check_classification(
                answer_text(response)
            )
        except ValidationError as error:
            print_calls([response], options.model)
            print_times([timing])
            sys.exit(f"askkit: {classification_error(error)}")
        print(classification.model_dump_json(indent=2))
        print()
        print_calls([response], options.model)
        print_times([timing])
        if left_out:
            print(f"Left out:      {left_out} quoted lines")
        if change is not None:
            print(f"Repaired:      {change}")
        return

    if options.command == "read":
        try:
            window = LIMITS[options.model]["context_window"]
            document = document_block(options.file, window)
        except OSError as error:
            reason = f"{error.filename}: {error.strerror}"
            sys.exit(f"askkit: cannot read {reason}")
        except ValueError as error:
            # Not a PDF: stopped before a call the API would reject.
            sys.exit(f"askkit: {error}")
        client = Anthropic(api_key=require_api_key())
        try:
            reply, timing = stream_json(
                client,
                options.model,
                options.max_tokens,
                render("read_invoice", {}),
                [{"role": "user", "content": [document]}],
                INVOICE_SCHEMA,
            )
        except APIError as error:
            sys.exit(f"askkit: {error_message(error)}")
        if reply is None:
            print(f"Time:          {timing_line(timing, cancelled=True)}")
            sys.exit(f"askkit: {CANCELLED}")
        response = reply
        warning = stop_warning(response, "read this invoice")
        if warning is not None:
            print_calls([response], options.model)
            print_times([timing])
            sys.exit(f"askkit: {warning}")
        try:
            invoice = Invoice.model_validate_json(answer_text(response))
        except ValidationError as error:
            print_calls([response], options.model)
            print_times([timing])
            sys.exit(f"askkit: {invoice_error(error)}")
        problems = invoice_faults(invoice)
        if problems:
            # A misread number or an invoice that is wrong itself: the
            # check cannot tell which, so a person reads the PDF.
            print_calls([response], options.model)
            print_times([timing])
            sys.exit(
                f"askkit: {'; '.join(problems)}; a person checks the invoice"
            )
        print(invoice.model_dump_json(indent=2))
        print()
        print_calls([response], options.model)
        print_times([timing])
        return

    if options.command == "ask":
        system = ASK_SYSTEM
        if options.system_file is not None:
            system = options.system_file.read_text(encoding="utf-8").strip()
        documents: dict[str, str] = {}
        for path in options.document:
            # Read as text, a PDF stops with a UnicodeDecodeError on its
            # first bytes that are not UTF-8: say so before that happens.
            if is_pdf(path.read_bytes()):
                sys.exit(
                    f"askkit: {path} is a PDF, and --document takes text;"
                    " read an invoice with askkit read"
                )
            documents[str(path)] = path.read_text(encoding="utf-8")
        question = options.question
        content: Content = question
        if documents:
            question = with_documents(options.question, documents)
            content = document_question(options.question, documents)
        if options.image:
            if options.history is not None:
                # A saved message holds text, and an image is not text.
                sys.exit(
                    "askkit: --history keeps text only; ask about an image"
                    " without --history"
                )
            try:
                content = with_images(question, options.image)
            except OSError as error:
                # A missing file, or one AskKit may not open: its name and
                # the reason, as the system words it.
                reason = f"{error.filename}: {error.strerror}"
                sys.exit(f"askkit: cannot read {reason}")
            except ValueError as error:
                sys.exit(f"askkit: {error}")
        history: list[SavedMessage] = []
        if options.history is not None:
            history = require_history(options.history)
        permissions = permissions_for(options.confirm, options.deny)
        sites = tuple(options.allow_site)
        if not sites:
            # With no site allowed, every fetch would be refused, so the
            # model is not offered the tool at all.
            permissions["fetch_page"] = "deny"
        client_tools = offered(TOOLS, permissions)
        offered_names = [tool["name"] for tool in client_tools]
        tools: list[ToolUnionParam] = list(client_tools)
        if options.search_site:
            # Anthropic runs the search, so no check of AskKit's sees the
            # request: naming the sites is your permission, given ahead.
            tools.append(web_search(tuple(options.search_site)))
        if options.use_tool is not None and (
            options.use_tool not in offered_names
        ):
            # The API refuses a choice that names a tool the call does not
            # offer, so AskKit says why before anything is sent.
            sys.exit(
                f"askkit: --use-tool {options.use_tool} names a tool that is"
                " not offered; take it out of --deny, or, for fetch_page,"
                " allow a site with --allow-site"
            )
        client = Anthropic(api_key=require_api_key())
        try:
            answered = tool_loop(
                client,
                content,
                options.model,
                options.max_tokens,
                system,
                options.effort,
                as_messages(history),
                tools or None,
                permissions,
                confirm_on_terminal,
                sites,
                options.use_tool,
            )
        except APIError as error:
            sys.exit(f"askkit: {error_message(error)}")
        timings = answered.timings
        if answered.cancelled:
            # Cancelled with Ctrl-C: the last call has no stop reason and
            # no usage, so its tokens are not counted; its time is shown.
            if timings[-1].first_token is not None:
                print()
            if answered.replies:
                print_calls(answered.replies, options.model, False)
            print_times(timings[:-1])
            print(
                f"Time:          {timing_line(timings[-1], cancelled=True)}"
            )
            sys.exit(f"askkit: {CANCELLED}")
        responses = answered.replies
        reply = responses[-1]
        text = final_text(responses)
        if text:
            # A blank line between the answer and its numbers.
            print()
        for url in cited_urls(responses):
            print(f"Source:        {url}")
        print_calls(responses, options.model)
        requests = requests_line(responses)
        if requests is not None:
            # Requests asked for in one turn share a round, and every
            # round is one more call: hence one Time: line per call below.
            print(f"Tool requests: {requests}")
        print_times(timings)
        warning = stop_warning(reply)
        if reply.stop_reason == "tool_use":
            # Still asking, although the last call carried tool_choice
            # "none": the API did not hold the model to it.
            warning = TOO_MANY_ROUNDS
        if warning is not None:
            if reply.stop_reason == "refusal" and text:
                print(f"askkit: {DECLINED_PART}", file=sys.stderr)
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
        # Saved only once the answer passed every check above, so the
        # file never holds a cut-off or declined answer.
        if options.history is not None:
            history.append(SavedMessage(role="user", content=question))
            history.append(SavedMessage(role="assistant", content=text))
            save_history(options.history, history)
            print(
                f"History:       {len(history)} messages in {options.history}"
            )
        return

    if options.command == "chat":
        fault = budget_fault(
            options.model, options.budget, options.max_tokens
        )
        if fault is not None:
            sys.exit(f"askkit: {fault}")
        system = ASK_SYSTEM
        if options.system_file is not None:
            system = options.system_file.read_text(encoding="utf-8").strip()
        history = require_history(options.history)
        notes = require_memory(options.memory)
        client = Anthropic(api_key=require_api_key())
        print(f"History:       {len(history)} messages in {options.history}")
        print(f"Budget:        {options.budget:,} input tokens a turn")
        print(f"Old turns:     {options.old_turns}")
        unit = "notes"
        if len(notes) == 1:
            unit = "note"
        print(f"Memory:        {len(notes)} {unit} in {options.memory}")
        print(f"Type {STOP} to stop.")
        print()
        try:
            chat(
                client,
                options.history,
                history,
                options.memory,
                notes,
                options.model,
                options.max_tokens,
                system,
                options.effort,
                options.budget,
                options.old_turns,
                input,
            )
        except KeyboardInterrupt:
            # Ctrl-C stops the chat at once. Every whole turn is in the
            # file already; only an answer still on its way is lost.
            print()
        print(f"History:       {len(history)} messages in {options.history}")
        return

    if options.command == "compare":
        text = options.file.read_text(encoding="utf-8")
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
    window = LIMITS[options.model]["context_window"]
    max_input = LIMITS[options.model]["max_input"]
    max_output = LIMITS[options.model]["max_output"]
    data = options.file.read_bytes()
    what: str | None = None
    if is_heic(data):
        # Checked before the recordings: it starts the way an M4A does.
        sys.exit(
            f"askkit: {options.file} is a HEIC photo, which Claude's API"
            " does not read; save it as JPEG"
        )
    if audio_format(data) is not None:
        # AskKit sends the model the transcript, never the sound, so the
        # recording has a price at OpenAI and no tokens until it is written
        # down. The transcript goes to the model given with -m, whichever
        # provider runs it.
        fault = audio_fault(options.file, data)
        if fault is not None:
            sys.exit(f"askkit: {fault}")
        print(f"Model:         {options.model}")
        recording = f"{audio_format(data)} recording"
        seconds = audio_seconds(data)
        if seconds is None:
            print(f"Input:         {recording}, length unknown")
            print("Transcribe:    unknown: only a WAV gives its length")
        else:
            cost = transcription_cost(seconds)
            print(f"Input:         {recording}, {seconds:.1f} s")
            print(f"Transcribe:    ${cost:.6f} at OpenAI, {TRANSCRIBE_MODEL}")
        print("Model reads:   the transcript; estimate it once it is written")
        return
    if media_type_of(data) is not None or is_pdf(data):
        if not options.model.startswith("claude-"):
            sys.exit(
                "askkit: AskKit counts image and PDF tokens for Claude only;"
                " each provider counts them its own way"
            )
        try:
            if is_pdf(data):
                media = pdf_input(options.file, data, options.model, window)
            else:
                media = image_input(options.file, data)
        except ValueError as error:
            sys.exit(f"askkit: {error}")
        what, input_tokens, how = media.what, media.tokens, media.how
    else:
        text = data.decode("utf-8")
        input_tokens, exact = count_tokens(options.model, text)
        if exact:
            how = "counted by the model's tokenizer"
        else:
            how = f"estimated from {len(text):,} characters"
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
    print(f"Model:         {options.model}")
    if what is not None:
        print(f"Input:         {what}")
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
