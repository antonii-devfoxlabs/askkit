import sys
from collections.abc import Callable
from pathlib import Path

from anthropic import Anthropic, APIError

from askkit.ask import (
    CANCELLED,
    DECLINED_PART,
    FINISHED,
    Effort,
    answer_text,
    stop_warning,
    stream_answer,
    timing_line,
)
from askkit.budget import input_tokens, over_budget
from askkit.cost import call_cost
from askkit.errors import error_message
from askkit.history import SavedMessage, as_messages, save_history
from askkit.memory import memory_reply, with_memory
from askkit.trim import (
    KEEP_TURNS,
    OldTurns,
    summarise,
    turns_label,
    with_summary,
)

# What you type to end a chat. The end of input ends it too: Ctrl-D on
# macOS and Linux, Ctrl-Z then Enter on Windows.
STOP = "/exit"

NOT_KEPT = "Not kept: ask again, or type /exit to stop."

NOT_SENT = (
    "Not sent: try a shorter question, a new --history file or a larger"
    " --budget."
)


def turn_tokens(
    client: Anthropic,
    model: str,
    system: str,
    question: str,
    sent_history: list[SavedMessage],
) -> int:
    messages = as_messages(sent_history)
    return input_tokens(client, model, system, question, messages)


def chat(
    client: Anthropic,
    path: Path,
    history: list[SavedMessage],
    memory_path: Path,
    notes: list[str],
    model: str,
    max_tokens: int,
    system: str,
    effort: Effort | None,
    budget: int,
    old_turns: OldTurns,
    read: Callable[[str], str],
) -> None:
    # One question a turn until you stop. A turn joins the history, in the
    # list and in the file, only once its answer is whole. The file keeps
    # every turn; the calls send history[start:], with the summary of the
    # turns before start in the system prompt. The notes in memory go
    # with every call, and outlast the chat in a file of their own.
    start = 0
    summary = ""
    while True:
        try:
            question = read("You: ").strip()
        except EOFError:
            print()
            return
        if question == STOP:
            return
        if not question:
            continue
        try:
            reply = memory_reply(question, notes, memory_path)
        except ValueError as error:
            print(f"askkit: {error}", file=sys.stderr)
            print()
            continue
        if reply is not None:
            # A memory command is AskKit's to answer: no count, no call.
            print(reply)
            print()
            continue
        # Built every turn: a note remembered now is sent from the next
        # question on.
        system_with_notes = with_memory(system, notes)
        first = start
        try:
            # Counted before the call: the count is free, and a turn over
            # the budget is never sent, so it is never paid for.
            sent_system = with_summary(system_with_notes, summary)
            needed = turn_tokens(
                client, model, sent_system, question, history[start:]
            )
            keep = max(start, len(history) - 2 * KEEP_TURNS)
            if needed > budget and old_turns == "summarise" and keep > start:
                # Everything but the last turns goes into one summary: at a
                # real budget, many turns then fit before the next fold.
                folded = summarise(
                    client, model, summary, history[start:keep]
                )
                if folded.stop_reason not in FINISHED:
                    # A cut-off or declined summary would lose facts, so
                    # the turn waits and nothing changes.
                    reason = folded.stop_reason
                    print(
                        f"askkit: the summary stopped early: {reason}",
                        file=sys.stderr,
                    )
                    print(NOT_KEPT)
                    print()
                    continue
                summary = answer_text(folded)
                start = keep
                cost = call_cost(
                    model,
                    folded.usage.input_tokens,
                    folded.usage.output_tokens,
                )
                label = turns_label(first // 2 + 1, start // 2)
                print(f"{'Summarised:':<15}{label}, ${cost:.6f}")
                sent_system = with_summary(system_with_notes, summary)
                needed = turn_tokens(
                    client, model, sent_system, question, history[start:]
                )
            while needed > budget and old_turns == "drop" and keep > start:
                # One turn at a time, the oldest first: counting is free,
                # so no more is dropped than the budget needs.
                start += 2
                needed = turn_tokens(
                    client, model, sent_system, question, history[start:]
                )
            if old_turns == "drop" and start > first:
                label = turns_label(first // 2 + 1, start // 2)
                print(f"{'Dropped:':<15}{label}")
        except APIError as error:
            print(f"askkit: {error_message(error)}", file=sys.stderr)
            print(NOT_KEPT)
            print()
            continue
        problem = over_budget(needed, budget)
        if problem is not None:
            # Even the last turns and the question do not fit: only a
            # shorter question, a new file or a larger budget helps.
            print(f"askkit: {problem}", file=sys.stderr)
            print(NOT_SENT)
            print()
            continue
        sent = as_messages(history[start:])
        try:
            # The answer shows as it is written; the turn is kept only once
            # the stream has ended with a whole answer.
            response, timing = stream_answer(
                client, question, model, max_tokens, sent_system, effort, sent
            )
        except APIError as error:
            # A failed call changes nothing, so the chat goes on and the
            # same question can simply be asked again.
            print(f"askkit: {error_message(error)}", file=sys.stderr)
            print(NOT_KEPT)
            print()
            continue
        if response is None:
            # Ctrl-C during the answer cancels only this answer: the chat
            # goes on, and Ctrl-C at the "You: " prompt still ends it.
            print(f"askkit: {CANCELLED}", file=sys.stderr)
            print(f"{'Time:':<15}{timing_line(timing, cancelled=True)}")
            print(NOT_KEPT)
            print()
            continue
        text = answer_text(response)
        warning = stop_warning(response)
        if warning is not None:
            if response.stop_reason == "refusal" and text:
                print(f"askkit: {DECLINED_PART}", file=sys.stderr)
            print(f"askkit: {warning}", file=sys.stderr)
            # Not kept, but the wait was real, so its times are shown too.
            print(f"{'Time:':<15}{timing_line(timing)}")
            print(NOT_KEPT)
            print()
            continue
        history.append(SavedMessage(role="user", content=question))
        history.append(SavedMessage(role="assistant", content=text))
        # Saved after every whole answer, so stopping at any moment loses
        # at most the turn that was still on its way.
        save_history(path, history)
        usage = response.usage
        cost = call_cost(model, usage.input_tokens, usage.output_tokens)
        label = f"Turn {len(history) // 2}:"
        share = usage.input_tokens / budget
        print(
            f"{label:<15}{usage.input_tokens:,} in ({share:.0%} of budget),"
            f" {usage.output_tokens:,} out, ${cost:.6f}"
        )
        print(f"{'Time:':<15}{timing_line(timing)}")
        print()
