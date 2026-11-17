import sys
from collections.abc import Callable
from pathlib import Path

from anthropic import Anthropic, APIError

from askkit.ask import Effort, answer_text, ask, stop_warning
from askkit.cost import call_cost
from askkit.errors import error_message
from askkit.history import SavedMessage, as_messages, save_history

# What you type to end a chat. The end of input ends it too: Ctrl-D on
# macOS and Linux, Ctrl-Z then Enter on Windows.
STOP = "/exit"

NOT_KEPT = "Not kept: ask again, or type /exit to stop."


def chat(
    client: Anthropic,
    path: Path,
    history: list[SavedMessage],
    model: str,
    max_tokens: int,
    system: str,
    effort: Effort | None,
    read: Callable[[str], str],
) -> None:
    # One question a turn until you stop. The whole conversation goes with
    # every question, and a turn joins the history, in the list and in
    # the file, only once its answer is whole.
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
            response = ask(
                client,
                question,
                model,
                max_tokens,
                system,
                effort,
                as_messages(history),
            )
        except APIError as error:
            # A failed call changes nothing, so the chat goes on and the
            # same question can simply be asked again.
            print(f"askkit: {error_message(error)}", file=sys.stderr)
            print(NOT_KEPT)
            print()
            continue
        text = answer_text(response)
        # Anthropic advises discarding the part of a declined answer.
        if response.stop_reason != "refusal":
            print(text)
        warning = stop_warning(response)
        if warning is not None:
            print(f"askkit: {warning}", file=sys.stderr)
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
        print(
            f"{label:<15}{usage.input_tokens:,} in,"
            f" {usage.output_tokens:,} out, ${cost:.6f}"
        )
        print()
