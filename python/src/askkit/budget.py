from anthropic import Anthropic
from anthropic.types import MessageParam

from askkit.ask import messages_for
from askkit.limits import LIMITS

# The most input tokens one call of askkit chat sends unless you give
# another --budget. Claude Haiku 5.5 bills every token of a call whose
# prompt is over 100,000 tokens at five times the price (Anthropic's
# pricing page, checked on 2026-11-17), and models read long input less
# well. The count before a call is an estimate, so the default stays
# under that line with room to spare.
DEFAULT_BUDGET = 90_000


def budget_fault(model: str, budget: int, max_tokens: int) -> str | None:
    # Checked once, when the chat starts: a budget the window cannot hold
    # would let a call through that the API then rejects or cuts off.
    if budget < 1:
        return f"--budget {budget} leaves no room for a question"
    window = LIMITS[model]["context_window"]
    if budget + max_tokens > window:
        return (
            f"--budget {budget:,} and --max-tokens {max_tokens:,} come to"
            f" {budget + max_tokens:,} tokens, more than the"
            f" {window:,} of {model}'s window"
        )
    return None


def input_tokens(
    client: Anthropic,
    model: str,
    system: str,
    question: str,
    history: list[MessageParam],
) -> int:
    # The same system prompt and messages the call will send, counted by
    # the API without a call: free, and an estimate close to the bill.
    count = client.messages.count_tokens(
        model=model,
        system=system,
        messages=messages_for(question, history),
    )
    return count.input_tokens


def over_budget(needed: int, budget: int) -> str | None:
    if needed <= budget:
        return None
    return (
        f"this turn would send {needed:,} tokens,"
        f" over the chat's budget of {budget:,}"
    )
