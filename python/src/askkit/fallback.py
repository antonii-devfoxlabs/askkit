import time
from collections.abc import Callable
from dataclasses import dataclass

import anthropic
from anthropic.types import Message
from any_llm.exceptions import AnyLLMError

from askkit.retry import reason, worth_retrying

# Seconds a model that used up its tries is passed over in a run, so the
# questions after it do not each wait up to two minutes for a model that
# has just failed. After that it is asked again first.
COOL_DOWN = 60.0

# Refusal categories another model is not meant to answer. Of
# reasoning_extraction Anthropic's "Refusals and fallback" page says:
# "This category has no recommended fallback model, so change the prompt
# rather than retry the request" (checked on 2026-11-20).
NO_FALLBACK_CATEGORIES = ("reasoning_extraction",)


@dataclass(frozen=True)
class Handover:
    # One model a question left for the next one: which, and why ("529",
    # "timeout", "no connection", "refusal", or "cooling down").
    model: str
    reason: str


@dataclass
class FirstAnswer:
    # The response that answers, the model of the chain that gave it,
    # every model the question left on the way, and the declined
    # responses before it, which may be billed too.
    response: Message
    model: str
    handovers: list[Handover]
    declined: list[tuple[Message, str]]


def is_claude(model: str) -> bool:
    return model.startswith("claude-")


def declined_onward(response: Message, next_model: str) -> bool:
    # A declined answer goes on only to another Claude model, as
    # Anthropic's page advises: "You can usually still get an answer by
    # sending the same request to another Claude model." A decline is a
    # judgement about the request, not an outage, and sending it to
    # another company to get round it fixes nothing.
    if response.stop_reason != "refusal" or not is_claude(next_model):
        return False
    details = response.stop_details
    if details is not None and details.category in NO_FALLBACK_CATEGORIES:
        return False
    return True


def first_answer(
    chain: list[str],
    send: Callable[[str], Message],
    down_until: dict[str, float],
    clock: Callable[[], float] = time.monotonic,
    cool_down: float = COOL_DOWN,
) -> FirstAnswer:
    # Sends the question to the models of the chain in order until one
    # answers. A model hands the question on when send() raises an error
    # another model can fix, which send() raises only once its own tries
    # are used up, or when it declines and the next model is Claude. A
    # model that failed is passed over until down_until says it may be
    # asked again; the last model is always asked. The last model's
    # error is raised as it is.
    handovers: list[Handover] = []
    declined: list[tuple[Message, str]] = []
    last = len(chain) - 1
    for position, model in enumerate(chain):
        if position < last and clock() < down_until.get(model, 0.0):
            handovers.append(Handover(model, "cooling down"))
            continue
        try:
            response = send(model)
        except (anthropic.APIError, AnyLLMError) as error:
            # Only what another model can fix hands the question on: the
            # failures worth another try (lesson 13.5 keeps one rule for
            # both). A request the API rejected (400, 413) or a key it
            # refused (401, 403) fails the same way on the next model,
            # and falling back would only hide it.
            if position == last or not worth_retrying(error):
                raise
            down_until[model] = clock() + cool_down
            handovers.append(Handover(model, reason(error)))
            continue
        if position < last and declined_onward(response, chain[position + 1]):
            declined.append((response, model))
            handovers.append(Handover(model, "refusal"))
            continue
        return FirstAnswer(response, model, handovers, declined)
    # Not reached: the last model either returns or raises.
    raise AssertionError("the chain is empty")


def fallbacks_line(answered: list[tuple[str, FirstAnswer]]) -> str | None:
    # One line for the run: how many questions another model answered,
    # and why the models before it handed them on.
    moved: list[str] = []
    for case, outcome in answered:
        if outcome.handovers:
            reasons = ", ".join(
                handover.reason for handover in outcome.handovers
            )
            moved.append(f"{case}: {reasons}")
    if not moved:
        return None
    return f"Fallbacks:     {len(moved)} ({'; '.join(moved)})"
