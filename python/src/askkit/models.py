from dataclasses import dataclass
from datetime import date

from anthropic import Anthropic
from openai import OpenAI

# The model every command calls unless -m names another. Each ID below
# is a pinned snapshot (lesson 1.7): the same model every day until the
# provider retires it. Moving AskKit to another model is this one line,
# once the eval says the new model is as good.
DEFAULT_MODEL = "claude-haiku-5-5"

# When the dates below were last read, each on its provider's own
# deprecations page:
# Anthropic: https://platform.claude.com/docs/en/about-claude/model-deprecations
# OpenAI: https://developers.openai.com/api/docs/deprecations
# Google (gemini-3.5-flash-lite, checked on 2026-11-21, "No shutdown
# date announced"): https://ai.google.dev/gemini-api/docs/deprecations
CHECKED = date(2026, 11, 20)


@dataclass(frozen=True)
class Lifecycle:
    # provider: whose servers the model runs on.
    # retires: the day requests to the model start to fail, or None when
    # the provider gives no date. The provider can still move it until
    # then; askkit models --live says when it has.
    # announced: True once the provider has set that day. False means
    # retires is only the earliest day it could be ("not sooner than").
    # replacement: the model the provider recommends instead, once it
    # names one.
    provider: str
    retires: date | None
    announced: bool
    replacement: str | None


# Every model AskKit calls, never an alias. A model added to any command
# needs a row here first (a test enforces it).
LIFECYCLES: dict[str, Lifecycle] = {
    "claude-haiku-5-5": Lifecycle(
        "Anthropic", date(2027, 10, 7), False, None
    ),
    "claude-sonnet-5-5": Lifecycle(
        "Anthropic", date(2027, 9, 28), False, None
    ),
    "gemini-3.5-flash-lite": Lifecycle("Google", None, False, None),
    "gpt-6-luna": Lifecycle("OpenAI", None, False, None),
    "gpt-transcribe": Lifecycle("OpenAI", None, False, None),
}


def instead(lifecycle: Lifecycle) -> str:
    if lifecycle.replacement is None:
        return ""
    return f"; move to {lifecycle.replacement}"


def retirement_fault(model: str, today: date) -> str | None:
    # A model whose announced day has come: every request to it fails,
    # so the command stops before the first one.
    lifecycle = LIFECYCLES[model]
    if not lifecycle.announced or lifecycle.retires is None:
        return None
    if today < lifecycle.retires:
        return None
    return f"{model} was retired on {lifecycle.retires}{instead(lifecycle)}"


def retirement_warning(model: str, today: date) -> str | None:
    # A model with an announced day still ahead: the command runs and
    # names the day on every run, from the day the date is entered.
    lifecycle = LIFECYCLES[model]
    if not lifecycle.announced or lifecycle.retires is None:
        return None
    days = (lifecycle.retires - today).days
    if days <= 0:
        return None
    unit = "day" if days == 1 else "days"
    return (
        f"{model} retires on {lifecycle.retires}, in {days} {unit}"
        f"{instead(lifecycle)}"
    )


def retires_text(lifecycle: Lifecycle) -> str:
    if lifecycle.retires is None and lifecycle.announced:
        # Deprecated, with the day still to come.
        return "retiring, day not given yet"
    if lifecycle.retires is None:
        return "no date announced"
    if lifecycle.announced:
        return f"{lifecycle.retires}"
    return f"not before {lifecycle.retires}"


def table_lines() -> list[str]:
    # The model column is as wide as its longest ID and two spaces, so a
    # longer ID never runs into its provider.
    width = max(len(model) for model in LIFECYCLES) + 2
    lines = [f"{'Model':<{width}}{'Provider':<11}Retires"]
    for model, lifecycle in LIFECYCLES.items():
        lines.append(
            f"{model:<{width}}{lifecycle.provider:<11}"
            f"{retires_text(lifecycle)}"
        )
    return lines


def claude_lifecycle(client: Anthropic, model: str) -> Lifecycle:
    # What Anthropic's Models API says today. lifecycle is "active",
    # "deprecated" or "retired"; retires_at is set once a retirement is
    # scheduled.
    info = client.models.retrieve(model)
    retires = None if info.retires_at is None else info.retires_at.date()
    return Lifecycle("Anthropic", retires, info.lifecycle != "active", None)


def openai_lifecycle(client: OpenAI, model: str) -> Lifecycle:
    # OpenAI's models endpoint names a shutdown_date once it announces
    # one, such as "2027-01-14", and None before.
    info = client.models.retrieve(model)
    if info.shutdown_date is None:
        return Lifecycle("OpenAI", None, False, None)
    return Lifecycle(
        "OpenAI", date.fromisoformat(info.shutdown_date), True, None
    )


def lifecycle_change(
    model: str, pinned: Lifecycle, live: Lifecycle
) -> str | None:
    # The table is what AskKit acts on; the provider's answer is checked
    # against it. Only an announced day counts: before one, Anthropic
    # gives no date at all and the table keeps the earliest one.
    if live.announced == pinned.announced and (
        not live.announced or live.retires == pinned.retires
    ):
        return None
    return (
        f"{model}: {live.provider} says {retires_text(live)}, and"
        f" models.py says {retires_text(pinned)}; update its row"
    )
