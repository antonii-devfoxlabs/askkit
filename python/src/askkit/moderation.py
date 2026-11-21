from openai import OpenAI
from openai.types import Moderation

from askkit.redact import redacted

# OpenAI's moderation model, from its moderation guide (checked on 21
# November 2026). Anthropic's API has no moderation endpoint, so this
# check runs at OpenAI, as transcribe does. The endpoint is free, and
# OpenAI upgrades the model behind this name from time to time.
MODERATION_MODEL = "omni-moderation-latest"

# How many emails one moderation request carries. The API reference
# names no limit (checked on 21 November 2026; it may set one); small
# requests keep a failed one cheap to send again.
PER_REQUEST = 20


def flagged_categories(result: Moderation) -> list[str]:
    # The categories the endpoint flagged for one text, under the names
    # OpenAI writes in its guide ("harassment/threatening"). by_alias
    # gives those names: the SDK's own fields are harassment_threatening.
    if not result.flagged:
        return []
    flags = result.categories.model_dump(by_alias=True)
    names = [name for name, flagged in flags.items() if flagged]
    # flagged with no category true is not in the reference, but a text
    # flagged must never read as one let through.
    return names or ["a category it did not name"]


def moderate(client: OpenAI, texts: list[str]) -> list[list[str]]:
    # One list of flagged categories for each text, in the order of the
    # texts; an empty list means nothing was flagged. Card numbers, email
    # addresses and phone numbers are taken out first (lesson 12.4): the
    # check needs what a text says, not how to reach the person who
    # wrote it. An API error is not caught here: the caller stops, so no
    # unchecked text goes on to the model.
    flags: list[list[str]] = []
    for start in range(0, len(texts), PER_REQUEST):
        part = texts[start : start + PER_REQUEST]
        response = client.moderations.create(
            model=MODERATION_MODEL, input=[redacted(text) for text in part]
        )
        flags.extend(
            flagged_categories(result) for result in response.results
        )
    return flags


def person_reason(categories: list[str]) -> str:
    # What a person who reads a flagged email is told, after "to a person:".
    return "flagged " + ", ".join(categories)
