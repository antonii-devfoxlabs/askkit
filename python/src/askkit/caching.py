from collections.abc import Sequence
from typing import cast

from anthropic.types import (
    CacheControlEphemeralParam,
    ContentBlockParam,
    TextBlockParam,
)

# A cache breakpoint. The API keeps the request up to and including the
# block that carries it for 5 minutes, and every read starts the minutes
# again. Its order is the tools, then the system prompt, then the
# messages, so a breakpoint keeps everything before it in that order.
CACHE: CacheControlEphemeralParam = {"type": "ephemeral"}


def cached_system(system: str) -> list[TextBlockParam]:
    # The tools come before the system prompt, so one breakpoint on it
    # keeps both: the opening every call of every ask sends again.
    return [{"type": "text", "text": system, "cache_control": CACHE}]


def cached(
    question: str | Sequence[ContentBlockParam],
) -> list[ContentBlockParam]:
    # A breakpoint on the question's last block, so every later call of a
    # tool loop reads the first call's whole input from the cache.
    if isinstance(question, str):
        return [{"type": "text", "text": question, "cache_control": CACHE}]
    blocks = list(question)
    last = cast(ContentBlockParam, {**blocks[-1], "cache_control": CACHE})
    return blocks[:-1] + [last]
