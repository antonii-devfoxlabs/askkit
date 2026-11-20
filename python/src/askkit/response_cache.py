import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from anthropic.types import Message

# Where AskKit keeps the answers it has paid for: one file per request,
# in a folder of the folder you run it from. .gitignore leaves it out,
# since a kept answer can hold what a customer wrote.
CACHE_FOLDER = Path(".askkit-cache")

# How long a kept answer is given back. The key changes when anything
# sent changes, but not when the serving behind a model ID does (lesson
# 10.6), and a kept answer can hold a customer's details: after 30 days
# the answer is asked for again, and the new one replaces it.
MAX_AGE = timedelta(days=30)


# What reading a kept answer back can raise. OSError: no such file (never
# stored) or one AskKit cannot open. ValueError: not JSON, or not a
# Message (pydantic's ValidationError is a ValueError). KeyError and
# TypeError: JSON of another shape.
UNREADABLE = (OSError, ValueError, KeyError, TypeError)


@dataclass
class Kept:
    # An answer from the cache, and when it was stored.
    response: Message
    stored: datetime


def request_key(request: dict[str, Any]) -> str:
    # The request written out the same way every time (sort_keys puts
    # the fields in one order), then hashed. Two requests get the same
    # key only when every field is the same, the model ID included.
    text = json.dumps(request, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def kept_answer(folder: Path, key: str, now: datetime) -> Kept | None:
    # The answer stored under key, or None: never stored, older than
    # MAX_AGE, or a file AskKit cannot read back (cut short when a run
    # stopped while writing it). None means the model is asked again.
    path = folder / f"{key}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        stored = datetime.fromisoformat(data["stored"])
        response = Message.model_validate(data["response"])
    except UNREADABLE:
        return None
    if now - stored > MAX_AGE:
        return None
    return Kept(response, stored)


def keep_answer(
    folder: Path, key: str, response: Message, now: datetime
) -> None:
    # Stores an answer that passed every check, replacing an older one.
    # The request itself is not stored, only its key.
    folder.mkdir(exist_ok=True)
    data = {"stored": now.isoformat(), "response": response.to_dict()}
    path = folder / f"{key}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
