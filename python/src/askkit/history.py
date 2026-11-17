import json
from pathlib import Path
from typing import Literal

from anthropic.types import MessageParam
from pydantic import BaseModel, ConfigDict

# The API keeps nothing between two calls, so the conversation is
# AskKit's to keep: a file of messages, read before a call and written
# after a whole answer.


class Turn(BaseModel):
    # One message of a saved conversation: who wrote it, and its text.
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


def load_history(path: Path) -> list[Turn]:
    # No file yet is a conversation that has not started.
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    turns: list[Turn] = []
    for item in data:
        turns.append(Turn.model_validate(item))
    return turns


def save_history(path: Path, turns: list[Turn]) -> None:
    data = []
    for turn in turns:
        data.append(turn.model_dump())
    text = json.dumps(data, indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def history_fault(turns: list[Turn]) -> str | None:
    # The API reads the list as turns: the user's first, then one each.
    # A file edited by hand can break that, so it is checked before a
    # call is paid for.
    expected = "user"
    number = 1
    for turn in turns:
        if turn.role != expected:
            return (
                f"message {number} should be the {expected}'s,"
                f" not the {turn.role}'s"
            )
        if expected == "user":
            expected = "assistant"
        else:
            expected = "user"
        number += 1
    if turns and turns[-1].role != "assistant":
        return "the last message should be the assistant's answer"
    return None


def as_messages(turns: list[Turn]) -> list[MessageParam]:
    messages: list[MessageParam] = []
    for turn in turns:
        # Written out per role: the SDK's types tell the two apart.
        if turn.role == "user":
            messages.append({"role": "user", "content": turn.content})
        else:
            messages.append({"role": "assistant", "content": turn.content})
    return messages
