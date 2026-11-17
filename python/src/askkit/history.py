import json
from pathlib import Path
from typing import Literal

from anthropic.types import MessageParam
from pydantic import BaseModel, ConfigDict

# The API keeps nothing between two calls, so the conversation is
# AskKit's to keep: a file of messages, read before a call and written
# after a whole answer.


class SavedMessage(BaseModel):
    # One message of a saved conversation: who wrote it, and its text.
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str


def load_history(path: Path) -> list[SavedMessage]:
    # No file yet is a conversation that has not started.
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path} holds no list of messages")
    saved: list[SavedMessage] = []
    for item in data:
        saved.append(SavedMessage.model_validate(item))
    return saved


def save_history(path: Path, saved: list[SavedMessage]) -> None:
    data = []
    for message in saved:
        data.append(message.model_dump())
    text = json.dumps(data, indent=2, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def history_fault(saved: list[SavedMessage]) -> str | None:
    # AskKit writes a question, then its answer, every time. The API
    # would merge two user messages in a row, but in this file they mean
    # a hand edit or a lost answer, so the order is checked before a
    # call is paid for.
    expected = "user"
    number = 1
    for message in saved:
        if message.role != expected:
            return (
                f"message {number} should be the {expected}'s,"
                f" not the {message.role}'s"
            )
        if expected == "user":
            expected = "assistant"
        else:
            expected = "user"
        number += 1
    if saved and saved[-1].role != "assistant":
        return "the last message should be the assistant's answer"
    return None


def as_messages(saved: list[SavedMessage]) -> list[MessageParam]:
    messages: list[MessageParam] = []
    for message in saved:
        # Written out per role: the SDK's types tell the two apart.
        if message.role == "user":
            messages.append({"role": "user", "content": message.content})
        else:
            messages.append({"role": "assistant", "content": message.content})
    return messages
