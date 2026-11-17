from pathlib import Path

from pydantic import BaseModel, ConfigDict

from askkit.prompts import render

# What AskKit keeps about you from one chat to the next: short notes you
# ask it to remember, in a file of their own. Every chat sends them in
# its system prompt, so a new conversation starts out knowing them.

# Typed in a chat, these show or change the notes. They are never sent
# to the model.
REMEMBER = "/remember"
FORGET = "/forget"
SHOW = "/memory"

# The notes go with every call, so they stay few and short.
MAX_NOTES = 20
MAX_NOTE_CHARACTERS = 200

# An Anthropic API key starts like this (lesson 2.1). A note is sent with
# every call and kept on disk as plain text, so a key never goes in.
KEY_START = "sk-ant-"


class Memory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    notes: list[str]


def load_memory(path: Path) -> list[str]:
    # No file yet is a memory with nothing in it.
    if not path.exists():
        return []
    return Memory.model_validate_json(path.read_text(encoding="utf-8")).notes


def save_memory(path: Path, notes: list[str]) -> None:
    text = Memory(notes=notes).model_dump_json(indent=2)
    path.write_text(text + "\n", encoding="utf-8")


def note_fault(note: str, notes: list[str]) -> str | None:
    if not note:
        return (
            f"write the note after {REMEMBER}, such as: {REMEMBER} be brief"
        )
    if len(note) > MAX_NOTE_CHARACTERS:
        return (
            f"a note holds up to {MAX_NOTE_CHARACTERS} characters,"
            f" and this one has {len(note)}"
        )
    if KEY_START in note:
        return (
            "that note holds an API key; notes go with every call,"
            " so keep keys out of them"
        )
    if note in notes:
        return "that note is in memory already"
    if len(notes) >= MAX_NOTES:
        return f"memory is full at {MAX_NOTES} notes; {FORGET} one first"
    return None


def is_command(line: str, command: str) -> bool:
    # "/remember be brief" is the command; "/remembering" is a question.
    return line == command or line.startswith(command + " ")


def memory_reply(line: str, notes: list[str], path: Path) -> str | None:
    # A memory command shows or changes the notes, saves a change at once
    # and says what it did. Any other line is a question for the model,
    # and gets None. A command that cannot be done raises ValueError.
    if line == SHOW:
        if not notes:
            return "Memory is empty."
        lines = []
        number = 1
        for note in notes:
            lines.append(f"{number}. {note}")
            number += 1
        return "\n".join(lines)
    if is_command(line, REMEMBER):
        note = line.removeprefix(REMEMBER).strip()
        fault = note_fault(note, notes)
        if fault is not None:
            raise ValueError(fault)
        notes.append(note)
        save_memory(path, notes)
        return f"{'Remembered:':<15}note {len(notes)}"
    if is_command(line, FORGET):
        chosen = line.removeprefix(FORGET).strip()
        if not notes:
            raise ValueError("memory is empty: there is nothing to forget")
        if not chosen.isdigit() or not 1 <= int(chosen) <= len(notes):
            raise ValueError(
                f"give the number of a note from {SHOW}, 1 to {len(notes)}"
            )
        note = notes.pop(int(chosen) - 1)
        save_memory(path, notes)
        return f"{'Forgotten:':<15}{note}"
    return None


def with_memory(system: str, notes: list[str]) -> str:
    # The notes hold for every chat, as the rules do, so they go in the
    # system prompt, after the rules.
    if not notes:
        return system
    lines = []
    for note in notes:
        lines.append(f"- {note}")
    return render("remembered", {"system": system, "notes": "\n".join(lines)})
