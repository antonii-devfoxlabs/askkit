import json
from pathlib import Path

import pytest

from askkit.memory import (
    MAX_NOTE_CHARACTERS,
    MAX_NOTES,
    load_memory,
    memory_reply,
    note_fault,
    save_memory,
    with_memory,
)
from askkit.prompts import render

HAIKU = "I build with Claude Haiku 5.5: compare other models with it."
BRIEF = "Answer in three sentences at most."


def test_no_file_is_an_empty_memory(tmp_path: Path) -> None:
    assert load_memory(tmp_path / "memory.json") == []


def test_notes_are_saved_as_readable_json(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    save_memory(path, [HAIKU, "Prices in euros, not dollars: €"])
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "notes": [HAIKU, "Prices in euros, not dollars: €"]
    }
    assert "€" in path.read_text(encoding="utf-8")
    assert load_memory(path) == [HAIKU, "Prices in euros, not dollars: €"]


@pytest.mark.parametrize(
    "text", ["not json", '["a list"]', '{"notes": ["a"], "extra": 1}']
)
def test_a_file_that_is_not_askkits_memory_is_refused(
    tmp_path: Path, text: str
) -> None:
    path = tmp_path / "memory.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_memory(path)


def test_a_short_new_note_is_fine() -> None:
    assert note_fault(BRIEF, [HAIKU]) is None


def test_an_empty_note_says_how_to_write_one() -> None:
    assert note_fault("", []) == (
        "write the note after /remember, such as: /remember be brief"
    )


def test_a_note_at_the_limit_is_fine_and_one_over_is_not() -> None:
    assert note_fault("a" * MAX_NOTE_CHARACTERS, []) is None
    assert note_fault("a" * (MAX_NOTE_CHARACTERS + 1), []) == (
        "a note holds up to 200 characters, and this one has 201"
    )


def test_a_note_with_an_api_key_is_refused() -> None:
    assert note_fault("My key is sk-ant-api03-abc", []) == (
        "that note holds an API key; notes go with every call,"
        " so keep keys out of them"
    )


def test_the_same_note_twice_is_refused() -> None:
    assert note_fault(HAIKU, [HAIKU]) == "that note is in memory already"


def test_a_full_memory_takes_no_more_notes() -> None:
    notes = [f"Note {number}" for number in range(MAX_NOTES)]
    assert note_fault(BRIEF, notes) == (
        "memory is full at 20 notes; /forget one first"
    )
    assert note_fault(BRIEF, notes[1:]) is None


def test_remember_adds_the_note_and_saves_it(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    notes = [HAIKU]
    reply = memory_reply(f"/remember   {BRIEF}  ", notes, path)
    assert reply == "Remembered:    note 2"
    assert notes == [HAIKU, BRIEF]
    assert load_memory(path) == [HAIKU, BRIEF]


def test_a_refused_note_changes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    notes = [HAIKU]
    with pytest.raises(ValueError, match="in memory already"):
        memory_reply(f"/remember {HAIKU}", notes, path)
    assert notes == [HAIKU]
    assert not path.exists()


def test_forget_takes_the_note_out_by_its_number(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    notes = [HAIKU, BRIEF]
    assert memory_reply("/forget 1", notes, path) == f"Forgotten:     {HAIKU}"
    assert notes == [BRIEF]
    assert load_memory(path) == [BRIEF]


@pytest.mark.parametrize(
    "line", ["/forget", "/forget 0", "/forget 3", "/forget one", "/forget ²"]
)
def test_forget_needs_the_number_of_a_note(tmp_path: Path, line: str) -> None:
    notes = [HAIKU, BRIEF]
    with pytest.raises(ValueError) as refused:
        memory_reply(line, notes, tmp_path / "memory.json")
    assert (
        str(refused.value) == "give the number of a note from /memory, 1 to 2"
    )
    assert notes == [HAIKU, BRIEF]


def test_forget_on_an_empty_memory_says_so(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nothing to forget"):
        memory_reply("/forget 1", [], tmp_path / "memory.json")


def test_memory_lists_the_notes_by_number(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    assert memory_reply("/memory", [HAIKU, BRIEF], path) == (
        f"1. {HAIKU}\n2. {BRIEF}"
    )
    assert memory_reply("/memory", [], path) == "Memory is empty."
    assert not path.exists()


@pytest.mark.parametrize(
    "line", ["What is a context window?", "/remembering", "/memory please"]
)
def test_any_other_line_is_a_question(tmp_path: Path, line: str) -> None:
    assert memory_reply(line, [HAIKU], tmp_path / "memory.json") is None


def test_no_notes_leave_the_system_prompt_as_it_was() -> None:
    assert with_memory("Be brief.", []) == "Be brief."


def test_the_notes_follow_the_rules_one_a_line() -> None:
    system = with_memory("Be brief.", [HAIKU, BRIEF])
    assert system == render(
        "remembered",
        {"system": "Be brief.", "notes": f"- {HAIKU}\n- {BRIEF}"},
    )
    assert system.startswith("Be brief.\n\n")
    assert f"<memory>\n- {HAIKU}\n- {BRIEF}\n</memory>" in system
