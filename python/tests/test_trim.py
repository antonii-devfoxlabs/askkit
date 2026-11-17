from askkit.history import SavedMessage
from askkit.prompts import render
from askkit.trim import transcript, turns_label, with_summary


def test_one_turn_and_several_are_named_apart() -> None:
    assert turns_label(3, 3) == "turn 3"
    assert turns_label(1, 4) == "turns 1 to 4"


def test_a_transcript_names_who_wrote_each_line() -> None:
    saved = [
        SavedMessage(role="user", content="Which costs less?"),
        SavedMessage(role="assistant", content="Haiku 5.5."),
    ]
    assert transcript(saved) == (
        "User: Which costs less?\nAssistant: Haiku 5.5."
    )


def test_no_summary_leaves_the_system_prompt_as_it_is() -> None:
    assert with_summary("Be brief.", "") == "Be brief."


def test_a_summary_follows_the_system_prompt() -> None:
    system = with_summary("Be brief.", "Order 4417.")
    assert system == render(
        "earlier_turns", {"system": "Be brief.", "summary": "Order 4417."}
    )
    assert system.startswith("Be brief.\n\n")
    assert system.endswith("<summary>\nOrder 4417.\n</summary>")
