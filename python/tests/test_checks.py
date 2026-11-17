import json
import re
from pathlib import Path

from askkit.checks import case_faults
from askkit.golden import Case, load_cases

EXTRACT = Path(__file__).parent.parent / "evals" / "extract"

TOASTER = Case(
    email="toaster",
    order=5566,
    wish="refund",
    mentions=["toaster", "spark"],
    kind="problems",
    why="Three problems, one of them a fire risk",
)

# A right ticket for every main case, written by a person from the
# email. A check that fails one of them is stricter than the job.
RIGHT_PROBLEMS = {
    "maria": "mug broken, box torn",
    "scarf": "scarf has a hole",
    "tom": "desk lamp flickers",
    "towels": "two towels torn at the seam",
    "chair": "chair leg missing",
    "jug": "glass jug leaks",
    "both": "teapot spout chipped",
    "pans": "pan handle bent",
    "reply": "mug in pieces, box torn",
    "blanket": "blanket smells of smoke",
    "kettle": "kettle stopped working",
    "plates": "2 plates cracked",
    "clock": "clock face cracked, hands bent",
    "toaster": "toaster sparks, cord melted",
    "taza": "cup broken",
    "lampe": "lamp arrived broken",
}


def answer(order: int | None, problem: str, wish: str) -> str:
    return json.dumps({"order": order, "problem": problem, "wish": wish})


def test_a_right_ticket_has_no_faults() -> None:
    text = answer(5566, "toaster sparks, cord melted", "refund")
    assert case_faults(text, TOASTER) == []


def test_a_problem_that_leaves_out_the_sparks_fails() -> None:
    # Lesson 5.1's check read the order and the wish only, so this
    # ticket passed it.
    text = answer(5566, "crumb tray missing", "refund")
    assert case_faults(text, TOASTER) == [
        "problem: no 'toaster' in 'crumb tray missing'",
        "problem: no 'spark' in 'crumb tray missing'",
    ]


def test_every_fault_is_named_not_only_the_first() -> None:
    text = answer(5556, "toaster sparks", "not stated")
    assert case_faults(text, TOASTER) == [
        "order: 5556, not 5566",
        "wish: 'not stated', not 'refund'",
    ]


def test_a_pattern_matches_in_any_capitals() -> None:
    text = answer(5566, "Toaster SPARKS", "refund")
    assert case_faults(text, TOASTER) == []


def test_any_word_of_a_pattern_will_do() -> None:
    case = Case(
        email="maria",
        order=4417,
        wish="replacement",
        mentions=["mug", "broke|smash|pieces|crack"],
        kind="plain",
        why="The email of lesson 1.2",
    )
    for problem in ("mug broken", "mug smashed", "mug in pieces"):
        assert case_faults(answer(4417, problem, "replacement"), case) == []
    assert case_faults(answer(4417, "mug chipped", "replacement"), case) == [
        "problem: no 'broke|smash|pieces|crack' in 'mug chipped'"
    ]


def test_an_answer_that_is_not_a_ticket_fails_on_the_schema() -> None:
    long = "toaster sparks, cord melted, tray missing"
    assert case_faults(answer(5566, long, "refund"), TOASTER) == [
        "problem: String should have at most 40 characters, not"
        " 'toaster sparks, cord melted, tray missing'"
    ]


def test_a_wish_in_other_capitals_is_scored_as_repaired() -> None:
    text = answer(5566, "toaster sparks", "Refund")
    assert case_faults(text, TOASTER) == []


def test_an_order_number_in_the_problem_fails() -> None:
    text = answer(5566, "order 5566 toaster sparks", "refund")
    assert case_faults(text, TOASTER) == [
        "problem: an order number in 'order 5566 toaster sparks'"
    ]


def test_a_quantity_in_the_problem_passes() -> None:
    case = Case(
        email="plates",
        order=7750,
        wish="replacement",
        mentions=["plate", "crack"],
        kind="look-alike",
        why="Quantities and a date",
    )
    text = answer(7750, "2 plates cracked", "replacement")
    assert case_faults(text, case) == []


def test_every_pattern_in_the_set_is_a_pattern_python_reads() -> None:
    for part in ("main", "held_out"):
        for case in load_cases(EXTRACT / part):
            assert case.mentions, case.email
            for pattern in case.mentions:
                re.compile(pattern)


def test_a_right_ticket_passes_every_main_case() -> None:
    cases = load_cases(EXTRACT / "main")
    assert sorted(RIGHT_PROBLEMS) == sorted(case.email for case in cases)
    for case in cases:
        text = answer(case.order, RIGHT_PROBLEMS[case.email], case.wish)
        assert case_faults(text, case) == [], case.email
