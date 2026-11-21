from pathlib import Path

import pytest

from askkit.baseline import KEYWORDS, keyword_misses, keyword_queue
from askkit.cli import main
from askkit.sorting_set import load_sorting_cases

SORTING = Path(__file__).parent.parent / "evals" / "classify"

# Lesson 1.8's six rules, the start the list grew from.
LESSON_ONE_EIGHT = [
    "refund",
    "money back",
    "broken",
    "damaged",
    "where is",
    "tracking",
]


def test_the_first_keyword_found_decides() -> None:
    email = "My order 2287 was broken. I'd like a refund."
    assert keyword_queue(email) == "damaged"


def test_an_email_with_no_keyword_goes_to_other() -> None:
    assert keyword_queue("Do you sell gift cards?") == "other"


def test_capitals_do_not_matter() -> None:
    assert keyword_queue("WHERE IS my parcel?") == "delivery"


def test_quoted_lines_are_not_read() -> None:
    # The shop's own words, quoted under the reply, name tracking; the
    # customer's new message names nothing the rules know.
    email = "Thanks!\n\nOn 3 Nov 2026, the shop wrote:\n> Tracking: soon."
    assert keyword_queue(email) == "other"


def test_lesson_one_eights_rules_are_all_still_there() -> None:
    for keyword in LESSON_ONE_EIGHT:
        assert keyword in KEYWORDS


def test_the_rules_score_on_both_parts() -> None:
    main_part = load_sorting_cases(SORTING / "main")
    held_out = load_sorting_cases(SORTING / "held_out")
    assert len(main_part) - len(keyword_misses(main_part)) == 21
    assert len(held_out) - len(keyword_misses(held_out)) == 11


def test_baseline_lists_the_main_part_and_its_misses_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["baseline", str(SORTING)])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Main:          21 of 28 right (75%)"
    assert lines[1] == (
        "Wrong:         s14  other, not damaged (other words)"
    )
    # The score and seven misses. The held-out part is never opened, so
    # no id of a held-out case is printed and no held-out score either.
    assert len(lines) == 8
    held_out_ids: list[str] = []
    for sorting_case, _ in load_sorting_cases(SORTING / "held_out"):
        held_out_ids.append(sorting_case.id)
    for line in lines:
        assert not line.startswith("Held out")
        for case_id in held_out_ids:
            assert case_id not in line


def test_baseline_does_not_need_the_held_out_part(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "main").mkdir()
    for name in ("cases.json", "emails.csv"):
        (tmp_path / "main" / name).write_bytes(
            (SORTING / "main" / name).read_bytes()
        )
    main(["baseline", str(tmp_path)])
    assert capsys.readouterr().out.startswith("Main:          21 of 28")
