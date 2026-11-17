from pathlib import Path

import pytest

from askkit.analysis import (
    Failure,
    by_field,
    by_kind,
    failures,
    fault_fields,
    fault_lines,
    group_lines,
)
from askkit.cli import main
from askkit.golden import Case, load_cases
from askkit.runs import Results, add_run

EXTRACT = Path(__file__).parent.parent / "evals" / "extract"
TOO_LONG = (
    "problem: String should have at most 40 characters, not 'clock face"
    " cracked, hands bent, box crushed'"
)
WISH = "wish: 'refund', not 'replacement'"
TOASTER = [
    "problem: no 'toaster' in 'crumb tray missing'",
    "problem: no 'spark' in 'crumb tray missing'",
]


def main_cases() -> dict[str, Case]:
    cases: dict[str, Case] = {}
    for case in load_cases(EXTRACT / "main"):
        cases[case.email] = case
    return cases


def v2_runs() -> Results:
    return Results(
        model="claude-haiku-5-5",
        prompt_version=2,
        cases={
            "maria": [[], []],
            "clock": [[TOO_LONG], [TOO_LONG]],
            "pans": [[WISH], []],
            "toaster": [TOASTER, TOASTER],
        },
    )


def test_every_failed_answer_is_listed_run_by_run() -> None:
    found = failures("v2.json", v2_runs())
    assert [(failure.source, failure.case) for failure in found] == [
        ("v2.json run 1", "clock"),
        ("v2.json run 1", "pans"),
        ("v2.json run 1", "toaster"),
        ("v2.json run 2", "clock"),
        ("v2.json run 2", "toaster"),
    ]
    assert found[2].faults == TOASTER


def test_a_fault_names_its_field_before_the_colon() -> None:
    assert fault_fields(TOO_LONG) == ["problem"]
    assert fault_fields(WISH) == ["wish"]
    assert fault_fields("stop: the answer was cut off at 5 tokens") == [
        "stop"
    ]


def test_a_schema_fault_names_every_field_that_broke() -> None:
    # ticket_faults joins one part per field: the second field used to be
    # lost, and a failure in it went uncounted.
    fault = "order: Input should be a valid integer, not 'abc'; " + TOO_LONG
    assert fault_fields(fault) == ["order", "problem"]
    found = [Failure(source="v2.json run 1", case="clock", faults=[fault])]
    assert by_field(found) == {"order": ["clock"], "problem": ["clock"]}


def test_a_value_holding_a_semicolon_adds_no_field() -> None:
    fault = "problem: no 'spark' in 'tray missing; cord melted'"
    assert fault_fields(fault) == ["problem"]


def test_an_answer_counts_once_for_each_field_it_failed() -> None:
    found = [
        Failure(source="v2.json run 1", case="toaster", faults=TOASTER),
        Failure(
            source="v2.json run 1", case="pans", faults=[WISH, TOASTER[0]]
        ),
    ]
    assert by_field(found) == {
        "problem": ["toaster", "pans"],
        "wish": ["pans"],
    }


def test_failed_answers_are_grouped_by_the_kind_of_their_email() -> None:
    groups = by_kind(failures("v2.json", v2_runs()), main_cases())
    assert groups == {
        "problems": ["clock", "toaster", "clock", "toaster"],
        "two wishes": ["pans"],
    }


def test_the_largest_group_comes_first_and_names_each_case_once() -> None:
    lines = group_lines(
        "By field",
        {"wish": ["pans"], "order": ["kettle"], "problem": ["clock"] * 3},
    )
    assert lines == [
        "By field      Answers  Cases",
        "problem       3        clock",
        "order         1        kettle",
        "wish          1        pans",
    ]


def test_the_lines_count_the_runs_and_list_every_failed_answer() -> None:
    lines = fault_lines({"v2.json": v2_runs()}, main_cases())
    assert lines[0] == "5 failed answers in 2 runs of 16 cases"
    assert "problem       4        clock, toaster" in lines
    assert "two wishes    1        pans" in lines
    assert lines[-3:] == [
        "v2.json run 2   toaster",
        "  " + TOASTER[0],
        "  " + TOASTER[1],
    ]


def test_a_run_with_no_failure_says_so_in_one_line() -> None:
    results = Results(
        model="claude-haiku-5-5", prompt_version=2, cases={"maria": [[]]}
    )
    lines = fault_lines({"v2.json": results}, main_cases())
    assert lines == ["0 failed answers in 1 run of 16 cases"]
    one = Results(
        model="claude-haiku-5-5", prompt_version=2, cases={"pans": [[WISH]]}
    )
    assert fault_lines({"v4.json": one}, main_cases())[0] == (
        "1 failed answer in 1 run of 16 cases"
    )


def test_held_out_results_are_refused_so_they_stay_unread() -> None:
    held_out = Results(
        model="claude-haiku-5-5",
        prompt_version=2,
        cases={"bowl": [[WISH]], "maria": [[]]},
    )
    with pytest.raises(ValueError, match="not in this part.*: bowl;"):
        fault_lines({"held.json": held_out}, main_cases())


def test_the_faults_command_prints_the_lines_and_refuses_a_bad_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "v2.json"
    add_run(path, "claude-haiku-5-5", 2, {"maria": [], "pans": [WISH]})
    main(["faults", str(path), "--cases", str(EXTRACT / "main")])
    output = capsys.readouterr().out
    assert "wish          1        pans" in output
    assert f"v2.json run 1   pans\n  {WISH}\n" in output
    held = tmp_path / "held.json"
    add_run(held, "claude-haiku-5-5", 2, {"bowl": [WISH]})
    with pytest.raises(SystemExit) as stopped:
        main(["faults", str(held), "--cases", str(EXTRACT / "main")])
    assert str(stopped.value).startswith("askkit: held.json holds cases")
