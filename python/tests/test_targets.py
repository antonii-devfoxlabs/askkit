from pathlib import Path

import pytest
from pydantic import ValidationError

from askkit.cli import main
from askkit.targets import (
    COURSE_MINIMUM_RIGHT,
    Targets,
    below_minimum,
    cases_needed,
    load_targets,
    month_cost,
)

SORTING = Path(__file__).parent.parent / "evals" / "classify"
RULES_RIGHT = 11 / 21


def targets(**changes: object) -> Targets:
    data: dict[str, object] = {
        "task": "Sort emails",
        "emails_a_month": 36000,
        "cost_per_error": 0.5,
        "held_out_right": 0.95,
        "attacks_won": 0,
        "month_budget": 5.0,
    }
    data.update(changes)
    return Targets.model_validate(data)


def test_askkits_targets_are_at_or_above_the_course_minimum() -> None:
    assert not below_minimum(
        load_targets(SORTING / "targets.json"), RULES_RIGHT
    )


def test_a_target_below_the_minimum_is_named() -> None:
    assert below_minimum(targets(held_out_right=0.85), RULES_RIGHT) == [
        "held_out_right: 0.85 is below the course minimum of 0.90"
    ]


def test_the_minimum_itself_is_allowed() -> None:
    plan = targets(held_out_right=COURSE_MINIMUM_RIGHT)
    assert below_minimum(plan, RULES_RIGHT) == []


def test_a_target_that_does_not_beat_the_rules_is_named() -> None:
    faults = below_minimum(targets(held_out_right=0.95), 0.96)
    assert faults[0] == (
        "held_out_right: 0.95 does not beat the keyword rules' 0.96"
    )


def test_an_attack_allowed_to_win_is_named() -> None:
    assert below_minimum(targets(attacks_won=1), RULES_RIGHT) == [
        "attacks_won: 1 allowed; the course minimum is 0"
    ]


def test_a_budget_that_costs_more_than_the_rules_is_named() -> None:
    faults = below_minimum(targets(month_budget=8000.0), RULES_RIGHT)
    assert faults == [
        "month_budget: $8,900.00 a month with the errors, no less than"
        " the keyword rules' $8,571.43"
    ]


def test_every_fault_is_named_at_once() -> None:
    plan = targets(held_out_right=0.5, attacks_won=2)
    assert len(below_minimum(plan, RULES_RIGHT)) == 4


def test_a_share_outside_0_to_1_is_refused() -> None:
    with pytest.raises(ValidationError):
        targets(held_out_right=95)


def test_an_unknown_key_is_refused() -> None:
    with pytest.raises(ValidationError):
        targets(model="claude-haiku-5-5")


def test_a_month_is_the_bill_plus_the_errors() -> None:
    assert month_cost(targets(), 0.95, 5.0) == pytest.approx(905.0)


def test_the_cases_needed_are_rounded_up() -> None:
    assert cases_needed(0.9, 21) == 19
    assert cases_needed(0.95, 21) == 20
    assert cases_needed(0.56, 25) == 14


def test_the_targets_command_prints_the_plan(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["targets", str(SORTING)])
    assert capsys.readouterr().out.splitlines()[1:] == [
        "Target:        20 of 21 held-out cases right (0.95)",
        "Minimum:       19 of 21 (0.90), and more than the rules' 11 (0.52)",
        "Attacks won:   0",
        "Bill:          at most $5.00 a month",
        "Month:         $905.00 with the errors, the rules' $8,571.43",
    ]


def test_targets_below_the_minimum_exit_with_every_fault(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "classify"
    folder.mkdir()
    (folder / "targets.json").write_text(
        targets(held_out_right=0.5).model_dump_json()
    )
    held_out = SORTING / "held_out"
    (folder / "held_out").mkdir()
    for name in ("cases.json", "emails.csv"):
        (folder / "held_out" / name).write_bytes(
            (held_out / name).read_bytes()
        )
    with pytest.raises(SystemExit) as stop:
        main(["targets", str(folder)])
    assert str(stop.value) == (
        "askkit: the targets are below the course minimum:"
        " held_out_right: 0.50 is below the course minimum of 0.90;"
        " held_out_right: 0.50 does not beat the keyword rules' 0.52;"
        " month_budget: $9,005.00 a month with the errors, no less than"
        " the keyword rules' $8,571.43"
    )
