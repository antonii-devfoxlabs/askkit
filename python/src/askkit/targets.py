import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

# The course's minimum for a Level 1 final project: the lowest share of
# held-out cases a model must get right before the app ships. Below it,
# more than one email in ten needs a person to move it. A project's own
# target may be higher, never lower.
COURSE_MINIMUM_RIGHT = 0.9


# What the final project must reach, written before any model runs on
# the held-out part. Numbers only, and no model named: the targets are
# the job's, so any model of any provider is held to the same ones.
class Targets(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: str
    emails_a_month: int = Field(gt=0)
    cost_per_error: float = Field(gt=0)
    held_out_right: float = Field(ge=0, le=1)
    attacks_won: int = Field(ge=0)
    month_budget: float = Field(gt=0)


def load_targets(path: Path) -> Targets:
    return Targets.model_validate_json(path.read_text(encoding="utf-8"))


def cases_needed(share: float, total: int) -> int:
    # The fewest cases of total that reach the share: 0.9 of 21 is 18.9,
    # so 19, since a part of 21 scores in steps of one case. Rounded
    # first: 0.56 * 25 is 14.000000000000002 in floating point,
    # which math.ceil would make 15.
    return math.ceil(round(share * total, 9))


def month_cost(targets: Targets, right: float, bill: float) -> float:
    # Lesson 1.8's month: the bill plus what the wrong answers cost.
    wrong = targets.emails_a_month * (1 - right)
    return bill + wrong * targets.cost_per_error


def below_minimum(targets: Targets, rules_right: float) -> list[str]:
    # Every way the targets fall short of the course's minimum, given the
    # share the keyword rules got right on the held-out part. An empty
    # list means the plan may go ahead.
    faults: list[str] = []
    if targets.held_out_right < COURSE_MINIMUM_RIGHT:
        faults.append(
            f"held_out_right: {targets.held_out_right:.2f} is below the"
            f" course minimum of {COURSE_MINIMUM_RIGHT:.2f}"
        )
    if targets.held_out_right <= rules_right:
        faults.append(
            f"held_out_right: {targets.held_out_right:.2f} does not beat"
            f" the keyword rules' {rules_right:.2f}"
        )
    if targets.attacks_won > 0:
        faults.append(
            f"attacks_won: {targets.attacks_won} allowed; the course"
            " minimum is 0"
        )
    model_month = month_cost(
        targets, targets.held_out_right, targets.month_budget
    )
    rules_month = month_cost(targets, rules_right, 0)
    if model_month >= rules_month:
        faults.append(
            f"month_budget: ${model_month:,.2f} a month with the errors,"
            f" no less than the keyword rules' ${rules_month:,.2f}"
        )
    return faults
