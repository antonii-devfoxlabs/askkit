from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from askkit.targets import Targets, cases_needed, month_cost

# The record of the one run that scored the held-out part, written into
# the sorting set's folder. While it exists, askkit release refuses to
# score that part again: a second score, after reading the first and
# changing the prompt, would describe a version tuned to those emails.
RELEASE_FILE = "release.json"


# What the release run measured, and every bar it missed. The misses of
# the held-out part are counted, never named: the part is run for a
# score, not read.
class Release(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scored_on: date
    model: str
    prompt_version: int
    held_out_right: int
    held_out_total: int
    attacks_won: list[str]
    attacks_total: int
    bill_per_email: float
    month_bill: float
    month: float
    rules_month: float
    faults: list[str]


def scored_before(folder: Path) -> Release | None:
    path = folder / RELEASE_FILE
    if not path.exists():
        return None
    return Release.model_validate_json(path.read_text(encoding="utf-8"))


def release_faults(
    targets: Targets,
    right: int,
    total: int,
    won: list[str],
    month_bill: float,
    rules_right: float,
) -> list[str]:
    # Every bar of the targets the run missed; an empty list means the
    # release may ship. The targets were held to the course minimum by
    # askkit targets, so meeting them meets the minimum too.
    faults: list[str] = []
    needed = cases_needed(targets.held_out_right, total)
    if right < needed:
        faults.append(f"held out: {right} of {total} right, {needed} needed")
    if len(won) > targets.attacks_won:
        faults.append(
            f"attacks: {len(won)} won ({', '.join(sorted(won))}),"
            f" {targets.attacks_won} allowed"
        )
    if month_bill > targets.month_budget:
        faults.append(
            f"bill: ${month_bill:,.2f} a month, over the budget of"
            f" ${targets.month_budget:,.2f}"
        )
    model_month = month_cost(targets, right / total, month_bill)
    rules_month = month_cost(targets, rules_right, 0)
    if model_month >= rules_month:
        faults.append(
            f"month: ${model_month:,.2f} with the errors, no less than the"
            f" keyword rules' ${rules_month:,.2f}"
        )
    return faults
