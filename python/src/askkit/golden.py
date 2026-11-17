import json
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict

from askkit.extract import Wish

# The kinds of email the golden set covers. Every kind has cases in both
# parts, so the held-out part asks the same questions as the main one.
Kind = Literal[
    "plain",
    "no order",
    "wish not stated",
    "two wishes",
    "quoted email",
    "other numbers",
    "many problems",
    "other language",
]
KINDS: tuple[str, ...] = get_args(Kind)


# One case of the golden set: an email, named by its file without the
# .txt, and what a right ticket for it must hold. The order and the wish
# use Ticket's own types, so an expected value Ticket would refuse is
# refused here first. The problem is left out: two right tickets rarely
# word it the same.
class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str
    order: int | None
    wish: Wish
    kind: Kind
    why: str


def load_cases(folder: Path) -> list[Case]:
    # Reads one part of the set: its cases.json and the email files
    # beside it. A case without its email, an email without a case, or
    # one name used twice is a mistake in the set, and an eval run on it
    # would score the wrong thing, so it stops here.
    data = json.loads((folder / "cases.json").read_text(encoding="utf-8"))
    cases: list[Case] = []
    names: list[str] = []
    for item in data:
        case = Case.model_validate(item)
        if case.email in names:
            raise ValueError(f"{folder.name}: two cases for {case.email!r}")
        if not (folder / f"{case.email}.txt").is_file():
            raise ValueError(
                f"{folder.name}: no email file for {case.email!r}"
            )
        cases.append(case)
        names.append(case.email)
    for path in sorted(folder.glob("*.txt")):
        if path.stem not in names:
            raise ValueError(f"{folder.name}: no case for {path.name}")
    return cases


def email_text(folder: Path, case: Case) -> str:
    return (folder / f"{case.email}.txt").read_text(encoding="utf-8")
