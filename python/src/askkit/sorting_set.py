import csv
import json
import random
from pathlib import Path
from typing import Literal, get_args

from anthropic import Anthropic
from pydantic import BaseModel, ConfigDict, ValidationError

from askkit.ask import answer_text, stop_warning
from askkit.classify import (
    CLASSIFICATION_SCHEMA,
    Queue,
    check_classification,
    classification_error,
    sorting_messages,
)
from askkit.clean import clean_email
from askkit.inbox import COLUMNS, TO_A_PERSON, read_inbox
from askkit.prompts import render
from askkit.structured import json_request

# The kinds of email the sorting set covers, each a way an email can be
# hard to sort. Every kind has cases in both parts, so the held-out part
# asks the same questions as the main one.
SortingKind = Literal[
    "plain",
    "other words",
    "misleading word",
    "two problems",
    "quoted",
    "language",
    "no order",
]
SORTING_KINDS: tuple[str, ...] = get_args(SortingKind)


# One case of the sorting set: the id of its email in the part's
# emails.csv, the queue a person who read the email chose for it, its
# kind and why it is in the set. The queue is classify's own type, so a
# queue the shop does not have is refused when the set loads.
class SortingCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    queue: Queue
    kind: SortingKind
    why: str


def load_sorting_cases(folder: Path) -> list[tuple[SortingCase, str]]:
    # One part of the set: its cases.json and its emails.csv, a file
    # askkit inbox can sort as it is. A case without its email, an email
    # without its case, or one id twice is a mistake in the set, and an
    # eval run on it would score the wrong thing, so it stops here.
    emails: dict[str, str] = {}
    for email in read_inbox(folder / "emails.csv"):
        emails[email.id] = email.email
    data = json.loads((folder / "cases.json").read_text(encoding="utf-8"))
    cases: list[tuple[SortingCase, str]] = []
    seen: list[str] = []
    for item in data:
        case = SortingCase.model_validate(item)
        if case.id in seen:
            raise ValueError(f"{folder.name}: two cases for {case.id}")
        if case.id not in emails:
            raise ValueError(f"{folder.name}: no email for {case.id}")
        seen.append(case.id)
        cases.append((case, emails[case.id]))
    for email_id in emails:
        if email_id not in seen:
            raise ValueError(f"{folder.name}: no case for {email_id}")
    return cases


def draw_held_out(
    cases: list[SortingCase], per_kind: int, seed: int
) -> list[str]:
    # The ids of the held-out part: per_kind cases of every kind, drawn
    # by chance. The seed makes the same set give the same draw, so the
    # draw can be checked; it is still made once, before anything runs.
    by_kind: dict[str, list[str]] = {}
    for kind in SORTING_KINDS:
        by_kind[kind] = []
    for case in cases:
        by_kind[case.kind].append(case.id)
    chance = random.Random(seed)
    drawn: list[str] = []
    for kind in SORTING_KINDS:
        if len(by_kind[kind]) <= per_kind:
            raise ValueError(
                f"{kind}: {len(by_kind[kind])} cases, so drawing"
                f" {per_kind} would leave none in the main part"
            )
        drawn.extend(chance.sample(by_kind[kind], per_kind))
    return drawn


def write_part(folder: Path, cases: list[tuple[SortingCase, str]]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "emails.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(COLUMNS)
        for case, email in cases:
            writer.writerow([case.id, email])
    data: list[dict[str, str]] = []
    for case, _ in cases:
        data.append(case.model_dump())
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    (folder / "cases.json").write_text(text, encoding="utf-8")


def split_set(set_folder: Path, per_kind: int, seed: int) -> list[str]:
    # Moves the drawn cases out of main/ into held_out/, emails and cases
    # alike. A held-out part is drawn once: a second draw would put cases
    # you have read and tuned on into it, so it refuses when one exists.
    held_out_folder = set_folder / "held_out"
    if held_out_folder.exists():
        raise ValueError(
            f"{held_out_folder} already exists: the held-out part is"
            " drawn once"
        )
    main_folder = set_folder / "main"
    cases = load_sorting_cases(main_folder)
    only_cases: list[SortingCase] = []
    for case, _ in cases:
        only_cases.append(case)
    drawn = draw_held_out(only_cases, per_kind, seed)
    kept: list[tuple[SortingCase, str]] = []
    held_out: list[tuple[SortingCase, str]] = []
    for case, email in cases:
        if case.id in drawn:
            held_out.append((case, email))
        else:
            kept.append((case, email))
    write_part(held_out_folder, held_out)
    write_part(main_folder, kept)
    return drawn


def queue_for(
    client: Anthropic, email: str, model: str, max_tokens: int
) -> tuple[str, str | None]:
    # The queue askkit inbox would give one email, and why it went to a
    # person when it did. The request is the one inbox sends: the email
    # cleaned and marked, the classify_email prompt in use, the schema.
    cleaned, _ = clean_email(email)
    response = client.messages.create(
        **json_request(
            model,
            max_tokens,
            render("classify_email", {}),
            sorting_messages(cleaned),
            CLASSIFICATION_SCHEMA,
        )
    )
    warning = stop_warning(response, "sort this email into a queue")
    if warning is not None:
        return TO_A_PERSON, warning
    try:
        classification, _ = check_classification(answer_text(response))
    except ValidationError as error:
        return TO_A_PERSON, classification_error(error)
    return classification.queue, None
