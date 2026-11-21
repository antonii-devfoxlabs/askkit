import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from test_routing import answering_in_turn

from askkit.classify import QUEUES, sorting_messages
from askkit.cli import main
from askkit.inbox import TO_A_PERSON, read_inbox
from askkit.prompts import render
from askkit.sorting_set import (
    SORTING_KINDS,
    SortingCase,
    draw_held_out,
    load_sorting_cases,
    queue_for,
    split_set,
    write_part,
)

SORTING = Path(__file__).parent.parent / "evals" / "classify"
MAIN = SORTING / "main"
HELD_OUT = SORTING / "held_out"

# A fingerprint of the held-out part's files. It was drawn once, with
# askkit split evals/classify --per-kind 3 --seed 14, before the keyword
# rules or a model ran on it; changing it after reading a result is
# tuning on it, so a change has to be made on purpose.
HELD_OUT_FINGERPRINT = (
    "95395e33c19b1e4e8df3621d0c98434b834d8e3f43cf6c7335ea535e427ff11c"
)


def fingerprint(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(folder.iterdir()):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def case(case_id: str, kind: str, queue: str = "other") -> SortingCase:
    return SortingCase.model_validate(
        {"id": case_id, "queue": queue, "kind": kind, "why": "a test"}
    )


def every_kind(count: int) -> list[tuple[SortingCase, str]]:
    cases: list[tuple[SortingCase, str]] = []
    for kind_number, kind in enumerate(SORTING_KINDS):
        for number in range(count):
            case_id = f"k{kind_number}n{number}"
            cases.append((case(case_id, kind), f"Email {case_id}"))
    return cases


def test_both_parts_load_with_every_kind_in_each() -> None:
    for folder in (MAIN, HELD_OUT):
        kinds: list[str] = []
        for sorting_case, _ in load_sorting_cases(folder):
            if sorting_case.kind not in kinds:
                kinds.append(sorting_case.kind)
        assert sorted(kinds) == sorted(SORTING_KINDS), folder.name


def test_the_set_holds_28_main_and_21_held_out_cases() -> None:
    assert len(load_sorting_cases(MAIN)) == 28
    assert len(load_sorting_cases(HELD_OUT)) == 21


def test_every_queue_is_in_the_held_out_part() -> None:
    queues: list[str] = []
    for sorting_case, _ in load_sorting_cases(HELD_OUT):
        queues.append(sorting_case.queue)
    for queue in QUEUES:
        assert queue in queues


def test_no_case_is_in_both_parts() -> None:
    main_ids: list[str] = []
    for sorting_case, _ in load_sorting_cases(MAIN):
        main_ids.append(sorting_case.id)
    for sorting_case, _ in load_sorting_cases(HELD_OUT):
        assert sorting_case.id not in main_ids


def test_a_part_is_a_file_askkit_inbox_reads() -> None:
    assert len(read_inbox(MAIN / "emails.csv")) == 28


def test_the_held_out_part_is_unchanged() -> None:
    assert fingerprint(HELD_OUT) == HELD_OUT_FINGERPRINT


def test_the_same_seed_draws_the_same_cases() -> None:
    cases = []
    for sorting_case, _ in every_kind(5):
        cases.append(sorting_case)
    first = draw_held_out(cases, 2, 14)
    assert draw_held_out(cases, 2, 14) == first
    assert len(first) == 2 * len(SORTING_KINDS)


def test_every_kind_gives_the_same_number() -> None:
    cases = []
    for sorting_case, _ in every_kind(5):
        cases.append(sorting_case)
    drawn = draw_held_out(cases, 2, 14)
    for kind_number in range(len(SORTING_KINDS)):
        prefix = f"k{kind_number}n"
        assert len([name for name in drawn if name.startswith(prefix)]) == 2


def test_a_draw_that_would_empty_a_kind_is_refused() -> None:
    cases = []
    for sorting_case, _ in every_kind(2):
        cases.append(sorting_case)
    with pytest.raises(ValueError, match="none in the main part"):
        draw_held_out(cases, 2, 14)


def test_split_moves_the_drawn_cases(tmp_path: Path) -> None:
    write_part(tmp_path / "main", every_kind(4))
    drawn = split_set(tmp_path, 1, 14)
    main_ids: list[str] = []
    for sorting_case, _ in load_sorting_cases(tmp_path / "main"):
        main_ids.append(sorting_case.id)
    held_ids: list[str] = []
    for sorting_case, email in load_sorting_cases(tmp_path / "held_out"):
        held_ids.append(sorting_case.id)
        assert email == f"Email {sorting_case.id}"
    assert sorted(held_ids) == sorted(drawn)
    assert len(main_ids) == 3 * len(SORTING_KINDS)
    for name in drawn:
        assert name not in main_ids


def test_a_second_split_is_refused(tmp_path: Path) -> None:
    write_part(tmp_path / "main", every_kind(4))
    split_set(tmp_path, 1, 14)
    with pytest.raises(ValueError, match="drawn once"):
        split_set(tmp_path, 1, 15)


def test_the_split_command_prints_both_parts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_part(tmp_path / "main", every_kind(4))
    main(["split", str(tmp_path), "--per-kind", "1", "--seed", "14"])
    assert capsys.readouterr().out == (
        "Held out:      7 cases, 1 of each kind\nMain:          21 cases\n"
    )


def test_an_email_without_its_case_stops_the_set(tmp_path: Path) -> None:
    write_part(tmp_path, every_kind(1))
    data = json.loads((tmp_path / "cases.json").read_text(encoding="utf-8"))
    (tmp_path / "cases.json").write_text(json.dumps(data[1:]))
    with pytest.raises(ValueError, match="no case for k0n0"):
        load_sorting_cases(tmp_path)


def test_a_queue_the_shop_does_not_have_stops_the_set(
    tmp_path: Path,
) -> None:
    write_part(tmp_path, every_kind(1))
    data = json.loads((tmp_path / "cases.json").read_text(encoding="utf-8"))
    data[0]["queue"] = "refunds"
    (tmp_path / "cases.json").write_text(json.dumps(data))
    with pytest.raises(ValidationError):
        load_sorting_cases(tmp_path)


def test_one_case_is_sent_as_askkit_inbox_sends_it() -> None:
    sent: list[dict[str, Any]] = []
    client = answering_in_turn(sent, ['{"queue": "damaged"}'])
    email = "It came cracked.\n\nOn 2 Nov 2026, the shop wrote:\n> Sent."
    assert queue_for(client, email, "claude-haiku-5-5", 1000) == (
        "damaged",
        None,
    )
    assert sent[0]["system"] == render("classify_email", {})
    assert sent[0]["messages"] == sorting_messages("It came cracked.")


def test_an_answer_that_is_not_a_queue_goes_to_a_person() -> None:
    client = answering_in_turn([], ['{"queue": "refunds"}'])
    queue, reason = queue_for(client, "Hi", "claude-haiku-5-5", 1000)
    assert queue == TO_A_PERSON
    assert reason is not None
