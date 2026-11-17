import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from askkit.cli import main
from askkit.golden import KINDS, Case, email_text, kind_counts, load_cases

EXTRACT = Path(__file__).parent.parent / "evals" / "extract"
MAIN = EXTRACT / "main"
HELD_OUT = EXTRACT / "held_out"

# A fingerprint of every file in the held-out part. Changing a held-out
# case after reading how a prompt did on it is tuning on it, so a change
# has to be made on purpose: the test fails until this line changes too.
HELD_OUT_FINGERPRINT = (
    "b0db444a27d6e42baf2f32fae0d91e0b6ee6ecdcf82263a43c8c0c2cedcd8d6c"
)


def fingerprint(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(folder.iterdir()):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def kinds_in(cases: list[Case]) -> list[str]:
    kinds: list[str] = []
    for case in cases:
        if case.kind not in kinds:
            kinds.append(case.kind)
    return kinds


def write_part(folder: Path, cases: list[dict[str, object]]) -> None:
    folder.mkdir()
    (folder / "cases.json").write_text(json.dumps(cases), encoding="utf-8")


MARIA = {
    "email": "maria",
    "order": 4417,
    "wish": "replacement",
    "kind": "plain",
    "why": "The email of lesson 1.2",
}


def test_both_parts_load() -> None:
    assert len(load_cases(MAIN)) == 16
    assert len(load_cases(HELD_OUT)) == 8


def test_every_kind_is_in_both_parts() -> None:
    assert sorted(kinds_in(load_cases(MAIN))) == sorted(KINDS)
    assert sorted(kinds_in(load_cases(HELD_OUT))) == sorted(KINDS)


def test_no_email_is_in_both_parts() -> None:
    main_texts: list[str] = []
    for case in load_cases(MAIN):
        main_texts.append(" ".join(email_text(MAIN, case).lower().split()))
    for case in load_cases(HELD_OUT):
        text = " ".join(email_text(HELD_OUT, case).lower().split())
        assert text not in main_texts, case.email
        assert (MAIN / f"{case.email}.txt").exists() is False, case.email


def test_the_held_out_part_is_unchanged() -> None:
    assert fingerprint(HELD_OUT) == HELD_OUT_FINGERPRINT


def test_a_case_without_its_email_is_refused(tmp_path: Path) -> None:
    write_part(tmp_path / "main", [MARIA])
    with pytest.raises(ValueError, match="no email file for 'maria'"):
        load_cases(tmp_path / "main")


def test_an_email_without_a_case_is_refused(tmp_path: Path) -> None:
    folder = tmp_path / "main"
    write_part(folder, [MARIA])
    (folder / "maria.txt").write_text("Order 4417", encoding="utf-8")
    (folder / "tom.txt").write_text("No order", encoding="utf-8")
    with pytest.raises(ValueError, match="no case for tom.txt"):
        load_cases(folder)


def test_one_name_used_twice_is_refused(tmp_path: Path) -> None:
    folder = tmp_path / "main"
    write_part(folder, [MARIA, MARIA])
    (folder / "maria.txt").write_text("Order 4417", encoding="utf-8")
    with pytest.raises(ValueError, match="two cases for 'maria'"):
        load_cases(folder)


def test_an_expected_wish_ticket_would_refuse_is_refused(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "main"
    write_part(folder, [dict(MARIA, wish="Replacement")])
    (folder / "maria.txt").write_text("Order 4417", encoding="utf-8")
    with pytest.raises(ValidationError, match="wish"):
        load_cases(folder)


def test_a_way_out_is_an_expected_value(tmp_path: Path) -> None:
    folder = tmp_path / "main"
    tom = dict(MARIA, email="tom", order=None, wish="not stated")
    write_part(folder, [tom])
    (folder / "tom.txt").write_text("No order", encoding="utf-8")
    case = load_cases(folder)[0]
    assert case.order is None
    assert case.wish == "not stated"


def test_kind_counts_shows_a_kind_with_no_case_as_zero() -> None:
    counts = kind_counts([Case.model_validate(MARIA)])
    assert counts["plain"] == 1
    assert counts["language"] == 0
    assert list(counts) == list(KINDS)


def test_cases_counts_both_parts_by_kind(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["cases", str(EXTRACT)])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Kind           Main  Held out"
    assert lines[1] == "plain             2         1"
    assert lines[-1] == "Total            16         8"
    assert len(lines) == len(KINDS) + 2


def test_cases_names_a_kind_missing_from_a_part(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for part in ("main", "held_out"):
        write_part(tmp_path / part, [MARIA])
        (tmp_path / part / "maria.txt").write_text("4417", encoding="utf-8")
    with pytest.raises(SystemExit) as stop:
        main(["cases", str(tmp_path)])
    assert str(stop.value) == (
        "askkit: no case in one part for no order, no wish, two wishes,"
        " quoted, look-alike, problems, language"
    )
    assert "plain             1         1" in capsys.readouterr().out


def test_cases_stops_on_a_broken_set(tmp_path: Path) -> None:
    write_part(tmp_path / "main", [MARIA])
    with pytest.raises(SystemExit) as stop:
        main(["cases", str(tmp_path)])
    assert str(stop.value) == "askkit: main: no email file for 'maria'"
