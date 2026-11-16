from pathlib import Path

import pytest

from askkit.cli import main
from askkit.compare import compare_models
from askkit.prices import PRICES


def test_every_model_is_compared_most_capable_first() -> None:
    rows = compare_models(101, 40, 36_000)
    assert [row.split()[0] for row in rows] == [
        "claude-fable-5-2",
        "claude-opus-5-5",
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
    ]


def test_each_model_shows_one_call_and_all_calls() -> None:
    assert compare_models(101, 40, 36_000) == [
        "claude-fable-5-2   $0.002408 a call   $86.69 in all",
        "claude-opus-5-5    $0.001204 a call   $43.34 in all",
        "claude-sonnet-5-5  $0.000602 a call   $21.67 in all",
        "claude-haiku-5-5   $0.000030 a call   $1.08 in all",
    ]


def test_a_call_that_does_not_fit_is_not_priced() -> None:
    rows = compare_models(999_990, 40, 1)
    assert rows == [f"{model:<18} does not fit" for model in PRICES]


def test_compare_command_prints_the_tokens_then_every_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    email = tmp_path / "email.txt"
    email.write_text("a" * 272, encoding="utf-8")
    main(["compare", str(email), "--output", "40", "--calls", "36000"])
    assert capsys.readouterr().out.splitlines() == [
        "Input tokens:  101 (from 272 characters)",
        "Output tokens: 40 (your guess)",
        "Calls:         36,000",
        "claude-fable-5-2   $0.002408 a call   $86.69 in all",
        "claude-opus-5-5    $0.001204 a call   $43.34 in all",
        "claude-sonnet-5-5  $0.000602 a call   $21.67 in all",
        "claude-haiku-5-5   $0.000030 a call   $1.08 in all",
    ]


def test_fable_can_be_estimated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    email = tmp_path / "email.txt"
    email.write_text("a" * 272, encoding="utf-8")
    main(["estimate", str(email), "-m", "claude-fable-5-2", "--output", "40"])
    assert "One call:      $0.002408" in capsys.readouterr().out.splitlines()
