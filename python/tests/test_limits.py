from pathlib import Path

import pytest

from askkit.cli import main
from askkit.limits import LIMITS
from askkit.prices import PRICES


def test_every_priced_model_has_limits() -> None:
    assert sorted(LIMITS) == sorted(PRICES)


def test_estimate_prints_the_window_used(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    email = tmp_path / "email.txt"
    email.write_text("a" * 270, encoding="utf-8")
    main(["estimate", str(email), "--output", "40"])
    assert "Window:        140 of 1,000,000 tokens" in (
        capsys.readouterr().out.splitlines()
    )


def test_text_and_answer_over_the_window_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    book = tmp_path / "book.txt"
    # 2,699,730 characters is 999,900 tokens: 101 more do not fit.
    book.write_text("a" * 2_699_730, encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        main(["estimate", str(book), "--output", "101"])
    assert exit_info.value.code == (
        "askkit: 1,000,001 tokens do not fit the context window"
        " of claude-haiku-5-5, 1,000,000 tokens"
    )
    assert capsys.readouterr().out == ""


def test_text_and_answer_that_just_fit_are_estimated(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    book = tmp_path / "book.txt"
    book.write_text("a" * 2_699_730, encoding="utf-8")
    main(["estimate", str(book), "--output", "100"])
    assert "Window:        1,000,000 of 1,000,000 tokens" in (
        capsys.readouterr().out.splitlines()
    )


def test_output_over_the_maximum_is_refused(tmp_path: Path) -> None:
    email = tmp_path / "email.txt"
    email.write_text("hi", encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        main(["estimate", str(email), "--output", "128001"])
    assert exit_info.value.code == (
        "askkit: --output 128,001 is over the most claude-haiku-5-5"
        " writes in one call, 128,000 tokens"
    )


def test_every_model_has_a_window_an_input_and_an_output_limit() -> None:
    for limits in LIMITS.values():
        assert sorted(limits) == ["context_window", "max_input", "max_output"]
        assert limits["max_input"] <= limits["context_window"]
        assert limits["max_output"] <= limits["context_window"]


def test_input_over_the_maximum_input_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # gpt-6-luna reads at most 922,000 tokens, below its 1,050,000
    # window: 3,688,004 characters is 922,001 estimated tokens.
    book = tmp_path / "book.txt"
    book.write_text("a" * 3_688_004, encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        main(["estimate", str(book), "-m", "gpt-6-luna", "--output", "40"])
    assert exit_info.value.code == (
        "askkit: 922,001 input tokens are over the most gpt-6-luna"
        " reads in one call, 922,000 tokens"
    )
    assert capsys.readouterr().out == ""
