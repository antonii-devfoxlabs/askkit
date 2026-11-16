from pathlib import Path

import pytest

from askkit.cli import main


def test_estimate_prints_tokens_and_costs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    email = tmp_path / "email.txt"
    email.write_text("a" * 1_350, encoding="utf-8")
    main(
        [
            "estimate",
            str(email),
            "-m",
            "claude-sonnet-5-5",
            "--output",
            "200",
            "--calls",
            "1000",
        ]
    )
    assert capsys.readouterr().out.splitlines() == [
        "Model:         claude-sonnet-5-5",
        "Input tokens:  500 (from 1,350 characters)",
        "Output tokens: 200 (your guess)",
        "Window:        700 of 1,000,000 tokens",
        "One call:      $0.003000",
        "1,000 calls: $3.00",
    ]


def test_one_call_prints_no_total(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    email = tmp_path / "email.txt"
    email.write_text("a" * 27, encoding="utf-8")
    main(["estimate", str(email), "--output", "10"])
    assert capsys.readouterr().out.splitlines() == [
        "Model:         claude-haiku-5-5",
        "Input tokens:  10 (from 27 characters)",
        "Output tokens: 10 (your guess)",
        "Window:        20 of 1,000,000 tokens",
        "One call:      $0.000006",
    ]


def test_unknown_model_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    email = tmp_path / "email.txt"
    email.write_text("hi", encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        main(["estimate", str(email), "--model", "claude-haiku-5.5"])
    assert exit_info.value.code == 2
    assert "invalid choice: 'claude-haiku-5.5'" in capsys.readouterr().err


def test_estimate_adds_what_the_errors_cost(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    email = tmp_path / "email.txt"
    email.write_text("a" * 270, encoding="utf-8")
    main(
        [
            "estimate",
            str(email),
            "--output",
            "5",
            "--calls",
            "36000",
            "--error-rate",
            "0.03",
            "--error-cost",
            "0.50",
        ]
    )
    assert capsys.readouterr().out.splitlines() == [
        "Model:         claude-haiku-5-5",
        "Input tokens:  100 (from 270 characters)",
        "Output tokens: 5 (your guess)",
        "Window:        105 of 1,000,000 tokens",
        "One call:      $0.000013",
        "36,000 calls: $0.45",
        "Errors:        1,080 wrong, $540.00",
        "With errors:   $540.45",
    ]


@pytest.mark.parametrize("rate", ["-0.1", "1.5", "3"])
def test_an_error_rate_outside_0_to_1_is_refused(
    tmp_path: Path, rate: str
) -> None:
    email = tmp_path / "email.txt"
    email.write_text("hi", encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        main(["estimate", str(email), "--error-rate", rate])
    assert exit_info.value.code == (
        f"askkit: --error-rate {rate} is not a share of the calls;"
        " give one from 0 to 1, such as 0.03 for 3 in 100"
    )


def test_compare_takes_no_error_options(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    email = tmp_path / "email.txt"
    email.write_text("hi", encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        main(["compare", str(email), "--error-rate", "0.03"])
    assert exit_info.value.code == 2
