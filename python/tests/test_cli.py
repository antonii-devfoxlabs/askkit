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
