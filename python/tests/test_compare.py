from pathlib import Path

import pytest

from askkit.cli import main
from askkit.compare import compare_models
from askkit.prices import PRICES

EMAIL = (
    "Summarise this support email in one line for the queue.\n"
    "\n"
    "Hi, my parcel for order 4417 arrived this morning with a torn box,\n"
    "and the mug inside was broken. I'd like a replacement, not a refund.\n"
    "Can you send one this week? I've attached a photo of the damage.\n"
    "Thanks, Maria\n"
)

ROWS = [
    "claude-haiku-5-5           ~101 in   $0.000030 a call   $1.08 in all",
    "gemini-3.5-flash-lite       ~68 in   $0.000120 a call   $4.33 in all",
    "gpt-6-luna                  ~68 in   $0.000027 a call   $0.96 in all",
    "qwen3.8-27b                  73 in   $0.000157 a call   $5.63 in all",
]


def test_every_model_is_compared_in_the_price_table_order() -> None:
    rows = compare_models(EMAIL, 40, 36_000)
    assert [row.split()[0] for row in rows] == list(PRICES)


def test_each_model_counts_the_text_its_own_way() -> None:
    # ~ marks an estimate; qwen3.8-27b is counted by its own tokenizer.
    assert compare_models(EMAIL, 40, 36_000) == ROWS


def test_a_call_that_does_not_fit_is_not_priced() -> None:
    rows = compare_models("a" * 2_699_973, 40, 1)
    assert rows[0] == f"{'claude-haiku-5-5':<22} does not fit"


def test_compare_command_prints_the_text_then_every_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    email = tmp_path / "email.txt"
    email.write_text(EMAIL, encoding="utf-8")
    main(["compare", str(email), "--output", "40", "--calls", "36000"])
    assert capsys.readouterr().out.splitlines() == [
        "Characters:    272",
        "Output tokens: 40 (your guess)",
        "Calls:         36,000",
        *ROWS,
    ]
