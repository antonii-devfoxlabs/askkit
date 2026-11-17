from pathlib import Path

import pytest
from anthropic import Anthropic

from askkit.checks import run_case
from askkit.golden import Case, load_cases

# These tests call the model, one billed call per case, so they live
# here and not in tests/: uv run pytest and CI never run them. Run them
# with uv run pytest evals, and the held-out part with -m held_out. The
# model and the prompt version come from the command line (conftest.py).
EXTRACT = Path(__file__).parent / "extract"
MAIN = EXTRACT / "main"
HELD_OUT = EXTRACT / "held_out"
MAX_TOKENS = 1000


def case_name(case: Case) -> str:
    return case.email


@pytest.mark.parametrize("case", load_cases(MAIN), ids=case_name)
def test_main(
    client: Anthropic,
    model: str,
    version: int,
    results: dict[str, list[str]],
    case: Case,
) -> None:
    faults = run_case(client, MAIN, case, model, MAX_TOKENS, version)
    results[case.email] = faults
    assert faults == [], "; ".join(faults)


@pytest.mark.held_out
@pytest.mark.parametrize("case", load_cases(HELD_OUT), ids=case_name)
def test_held_out(
    client: Anthropic,
    model: str,
    version: int,
    results: dict[str, list[str]],
    case: Case,
) -> None:
    faults = run_case(client, HELD_OUT, case, model, MAX_TOKENS, version)
    results[case.email] = faults
    assert faults == [], "; ".join(faults)
