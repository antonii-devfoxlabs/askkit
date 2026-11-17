from pathlib import Path

import pytest
from anthropic import Anthropic

from askkit.checks import run_case
from askkit.golden import Case, load_cases

# These tests call the model, one billed call per case, so they live
# here and not in tests/: uv run pytest and CI never run them. Run them
# with uv run pytest evals, and the held-out part with -m held_out.
EXTRACT = Path(__file__).parent / "extract"
MAIN = EXTRACT / "main"
HELD_OUT = EXTRACT / "held_out"
MODEL = "claude-haiku-5-5"
MAX_TOKENS = 1000


def case_name(case: Case) -> str:
    return case.email


@pytest.mark.parametrize("case", load_cases(MAIN), ids=case_name)
def test_main(client: Anthropic, case: Case) -> None:
    faults = run_case(client, MAIN, case, MODEL, MAX_TOKENS)
    assert faults == [], "; ".join(faults)


@pytest.mark.held_out
@pytest.mark.parametrize("case", load_cases(HELD_OUT), ids=case_name)
def test_held_out(client: Anthropic, case: Case) -> None:
    faults = run_case(client, HELD_OUT, case, MODEL, MAX_TOKENS)
    assert faults == [], "; ".join(faults)
