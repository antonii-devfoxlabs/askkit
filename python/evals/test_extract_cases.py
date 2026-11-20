from pathlib import Path

import pytest
from anthropic import Anthropic

from askkit.checks import run_case, run_routed_case
from askkit.golden import Case, load_cases

# These tests call the model, one billed call per case, so they live
# here and not in tests/: uv run pytest and CI never run them. Run them
# with uv run pytest evals, and the held-out part with -m held_out. The
# model, the prompt version and the model to escalate to come from the
# command line (conftest.py).
EXTRACT = Path(__file__).parent / "extract"
MAIN = EXTRACT / "main"
HELD_OUT = EXTRACT / "held_out"
MAX_TOKENS = 1000


def case_name(case: Case) -> str:
    return case.email


def score_case(
    client: Anthropic,
    folder: Path,
    case: Case,
    model: str,
    version: int,
    escalate_to: str | None,
    escalated: dict[str, bool],
) -> list[str]:
    # The first answer alone, or, with --escalate-to, the answer extract
    # would print after its one retry on the larger model.
    if escalate_to is None:
        return run_case(client, folder, case, model, MAX_TOKENS, version)
    faults, retried = run_routed_case(
        client, folder, case, model, escalate_to, MAX_TOKENS, version
    )
    escalated[case.email] = retried
    return faults


@pytest.mark.parametrize("case", load_cases(MAIN), ids=case_name)
def test_main(
    client: Anthropic,
    model: str,
    version: int,
    escalate_to: str | None,
    results: dict[str, list[str]],
    escalated: dict[str, bool],
    case: Case,
) -> None:
    faults = score_case(
        client, MAIN, case, model, version, escalate_to, escalated
    )
    results[case.email] = faults
    assert faults == [], "; ".join(faults)


@pytest.mark.held_out
@pytest.mark.parametrize("case", load_cases(HELD_OUT), ids=case_name)
def test_held_out(
    client: Anthropic,
    model: str,
    version: int,
    escalate_to: str | None,
    results: dict[str, list[str]],
    escalated: dict[str, bool],
    case: Case,
) -> None:
    faults = score_case(
        client, HELD_OUT, case, model, version, escalate_to, escalated
    )
    results[case.email] = faults
    assert faults == [], "; ".join(faults)
