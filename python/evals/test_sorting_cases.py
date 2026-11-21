from pathlib import Path

import pytest
from anthropic import Anthropic

from askkit.sorting_set import SortingCase, load_sorting_cases, queue_for

# The sorting set of the final project, each email sent as askkit inbox
# sends it, one billed call a case. Each part has a marker of its own,
# deselected by default, so the extract set's runs stay as they were:
# -m sorting runs the main part, -m sorting_held_out the held-out part,
# once a version is ready.
SORTING = Path(__file__).parent / "classify"
MAIN = load_sorting_cases(SORTING / "main")
HELD_OUT = load_sorting_cases(SORTING / "held_out")
MAX_TOKENS = 1000


def case_name(item: tuple[SortingCase, str]) -> str:
    return item[0].id


def assert_sorted_right(
    client: Anthropic, model: str, item: tuple[SortingCase, str]
) -> None:
    case, email = item
    queue, reason = queue_for(client, email, model, MAX_TOKENS)
    assert reason is None, f"went to a person: {reason}"
    assert queue == case.queue, f"{queue}, not {case.queue} ({case.kind})"


@pytest.mark.sorting
@pytest.mark.parametrize("item", MAIN, ids=case_name)
def test_sorting(
    client: Anthropic, model: str, item: tuple[SortingCase, str]
) -> None:
    assert_sorted_right(client, model, item)


@pytest.mark.sorting_held_out
@pytest.mark.parametrize("item", HELD_OUT, ids=case_name)
def test_sorting_held_out(
    client: Anthropic, model: str, item: tuple[SortingCase, str]
) -> None:
    assert_sorted_right(client, model, item)
