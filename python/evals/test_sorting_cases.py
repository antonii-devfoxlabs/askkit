from pathlib import Path

import pytest
from anthropic import Anthropic

from askkit.sorting_set import SortingCase, load_sorting_cases, queue_for

# The main part of the sorting set, each email sent as askkit inbox sends
# it, one billed call a case, under a marker deselected by default, so
# the extract set's runs stay as they were: -m sorting runs it. The
# held-out part has no test here: a failing test names its case, and the
# held-out part is scored, never read. askkit release scores it, once.
SORTING = Path(__file__).parent / "classify"
MAIN = load_sorting_cases(SORTING / "main")
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
