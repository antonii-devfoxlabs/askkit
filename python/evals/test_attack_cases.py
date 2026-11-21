from pathlib import Path

import pytest
from anthropic import Anthropic

from askkit.attacks import Attack, attack_won, load_attacks
from askkit.sorting_set import queue_for

# Every attack email of samples/attacks.csv, sent as askkit inbox sends
# it, one billed call each. A case fails when its attack won, and every
# case runs, so the run counts how many won instead of stopping at the
# first. Run them alone with uv run pytest evals -m attacks, after every
# change to the classify_email prompt, to the marking or to the model.
ATTACKS = load_attacks(
    Path(__file__).parent / "attacks" / "cases.json",
    Path(__file__).parents[1] / "samples" / "attacks.csv",
)
MAX_TOKENS = 1000


def attack_name(item: tuple[Attack, str]) -> str:
    return item[0].id


@pytest.mark.attacks
@pytest.mark.parametrize("item", ATTACKS, ids=attack_name)
def test_attack(
    client: Anthropic,
    model: str,
    attacks_won: list[str],
    attacks_run: list[str],
    item: tuple[Attack, str],
) -> None:
    attack, email = item
    queue, reason = queue_for(client, email, model, MAX_TOKENS)
    attacks_run.append(attack.id)
    if attack_won(attack, queue):
        attacks_won.append(attack.id)
    assert not attack_won(attack, queue), (
        f"the attack won: {queue}, not {attack.queue} ({attack.how})"
    )
    assert reason is None, f"went to a person: {reason}"
