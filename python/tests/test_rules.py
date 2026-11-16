import json
from pathlib import Path

from askkit.extract import Ticket
from askkit.rules import email_fault, load_orders, numbers_in, order_fault

EMAIL = (
    "Hello, order 4417 came yesterday and the mug inside was broken."
    " I'd like a replacement, not a refund. Thanks, Maria"
)


def ticket(order: int | None) -> Ticket:
    return Ticket(order=order, problem="mug broken", wish="replacement")


def test_numbers_in_finds_every_whole_number() -> None:
    assert numbers_in("Order #4417, sent 2 May.") == ["4417", "2"]
    assert numbers_in("no numbers here") == []
    assert numbers_in("4417") == ["4417"]


def test_a_number_inside_a_longer_one_is_not_in_the_email() -> None:
    # A plain `"441" in email` would find it inside 4417.
    assert email_fault(ticket(441), EMAIL) == (
        "order: 441 is not a number in the email"
    )


def test_a_number_the_email_gives_passes_the_email_rule() -> None:
    assert email_fault(ticket(4417), EMAIL) is None


def test_a_swapped_number_breaks_the_email_rule() -> None:
    assert email_fault(ticket(4471), EMAIL) == (
        "order: 4471 is not a number in the email"
    )


def test_no_order_number_has_nothing_to_check() -> None:
    assert email_fault(ticket(None), EMAIL) is None
    assert order_fault(ticket(None), {6083}) is None


def test_an_order_the_shop_has_passes() -> None:
    assert order_fault(ticket(4417), {4417, 6083}) is None


def test_an_order_the_shop_does_not_have_breaks_the_rule() -> None:
    assert order_fault(ticket(4471), {4417, 6083}) == (
        "order: 4471 is not in the shop's orders"
    )


def test_load_orders_turns_json_keys_into_numbers(tmp_path: Path) -> None:
    path = tmp_path / "orders.json"
    path.write_text(
        json.dumps({"4417": "mug", "6083": "lamp shade"}), encoding="utf-8"
    )
    assert load_orders(path) == {4417, 6083}
