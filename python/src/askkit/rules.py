import json
from pathlib import Path

from askkit.extract import Ticket


def numbers_in(text: str) -> list[str]:
    # Every run of digits in the text, whole: "order 4417, 2 mugs" gives
    # ["4417", "2"]. Whole runs, so 441 is not found in an email that
    # says 4417.
    numbers: list[str] = []
    digits = ""
    for character in text + " ":
        if character.isdigit():
            digits += character
        elif digits:
            numbers.append(digits)
            digits = ""
    return numbers


def load_orders(path: Path) -> set[int]:
    # The shop's orders as the warehouse exports them: a JSON object whose
    # keys are order numbers. JSON keys are always text, so each one is
    # turned into a whole number, the type Ticket gives `order`.
    orders: set[int] = set()
    for number in json.loads(path.read_text(encoding="utf-8")):
        orders.add(int(number))
    return orders


def email_fault(ticket: Ticket, email: str) -> str | None:
    # The model's fault: an order number the email does not give. The
    # model can read the email again, so this fault is asked again.
    if ticket.order is None or str(ticket.order) in numbers_in(email):
        return None
    return f"order: {ticket.order} is not a number in the email"


def order_fault(ticket: Ticket, orders: set[int]) -> str | None:
    # A number the email gives but the shop does not have: a typo of the
    # customer's, or orders that are out of date. Asking again returns
    # the same number, so only a person can settle it.
    if ticket.order is None or ticket.order in orders:
        return None
    return f"order: {ticket.order} is not in the shop's orders"
