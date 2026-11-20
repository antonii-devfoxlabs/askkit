# Writes samples/inbox.csv: 1,000 emails to the shop, made up for this
# course from the templates below. The names are invented and the
# order numbers belong to no real shop. Run it from the python folder:
#
#     uv run python samples/make_inbox.py
#
# The same seed gives the same file every time.

import csv
import random
from pathlib import Path

NAMES = [
    "Maria",
    "Tom",
    "Ines",
    "Kofi",
    "Yuki",
    "Lena",
    "Omar",
    "Priya",
    "Jonas",
    "Aiko",
    "Sam",
    "Rosa",
]
ITEMS = [
    "mug",
    "teapot",
    "lamp",
    "vase",
    "bowl",
    "candle holder",
    "plant pot",
    "set of plates",
]

DAMAGED = [
    "Order {order} came with a broken {item}. I'd like a new one.",
    "Hi, the {item} from order {order} arrived cracked, and the box was"
    " torn too. Can you send another?",
    "Order {order}: the {item} is not the one I ordered, it's blue, not"
    " green. Please swap it.",
    "The {item} in order {order} was in pieces when I opened it.\n"
    "Photos attached.\n{name}",
]
REFUND = [
    "I sent the {item} from order {order} back last week, as it was too"
    " big. When will I get my money back?",
    "Hello, I no longer want the {item} (order {order}). It arrived fine,"
    " but I'd like a refund, please.",
    "Order {order}: I returned the {item} on Monday. Could you refund it"
    " to my card?",
]
DELIVERY = [
    "Where is order {order}? It's been ten days, and the {item} is a gift.",
    "Hi, my {item} (order {order}) has not arrived yet. When will it come?",
    "Tracking for order {order} has not moved since Tuesday. Is my {item}"
    " lost?",
]
OTHER = [
    "Do you sell gift cards? I'd like to give one to a friend.",
    "Thank you, the {item} is lovely! {name}",
    "Can I change the delivery address for order {order}? I've moved,"
    " and the new one is 12 Elm Road.",
    "Hello, do you ship to Canada, and how long does it take?",
]
# A reply that quotes the shop's last email, as lesson 4.6's cleaning
# expects: the quoted lines are not the customer's new message.
QUOTED = (
    "\n\nOn 3 Nov 2026, the shop wrote:\n> Thank you for your order {order}."
    "\n> Your {item} is on its way."
)

TEMPLATES = [DAMAGED, REFUND, DELIVERY, OTHER]
# How often each kind comes, out of 11: broken items and late parcels
# most, as the shop's support team sees them.
WEIGHTS = [4, 2, 3, 2]


def main() -> None:
    chooser = random.Random(1059)
    path = Path(__file__).parent / "inbox.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "email"])
        for number in range(1, 1001):
            templates = chooser.choices(TEMPLATES, weights=WEIGHTS)[0]
            text = chooser.choice(templates)
            if chooser.random() < 0.1:
                text += QUOTED
            email = text.format(
                order=chooser.randint(4000, 4999),
                item=chooser.choice(ITEMS),
                name=chooser.choice(NAMES),
            )
            writer.writerow([f"e{number:04}", email])


if __name__ == "__main__":
    main()
