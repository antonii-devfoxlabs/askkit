from askkit.classify import Queue
from askkit.clean import clean_email
from askkit.sorting_set import SortingCase

# The sorting job without a model: lesson 1.8's keyword rules, tried in
# this order, the first one found deciding. Six are 1.8's own; the rest
# were added after reading the rules' misses on the main part of the
# sorting set, never the held-out part: the same chance a prompt gets,
# so the model is measured against an honest try, not a weak one.
# Damage comes first, because "broken ... I'd like a refund" is a
# damaged item (case s23).
KEYWORDS: dict[str, Queue] = {
    "broken": "damaged",
    "damaged": "damaged",
    "cracked": "damaged",
    "chipped": "damaged",
    "in pieces": "damaged",
    "does not work": "damaged",
    "refund": "refund",
    "money back": "refund",
    "it back": "refund",
    "where is": "delivery",
    "tracking": "delivery",
    "hasn't come": "delivery",
    "has not come": "delivery",
    "nothing has come": "delivery",
}


def keyword_queue(email: str) -> Queue:
    # The same cleaned email the model gets (lesson 4.6), so the two are
    # measured on the same text: quoted lines are not the new message.
    cleaned, _ = clean_email(email)
    lowered = cleaned.lower()
    for keyword in KEYWORDS:
        if keyword in lowered:
            return KEYWORDS[keyword]
    return "other"


def keyword_misses(
    cases: list[tuple[SortingCase, str]],
) -> list[tuple[SortingCase, Queue]]:
    # Every case the rules put in the wrong queue, with the queue they
    # chose. The right ones are len(cases) minus these.
    misses: list[tuple[SortingCase, Queue]] = []
    for case, email in cases:
        queue = keyword_queue(email)
        if queue != case.queue:
            misses.append((case, queue))
    return misses


def score_text(
    cases: list[tuple[SortingCase, str]],
    misses: list[tuple[SortingCase, Queue]],
) -> str:
    right = len(cases) - len(misses)
    return f"{right} of {len(cases)} right ({right / len(cases):.0%})"
