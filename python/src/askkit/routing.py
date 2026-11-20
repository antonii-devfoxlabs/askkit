from pydantic import ValidationError

from askkit.extract import check_ticket, ticket_faults
from askkit.models import DEFAULT_MODEL
from askkit.rules import email_fault

# The two models a ticket can go to: the small one first, every time,
# and the large one only for an email the small one got visibly wrong.
# Claude Sonnet 5.5 costs twenty times Claude Haiku 5.5 a token
# (prices.py), so it is worth paying only where it changes the answer.
SMALL_MODEL = DEFAULT_MODEL
LARGE_MODEL = "claude-sonnet-5-5"


def call_model(attempt: int, model: str, escalate_to: str | None) -> str:
    # The first call goes to the model asked for. A call after it is a
    # retry, made because the answer failed a check: it goes to
    # escalate_to when one is given, or to the same model again.
    if attempt == 0 or escalate_to is None:
        return model
    return escalate_to


def escalation_fault(answer: str, email: str) -> str | None:
    # The faults that send a ticket to a second call, as extract finds
    # them: the answer breaks the Ticket contract, or names an order
    # number the email does not give. None when the ticket passes. A
    # ticket that passes but is wrong, such as one that names the least
    # serious of three problems, is not caught here: no code can see it.
    try:
        ticket, _ = check_ticket(answer)
    except ValidationError as error:
        return ticket_faults(error)
    return email_fault(ticket, email)
