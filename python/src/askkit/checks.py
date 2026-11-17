import re
from pathlib import Path

from anthropic import Anthropic
from pydantic import ValidationError

from askkit.ask import answer_text, stop_warning
from askkit.clean import clean_email
from askkit.extract import check_ticket, extract_ticket, ticket_faults
from askkit.golden import Case, email_text

# Four digits in a row: an order number. The warehouse finds a ticket by
# its order field, so the number said again in the problem only takes
# room from what went wrong. Other numbers, such as the 2 in "2 plates
# cracked", are part of the problem and pass.
ORDER_NUMBER = r"\d{4}"


def case_faults(answer: str, case: Case) -> list[str]:
    # Every way an answer fails its case, one fault each; no fault is a
    # pass. The schema check comes first, the same one extract runs,
    # repair included, so the eval scores the ticket AskKit would print.
    # An answer that is not a Ticket has no fields to check further.
    try:
        ticket, _ = check_ticket(answer)
    except ValidationError as error:
        return [ticket_faults(error)]
    faults: list[str] = []
    if ticket.order != case.order:
        faults.append(f"order: {ticket.order!r}, not {case.order!r}")
    if ticket.wish != case.wish:
        faults.append(f"wish: {ticket.wish!r}, not {case.wish!r}")
    for pattern in case.mentions:
        if re.search(pattern, ticket.problem, re.IGNORECASE) is None:
            faults.append(f"problem: no {pattern!r} in {ticket.problem!r}")
    if re.search(ORDER_NUMBER, ticket.problem):
        faults.append(f"problem: an order number in {ticket.problem!r}")
    return faults


def run_case(
    client: Anthropic, folder: Path, case: Case, model: str, max_tokens: int
) -> list[str]:
    # One case of the golden set through the steps extract takes on its
    # first call: the cleaned email, the prompt in use, the stop reason
    # read before the JSON. No retry: the eval scores the prompt, and a
    # second call would hide the first answer's faults.
    email, _ = clean_email(email_text(folder, case))
    response = extract_ticket(client, email, model, max_tokens)
    warning = stop_warning(response, "turn this email into a ticket")
    if warning is not None:
        return [f"stop: {warning}"]
    return case_faults(answer_text(response), case)
