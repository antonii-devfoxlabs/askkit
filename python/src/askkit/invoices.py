from datetime import date
from typing import Any

from anthropic import transform_schema
from pydantic import BaseModel, ConfigDict, ValidationError

from askkit.extract import ticket_faults


# The fields of a supplier's invoice that the shop's books need. One class
# is the schema sent with the call and the check of the answer, as Ticket
# is. Only the due date has a way out: an invoice without a seller, a
# number, a date or a total is not an invoice.
class Invoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seller: str
    number: str
    issued: date
    due: date | None
    currency: str
    subtotal: float
    tax: float
    total: float


INVOICE_SCHEMA: dict[str, Any] = transform_schema(Invoice)

# What Invoice and invoice_faults checked, for an invoice's label
# (label.py). An invoice is printed only when all three hold.
INVOICE_CHECKS = [
    "types",
    "subtotal + tax = total",
    "due date not before issued",
]


def cents(amount: float) -> int:
    # A float stores most decimals only nearly, so amounts are compared
    # as whole cents, which are exact.
    return round(amount * 100)


def invoice_faults(invoice: Invoice) -> list[str]:
    # Rules against the invoice itself, after Invoice has passed: the
    # schema holds each field's type, not how the fields agree.
    faults: list[str] = []
    if cents(invoice.subtotal) + cents(invoice.tax) != cents(invoice.total):
        faults.append(
            f"subtotal {invoice.subtotal:.2f} and tax {invoice.tax:.2f}"
            f" do not add up to the total {invoice.total:.2f}"
        )
    if invoice.due is not None and invoice.due < invoice.issued:
        faults.append(f"due {invoice.due} is before issued {invoice.issued}")
    return faults


def invoice_error(error: ValidationError) -> str:
    # ticket_faults reads any ValidationError, one part per field.
    return "the invoice does not match its model: " + ticket_faults(error)
