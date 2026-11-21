from datetime import date
from typing import Any, Self

from anthropic import transform_schema
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from askkit.extract import ticket_faults


def cents(amount: float) -> int:
    # A float stores most decimals only nearly, so amounts are compared
    # as whole cents, which are exact.
    return round(amount * 100)


def in_whole_cents(amount: float) -> float:
    # Money on an invoice is printed to the cent. An amount with more
    # decimals was not copied from the page, so it is refused rather
    # than rounded: rounding would hide the misreading.
    if round(amount, 2) != amount:
        raise ValueError(f"{amount} is not an amount in whole cents")
    return amount


# One row of the invoice's item table. A model of its own, so every row
# is checked the same way, and a fault names the row it is in.
class InvoiceLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: str = Field(
        description="The item as the invoice names it, such as"
        ' "Teapot, blue glaze, 1 litre".'
    )
    # gt=0 is a limit the API does not hold: the SDK writes it into the
    # description, and this class checks it.
    quantity: int = Field(gt=0, description="How many were billed.")
    unit_price: float = Field(description="The price of one, as printed.")
    amount: float = Field(
        description="The row's amount, as printed. Never multiply it"
        " out yourself: the shop's code checks it."
    )

    @field_validator("amount")
    @classmethod
    def amount_in_whole_cents(cls, amount: float) -> float:
        return in_whole_cents(amount)

    @model_validator(mode="after")
    def amount_is_quantity_times_price(self) -> Self:
        expected = self.quantity * self.unit_price
        if cents(expected) != cents(self.amount):
            raise ValueError(
                f"{self.quantity} x {self.unit_price} is {expected:.2f},"
                f" not {self.amount:.2f}"
            )
        return self


# The fields of a supplier's invoice that the shop's books need. One class
# is the schema sent with the call and the check of the answer, as Ticket
# is. Only the due date has a way out: an invoice without a seller, a
# number, a date, its rows or a total is not an invoice. What each field
# means is in its description, which goes with the schema to the model.
class Invoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seller: str = Field(
        description="The name of the company that sent the invoice."
    )
    number: str = Field(description="The invoice number, exactly as printed.")
    issued: date = Field(description="The date the invoice was issued.")
    due: date | None = Field(
        description="The date payment is due; null if the invoice prints"
        " no due date."
    )
    currency: str = Field(
        pattern=r"^[A-Z]{3}$",
        description="The three-letter code of the currency, such as EUR.",
    )
    lines: list[InvoiceLine] = Field(
        min_length=1,
        description="Every row of the item table, in the order printed,"
        " packing and delivery rows included.",
    )
    subtotal: float = Field(description="The amount before tax.")
    tax: float = Field(
        description="The tax amount, such as VAT; 0 if the invoice prints"
        " no tax."
    )
    total: float = Field(description="The amount to pay, tax included.")

    @field_validator("subtotal", "tax", "total")
    @classmethod
    def total_in_whole_cents(cls, amount: float) -> float:
        return in_whole_cents(amount)

    @model_validator(mode="after")
    def sums_and_dates_agree(self) -> Self:
        # Rules across fields, run only once every field and every row
        # has passed. One validator collects them all, so a fault in one
        # sum does not hide a fault in another.
        faults: list[str] = []
        rows = 0
        for line in self.lines:
            rows += cents(line.amount)
        if rows != cents(self.subtotal):
            faults.append(
                f"the rows add up to {rows / 100:.2f},"
                f" not the subtotal {self.subtotal:.2f}"
            )
        if cents(self.subtotal) + cents(self.tax) != cents(self.total):
            faults.append(
                f"subtotal {self.subtotal:.2f} and tax {self.tax:.2f}"
                f" do not add up to the total {self.total:.2f}"
            )
        if self.due is not None and self.due < self.issued:
            faults.append(f"due {self.due} is before issued {self.issued}")
        if faults:
            raise ValueError("; ".join(faults))
        return self


INVOICE_SCHEMA: dict[str, Any] = transform_schema(Invoice)

# What Invoice checked, for an invoice's label (label.py). An invoice is
# printed only when all of them hold.
INVOICE_CHECKS = [
    "types and formats",
    "amounts in whole cents",
    "each row: quantity x unit price = amount",
    "rows add up to the subtotal",
    "subtotal + tax = total",
    "due date not before issued",
]


def invoice_error(error: ValidationError) -> str:
    # ticket_faults reads any ValidationError, one part per fault, with
    # its place inside the invoice ("lines.1.amount"). A misread number
    # and an invoice that is wrong itself look the same here, so a
    # person reads the PDF either way.
    return (
        "the invoice does not pass its checks: "
        + ticket_faults(error)
        + "; a person checks the invoice"
    )
