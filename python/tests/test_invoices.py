import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from test_classify import use_fake_client, with_answer
from test_label import printed_json

from askkit.cli import main
from askkit.invoices import (
    INVOICE_CHECKS,
    INVOICE_SCHEMA,
    Invoice,
    invoice_error,
)
from askkit.label import LABEL_KEY
from askkit.prompts import render

INVOICE = Path(__file__).parent.parent / "samples" / "invoice.pdf"

# What the sample invoice prints, as the model is asked to give it.
FIELDS: dict[str, Any] = {
    "seller": "Kilnworth Ceramics Ltd",
    "number": "KW-2026-0311",
    "issued": "2026-11-02",
    "due": "2026-12-02",
    "currency": "EUR",
    "lines": [
        {
            "item": "Stoneware mug, white, 350 ml",
            "quantity": 96,
            "unit_price": 4.35,
            "amount": 417.60,
        },
        {
            "item": "Teapot, blue glaze, 1 litre",
            "quantity": 24,
            "unit_price": 12.80,
            "amount": 307.20,
        },
        {
            "item": "Packing and pallet",
            "quantity": 1,
            "unit_price": 18.50,
            "amount": 18.50,
        },
    ],
    "subtotal": 743.30,
    "tax": 148.66,
    "total": 891.96,
}


def invoice_with(**changes: Any) -> Invoice:
    return Invoice.model_validate({**FIELDS, **changes})


def with_row(index: int, **changes: Any) -> dict[str, Any]:
    # The sample's fields with one row changed.
    lines = [dict(line) for line in FIELDS["lines"]]
    lines[index].update(changes)
    return {**FIELDS, "lines": lines}


def faults_of(fields: dict[str, Any]) -> str:
    with pytest.raises(ValidationError) as raised:
        Invoice.model_validate(fields)
    return invoice_error(raised.value)


@pytest.fixture
def keyed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")


def test_the_schema_holds_every_field_and_nothing_else() -> None:
    assert INVOICE_SCHEMA["additionalProperties"] is False
    assert INVOICE_SCHEMA["required"] == list(FIELDS)
    assert INVOICE_SCHEMA["properties"]["issued"]["format"] == "date"
    assert INVOICE_SCHEMA["properties"]["total"]["type"] == "number"


def test_the_rows_are_a_model_of_their_own_in_the_schema() -> None:
    lines = INVOICE_SCHEMA["properties"]["lines"]
    assert lines["type"] == "array"
    assert lines["minItems"] == 1
    assert lines["items"] == {"$ref": "#/$defs/InvoiceLine"}
    row = INVOICE_SCHEMA["$defs"]["InvoiceLine"]
    assert row["additionalProperties"] is False
    assert row["required"] == ["item", "quantity", "unit_price", "amount"]


def test_every_field_has_a_description_the_model_reads() -> None:
    row = INVOICE_SCHEMA["$defs"]["InvoiceLine"]["properties"]
    fields = INVOICE_SCHEMA["properties"]
    for name in FIELDS:
        assert fields[name]["description"], name
    for name in FIELDS["lines"][0]:
        assert row[name]["description"], name


def test_limits_the_api_does_not_hold_are_written_into_the_description() -> (
    None
):
    # The SDK takes gt=0 and the pattern out of the schema it sends and
    # adds them to the description; Invoice still checks both.
    row = INVOICE_SCHEMA["$defs"]["InvoiceLine"]["properties"]
    assert "exclusiveMinimum" not in row["quantity"]
    assert row["quantity"]["description"].endswith("{exclusiveMinimum: 0}")
    currency = INVOICE_SCHEMA["properties"]["currency"]
    assert "pattern" not in currency
    assert currency["description"].endswith("{pattern: ^[A-Z]{3}$}")


def test_the_template_leaves_the_fields_to_their_descriptions() -> None:
    text = render("read_invoice", {})
    for name in FIELDS:
        assert f"- {name}: " not in text
    assert "description" in text


def test_the_sample_invoice_passes_every_check() -> None:
    # 743.30 + 148.66 is 891.9599999999999 as floats: cents are compared.
    assert 743.30 + 148.66 != 891.96
    invoice = invoice_with()
    assert len(invoice.lines) == 3
    assert invoice.lines[1].item == "Teapot, blue glaze, 1 litre"


def test_a_row_that_is_not_quantity_times_price_names_its_row() -> None:
    assert faults_of(with_row(1, amount=370.20)) == (
        "the invoice does not pass its checks: lines.1: 24 x 12.8 is"
        " 307.20, not 370.20; a person checks the invoice"
    )


def test_an_amount_in_part_cents_is_refused_not_rounded() -> None:
    assert faults_of(with_row(0, amount=417.604)) == (
        "the invoice does not pass its checks: lines.0.amount: 417.604 is"
        " not an amount in whole cents; a person checks the invoice"
    )
    assert faults_of({**FIELDS, "tax": 148.655}) == (
        "the invoice does not pass its checks: tax: 148.655 is not an"
        " amount in whole cents; a person checks the invoice"
    )


def test_a_quantity_of_none_is_refused() -> None:
    assert faults_of(with_row(2, quantity=0)) == (
        "the invoice does not pass its checks: lines.2.quantity: Input"
        " should be greater than 0, not 0; a person checks the invoice"
    )


def test_an_invoice_with_no_rows_is_refused() -> None:
    assert faults_of({**FIELDS, "lines": []}).startswith(
        "the invoice does not pass its checks: lines: List should have at"
        " least 1 item after validation"
    )


def test_a_currency_code_in_small_letters_is_refused() -> None:
    assert faults_of({**FIELDS, "currency": "eur"}).startswith(
        "the invoice does not pass its checks: currency: String should"
        " match pattern"
    )


def test_rows_that_do_not_add_up_to_the_subtotal_are_a_fault() -> None:
    assert faults_of({**FIELDS, "subtotal": 734.30, "total": 882.96}) == (
        "the invoice does not pass its checks: answer: the rows add up to"
        " 743.30, not the subtotal 734.30; a person checks the invoice"
    )


def test_every_sum_that_disagrees_is_named_at_once() -> None:
    # One validator collects every rule, so a wrong subtotal does not hide
    # a due date before the issue date.
    assert faults_of({**FIELDS, "total": 819.96, "due": "2026-10-02"}) == (
        "the invoice does not pass its checks: answer: subtotal 743.30 and"
        " tax 148.66 do not add up to the total 819.96; due 2026-10-02 is"
        " before issued 2026-11-02; a person checks the invoice"
    )


def test_the_sums_are_not_checked_while_a_row_is_wrong() -> None:
    # Pydantic checks inside out: the invoice's own rules run only once
    # every row has passed, so only the row is named.
    fields = with_row(1, amount=370.20)
    fields["total"] = 819.96
    assert faults_of(fields) == (
        "the invoice does not pass its checks: lines.1: 24 x 12.8 is"
        " 307.20, not 370.20; a person checks the invoice"
    )


def test_no_due_date_is_allowed() -> None:
    assert invoice_with(due=None).due is None


def test_a_date_that_is_not_a_date_breaks_the_model() -> None:
    assert faults_of({**FIELDS, "issued": "2 November 2026"}).startswith(
        "the invoice does not pass its checks: issued: Input should be a"
        " valid date"
    )


def test_read_sends_the_pdf_alone_with_the_schema(
    keyed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent, with_answer(json.dumps(FIELDS)))
    main(["read", str(INVOICE)])
    request = sent[0]
    assert request["system"] == render("read_invoice", {})
    assert request["stream"] is True
    assert request["output_config"] == {
        "format": {"type": "json_schema", "schema": INVOICE_SCHEMA}
    }
    [message] = request["messages"]
    [block] = message["content"]
    assert block["type"] == "document"
    assert block["source"]["media_type"] == "application/pdf"


def test_read_prints_the_invoice_and_the_call(
    keyed: None,
    labelled_now: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    use_fake_client(monkeypatch, [], with_answer(json.dumps(FIELDS)))
    main(["read", str(INVOICE)])
    invoice, lines = printed_json(capsys.readouterr().out)
    assert invoice.pop(LABEL_KEY) == {
        "model": "claude-haiku-5-5",
        "via": "askkit read",
        "written": "2026-11-21T09:30:00Z",
        "checked_by_code": INVOICE_CHECKS,
        "read_by_a_person": False,
    }
    assert invoice == FIELDS
    assert lines == [
        "",
        "Model:         claude-haiku-5-5",
        "Tokens:        214 in, 12 out",
        "Cost:          $0.000027",
        "Stop reason:   end_turn",
        "Time:          first token 0.42 s, total 0.84 s",
    ]
    assert isinstance(invoice["issued"], str)
    assert date.fromisoformat(invoice["issued"])


def test_sums_that_disagree_go_to_a_person(
    keyed: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    misread = json.dumps({**FIELDS, "total": 819.96})
    use_fake_client(monkeypatch, [], with_answer(misread))
    with pytest.raises(SystemExit) as stopped:
        main(["read", str(INVOICE)])
    assert stopped.value.code == (
        "askkit: the invoice does not pass its checks: answer: subtotal"
        " 743.30 and tax 148.66 do not add up to the total 819.96; a person"
        " checks the invoice"
    )
    assert '"seller"' not in capsys.readouterr().out


def test_an_answer_that_breaks_the_model_is_not_used(
    keyed: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    broken = json.dumps({**FIELDS, "issued": "2 November 2026"})
    use_fake_client(monkeypatch, [], with_answer(broken))
    with pytest.raises(SystemExit) as stopped:
        main(["read", str(INVOICE)])
    assert str(stopped.value.code).startswith(
        "askkit: the invoice does not pass its checks: issued:"
    )
    assert '"seller"' not in capsys.readouterr().out


def test_a_cut_off_answer_is_not_read(
    keyed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_fake_client(
        monkeypatch, [], with_answer('{"seller":"Kiln', "max_tokens")
    )
    with pytest.raises(SystemExit) as stopped:
        main(["read", str(INVOICE), "--max-tokens", "12"])
    assert stopped.value.code == (
        "askkit: the answer was cut off at 12 tokens;"
        " give it a larger --max-tokens"
    )


def test_a_file_that_is_no_pdf_stops_before_a_call(
    keyed: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    word = tmp_path / "invoice.docx"
    word.write_bytes(b"PK\x03\x04\x14\x00\x06\x00")
    with pytest.raises(SystemExit) as stopped:
        main(["read", str(word)])
    assert stopped.value.code == f"askkit: {word} is not a PDF"
    assert sent == []


def test_a_missing_file_stops_before_a_call(
    keyed: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    with pytest.raises(SystemExit) as stopped:
        main(["read", "missing.pdf"])
    assert stopped.value.code == (
        "askkit: cannot read missing.pdf: No such file or directory"
    )
    assert sent == []


def test_ask_refuses_a_pdf_as_a_text_document(
    keyed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Read as text, a PDF failed with a UnicodeDecodeError traceback on its
    # first bytes that are not UTF-8.
    sent: list[dict[str, Any]] = []
    use_fake_client(monkeypatch, sent)
    with pytest.raises(SystemExit) as stopped:
        main(["ask", "When is it due?", "--document", str(INVOICE)])
    assert stopped.value.code == (
        f"askkit: {INVOICE} is a PDF, and --document takes text;"
        " read an invoice with askkit read"
    )
    assert sent == []
