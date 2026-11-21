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
    invoice_faults,
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
    "subtotal": 743.30,
    "tax": 148.66,
    "total": 891.96,
}


def invoice_with(**changes: Any) -> Invoice:
    return Invoice.model_validate({**FIELDS, **changes})


@pytest.fixture
def keyed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-a-test-key-Ab12")


def test_the_schema_holds_every_field_and_nothing_else() -> None:
    assert INVOICE_SCHEMA["additionalProperties"] is False
    assert INVOICE_SCHEMA["required"] == list(FIELDS)
    assert INVOICE_SCHEMA["properties"]["issued"]["format"] == "date"
    assert INVOICE_SCHEMA["properties"]["total"]["type"] == "number"


def test_every_field_is_defined_in_the_template() -> None:
    text = render("read_invoice", {})
    for name in FIELDS:
        assert f"- {name}: " in text


def test_the_sample_invoice_adds_up() -> None:
    # 743.30 + 148.66 is 891.9599999999999 as floats: cents are compared.
    assert 743.30 + 148.66 != 891.96
    assert invoice_faults(invoice_with()) == []


def test_a_total_that_does_not_add_up_is_a_fault() -> None:
    assert invoice_faults(invoice_with(total=819.96)) == [
        "subtotal 743.30 and tax 148.66 do not add up to the total 819.96"
    ]


def test_a_due_date_before_the_issue_date_is_a_fault() -> None:
    assert invoice_faults(invoice_with(due="2026-10-02")) == [
        "due 2026-10-02 is before issued 2026-11-02"
    ]


def test_no_due_date_is_allowed() -> None:
    invoice = invoice_with(due=None)
    assert invoice.due is None
    assert invoice_faults(invoice) == []


def test_a_date_that_is_not_a_date_breaks_the_model() -> None:
    with pytest.raises(ValidationError) as raised:
        invoice_with(issued="2 November 2026")
    assert invoice_error(raised.value).startswith(
        "the invoice does not match its model: issued: Input should be a"
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
        "askkit: subtotal 743.30 and tax 148.66 do not add up to the total"
        " 819.96; a person checks the invoice"
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
        "askkit: the invoice does not match its model: issued:"
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
