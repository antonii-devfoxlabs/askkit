import base64
from pathlib import Path
from typing import Any

import pytest

from askkit.pdfs import document_block, is_pdf

# Made for the course: a two-page invoice from an invented company.
INVOICE = Path(__file__).parent.parent / "samples" / "invoice.pdf"


def test_the_sample_invoice_is_a_pdf() -> None:
    data = INVOICE.read_bytes()
    assert data.startswith(b"%PDF-1.7\n")
    assert is_pdf(data)


@pytest.mark.parametrize(
    "start",
    [
        b"Order 4417 came with a broken mug.",
        # A Word file is a ZIP archive: it starts with PK.
        b"PK\x03\x04\x14\x00\x06\x00",
        b"\x89PNG\r\n\x1a\n\x00\x00",
        # The signature has to be the very first bytes.
        b" %PDF-1.7\n",
        b"",
    ],
    ids=["text", "docx", "png", "not-first", "empty"],
)
def test_anything_else_is_not_a_pdf(start: bytes) -> None:
    assert not is_pdf(start)


def test_a_pdf_goes_whole_as_base64_text() -> None:
    block = document_block(INVOICE)
    assert block["type"] == "document"
    source: Any = block["source"]
    assert source["type"] == "base64"
    assert source["media_type"] == "application/pdf"
    assert isinstance(source["data"], str)
    assert base64.b64decode(source["data"]) == INVOICE.read_bytes()


def test_the_bytes_decide_not_the_name(tmp_path: Path) -> None:
    named_pdf = tmp_path / "invoice.pdf"
    named_pdf.write_bytes(b"PK\x03\x04\x14\x00\x06\x00")
    with pytest.raises(ValueError) as caught:
        document_block(named_pdf)
    assert str(caught.value) == f"{named_pdf} is not a PDF"


def test_a_missing_file_raises_the_system_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        document_block(tmp_path / "missing.pdf")
