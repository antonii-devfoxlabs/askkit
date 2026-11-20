import base64
import io
from pathlib import Path
from typing import Any

import pytest
from pypdf import PdfWriter

from askkit.pdfs import document_block, is_pdf, page_texts, pdf_fault

# Made for the course: a two-page invoice from an invented company.
INVOICE = Path(__file__).parent.parent / "samples" / "invoice.pdf"
# Claude Haiku 5.5's context window.
WINDOW = 1_000_000


def blank_pdf(pages: int) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=595, height=842)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


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
    block = document_block(INVOICE, WINDOW)
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
        document_block(named_pdf, WINDOW)
    assert str(caught.value) == f"{named_pdf} is not a PDF"


def test_a_missing_file_raises_the_system_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        document_block(tmp_path / "missing.pdf", WINDOW)


def test_each_page_gives_its_text() -> None:
    texts = page_texts(INVOICE.read_bytes())
    assert len(texts) == 2
    assert "KW-2026-0311" in texts[0]
    assert "SAMPLE INVOICE" in texts[1]


def test_a_page_without_text_gives_an_empty_string() -> None:
    assert page_texts(blank_pdf(1)) == [""]


def test_600_pages_fit_a_large_window(tmp_path: Path) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(blank_pdf(600))
    assert document_block(path, WINDOW)["type"] == "document"


def test_601_pages_are_refused_before_any_call(tmp_path: Path) -> None:
    path = tmp_path / "report.pdf"
    path.write_bytes(blank_pdf(601))
    with pytest.raises(ValueError) as caught:
        document_block(path, WINDOW)
    assert str(caught.value) == (
        f"{path} has 601 pages, over the 600 a request may hold;"
        " send it in parts"
    )


def test_a_window_under_a_million_takes_100_pages() -> None:
    path = Path("report.pdf")
    assert pdf_fault(path, b"%PDF-", 100, 200_000) is None
    assert pdf_fault(path, b"%PDF-", 101, 200_000) == (
        "report.pdf has 101 pages, over the 100 a request may hold;"
        " send it in parts"
    )


def test_over_128_mb_of_base64_is_refused() -> None:
    # 96 MiB of bytes are 128 MiB of base64; one byte more is too much.
    largest = b"%PDF-" + bytes(96 * 1024 * 1024 - 5)
    assert pdf_fault(Path("scan.pdf"), largest, 1, WINDOW) is None
    assert pdf_fault(Path("scan.pdf"), largest + b"\x00", 1, WINDOW) == (
        "scan.pdf is 128.0 MB as base64, over the 128 MB a request may"
        " be; send it in parts"
    )


def test_a_broken_pdf_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "invoice.pdf"
    path.write_bytes(b"%PDF-1.7\nnot really a PDF")
    with pytest.raises(ValueError) as caught:
        document_block(path, WINDOW)
    assert (
        str(caught.value) == f"{path} starts like a PDF, but cannot be read"
    )
