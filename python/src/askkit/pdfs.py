import base64
import io
from pathlib import Path

from anthropic.types import DocumentBlockParam
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from askkit.images import base64_length

# Every PDF starts with these five bytes, whatever the file is called.
PDF_SIGNATURE = b"%PDF-"


def is_pdf(data: bytes) -> bool:
    return data.startswith(PDF_SIGNATURE)


# From Anthropic's PDF support page, checked on 20 November 2026: a
# request holds at most 600 pages (100 when its context window is under
# 1,000,000 tokens) and 128 MB. Each page's image costs at most 1,568
# tokens, on every model; its text is billed as text on top.
MAX_PAGES = 600
MAX_PAGES_SMALL_WINDOW = 100
MAX_REQUEST_BYTES = 128 * 1024 * 1024
PAGE_IMAGE_TOKENS = 1568


def page_texts(data: bytes) -> list[str]:
    # pypdf takes out the text a program can read, page by page; a page
    # that is only a scan gives an empty string.
    reader = PdfReader(io.BytesIO(data))
    return [page.extract_text() for page in reader.pages]


def pdf_fault(path: Path, data: bytes, pages: int, window: int) -> str | None:
    most = MAX_PAGES if window >= 1_000_000 else MAX_PAGES_SMALL_WINDOW
    if pages > most:
        return (
            f"{path} has {pages:,} pages, over the {most} a request may"
            " hold; send it in parts"
        )
    if base64_length(data) > MAX_REQUEST_BYTES:
        megabytes = base64_length(data) / 1024 / 1024
        return (
            f"{path} is {megabytes:.1f} MB as base64, over the 128 MB a"
            " request may be; send it in parts"
        )
    return None


def document_block(path: Path, window: int) -> DocumentBlockParam:
    # The whole file goes in the request: the API turns each page into an
    # image and takes out its text, so the model gets both.
    data = path.read_bytes()
    if not is_pdf(data):
        raise ValueError(f"{path} is not a PDF")
    try:
        pages = len(page_texts(data))
    except PdfReadError as error:
        message = f"{path} starts like a PDF, but cannot be read"
        raise ValueError(message) from error
    fault = pdf_fault(path, data, pages, window)
    if fault is not None:
        raise ValueError(fault)
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(data).decode("ascii"),
        },
    }
