import base64
from pathlib import Path

from anthropic.types import DocumentBlockParam

# Every PDF starts with these five bytes, whatever the file is called.
PDF_SIGNATURE = b"%PDF-"


def is_pdf(data: bytes) -> bool:
    return data.startswith(PDF_SIGNATURE)


def document_block(path: Path) -> DocumentBlockParam:
    # The whole file goes in the request: the API turns each page into an
    # image and takes out its text, so the model gets both.
    data = path.read_bytes()
    if not is_pdf(data):
        raise ValueError(f"{path} is not a PDF")
    return {
        "type": "document",
        "source": {
            "type": "base64",
            "media_type": "application/pdf",
            "data": base64.b64encode(data).decode("ascii"),
        },
    }
