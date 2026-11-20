import math
from dataclasses import dataclass
from pathlib import Path

from askkit.images import (
    SQUARE,
    image_fault,
    image_size,
    image_tokens,
    media_type_of,
    resized_size,
)
from askkit.pdfs import PAGE_IMAGE_TOKENS, page_texts, pdf_fault
from askkit.tokens import count_tokens


@dataclass
class InputTokens:
    # What the file is, the tokens Claude reads from it, and how AskKit
    # got that number: printed on the Input and Input tokens lines.
    what: str
    tokens: int
    how: str


def image_input(path: Path, data: bytes) -> InputTokens:
    fault = image_fault(path, data)
    if fault is not None:
        raise ValueError(fault)
    width, height = image_size(data)
    media_type = media_type_of(data) or ""
    what = f"{media_type.removeprefix('image/').upper()} image"
    what += f", {width:,} x {height:,} px"
    sent_width, sent_height = resized_size(width, height)
    if (sent_width, sent_height) != (width, height):
        # The API shrinks it and bills the smaller size.
        what += f", shrunk to {sent_width:,} x {sent_height:,}"
    across = math.ceil(sent_width / SQUARE)
    down = math.ceil(sent_height / SQUARE)
    return InputTokens(
        what,
        image_tokens(sent_width, sent_height),
        f"{across} x {down} squares of 28 x 28 px",
    )


def pdf_input(
    path: Path, data: bytes, model: str, window: int
) -> InputTokens:
    texts = page_texts(data)
    fault = pdf_fault(path, data, len(texts), window)
    if fault is not None:
        raise ValueError(fault)
    text = "\n".join(texts)
    text_tokens, _ = count_tokens(model, text)
    # Each page's image costs at most this much: the API draws it at a
    # size it chooses, so AskKit can give the most, not the exact count.
    image_tokens = PAGE_IMAGE_TOKENS * len(texts)
    pages = "page" if len(texts) == 1 else "pages"
    return InputTokens(
        f"PDF, {len(texts)} {pages}, {len(text):,} characters of text",
        text_tokens + image_tokens,
        f"at most: page images {image_tokens:,}, text {text_tokens:,}",
    )
