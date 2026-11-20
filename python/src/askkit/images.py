import base64
import io
import math
from pathlib import Path
from typing import Literal

from anthropic.types import ImageBlockParam, TextBlockParam
from PIL import Image

# The four image formats Anthropic's API reads, as the API names them.
MediaType = Literal["image/png", "image/jpeg", "image/gif", "image/webp"]

# Every file of these formats starts with the same few bytes. A file's
# name can say .png and hold a JPEG; its first bytes say what it is.
SIGNATURES: dict[bytes, MediaType] = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
}


def media_type_of(data: bytes) -> MediaType | None:
    for signature, media_type in SIGNATURES.items():
        if data.startswith(signature):
            return media_type
    # A WebP file starts with RIFF, four bytes of its size, then WEBP.
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


# From Anthropic's vision guide, checked on 20 November 2026. Claude reads
# an image in squares of 28 x 28 pixels, one token each. Claude 4.7 and
# later models, every Claude model AskKit knows, are on the guide's
# high-resolution tier: an image with a side over 2,576 px, or one that
# would cost more than 4,784 tokens, is shrunk until it fits both.
SQUARE = 28
MAX_EDGE = 2576
MAX_IMAGE_TOKENS = 4784
# Larger than these, the API rejects an image instead of shrinking it.
LARGEST_SIDE = 8000
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # of base64 text, not of the file


def image_tokens(width: int, height: int) -> int:
    return math.ceil(width / SQUARE) * math.ceil(height / SQUARE)


def fits(width: int, height: int) -> bool:
    # A side counts as whole squares: the API pads the last one.
    return (
        math.ceil(width / SQUARE) * SQUARE <= MAX_EDGE
        and math.ceil(height / SQUARE) * SQUARE <= MAX_EDGE
        and image_tokens(width, height) <= MAX_IMAGE_TOKENS
    )


def resized_size(width: int, height: int) -> tuple[int, int]:
    # The size the API shrinks an image to, from the reference code in
    # Anthropic's guide "Coordinates and bounding boxes": the largest
    # size with the same proportions that fits.
    if fits(width, height):
        return (width, height)
    if height > width:
        short, long = resized_size(height, width)
        return (long, short)
    proportion = width / height
    smallest, largest = 1, width  # smallest always fits, largest never
    while smallest + 1 < largest:
        middle = (smallest + largest) // 2
        if fits(middle, max(round(middle / proportion), 1)):
            smallest = middle
        else:
            largest = middle
    return (smallest, max(round(smallest / proportion), 1))


# Pillow refuses to open an image of more than about 179 million pixels,
# in case unpacking it fills the memory. AskKit reads only the size, never
# the pixels, and must open such an image to say it is too large.
Image.MAX_IMAGE_PIXELS = None


def image_size(data: bytes) -> tuple[int, int]:
    # Pillow reads the width and height from the file's header; it does
    # not unpack the pixels to answer.
    with Image.open(io.BytesIO(data)) as image:
        return image.size


def base64_length(data: bytes) -> int:
    # Base64 writes every 3 bytes, the last ones padded, as 4 characters.
    return 4 * math.ceil(len(data) / 3)


def image_fault(path: Path, data: bytes) -> str | None:
    # Checked before the call: the API would reject both, after the whole
    # image had gone over the network.
    try:
        width, height = image_size(data)
    except OSError:
        # Pillow's UnidentifiedImageError is an OSError. Its first bytes
        # say it is an image, but its header is broken: the API could not
        # read it either.
        return f"{path} starts like an image, but its size cannot be read"
    if max(width, height) > LARGEST_SIDE:
        return (
            f"{path} is {width:,} x {height:,} px, over the {LARGEST_SIDE:,}"
            " px a side the API takes; make it smaller"
        )
    if base64_length(data) > MAX_IMAGE_BYTES:
        megabytes = base64_length(data) / 1024 / 1024
        return (
            f"{path} is {megabytes:.1f} MB as base64, over the 10 MB an"
            " image may be; save it as JPEG, or make it smaller"
        )
    return None


def image_block(path: Path) -> ImageBlockParam:
    data = path.read_bytes()
    media_type = media_type_of(data)
    if media_type is None:
        raise ValueError(f"{path} is not a PNG, JPEG, GIF or WebP image")
    fault = image_fault(path, data)
    if fault is not None:
        raise ValueError(fault)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            # b64encode gives bytes, and JSON carries only text, so the
            # bytes are decoded: base64 is plain ASCII letters and digits.
            "data": base64.b64encode(data).decode("ascii"),
        },
    }


def with_images(
    question: str, paths: list[Path]
) -> list[ImageBlockParam | TextBlockParam]:
    # The images first, the question last: Anthropic's guide finds the
    # model works best that way. With more than one image, a label before
    # each lets the question name them: "Image 1", "Image 2".
    blocks: list[ImageBlockParam | TextBlockParam] = []
    for number, path in enumerate(paths, start=1):
        if len(paths) > 1:
            blocks.append({"type": "text", "text": f"Image {number}:"})
        blocks.append(image_block(path))
    blocks.append({"type": "text", "text": question})
    return blocks
