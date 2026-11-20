import base64
from pathlib import Path
from typing import Literal

from anthropic.types import ImageBlockParam, TextBlockParam

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


def image_block(path: Path) -> ImageBlockParam:
    data = path.read_bytes()
    media_type = media_type_of(data)
    if media_type is None:
        raise ValueError(f"{path} is not a PNG, JPEG, GIF or WebP image")
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
