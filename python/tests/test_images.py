import base64
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from test_tools import use_answers

from askkit.cli import main
from askkit.images import image_block, media_type_of, with_images

# Made for the course: a terminal with a traceback, and a bar chart of
# invented daily costs.
SAMPLES = Path(__file__).parent.parent / "samples"
SCREENSHOT = SAMPLES / "screenshot.png"
CHART = SAMPLES / "chart.png"

QUESTION = "On which day did AskKit cost the most, and about how much?"
ANSWER = "Wednesday, 11 November: about $1.95, the only day over $1."


def answered() -> dict[str, Any]:
    return {
        "id": "msg_01AskKitChart",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": ANSWER}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1474, "output_tokens": 24},
    }


@pytest.mark.parametrize(
    ("start", "media_type"),
    [
        (b"\x89PNG\r\n\x1a\n\x00\x00", "image/png"),
        (b"\xff\xd8\xff\xe0\x00\x10JFIF", "image/jpeg"),
        (b"GIF87a\x01\x00", "image/gif"),
        (b"GIF89a\x01\x00", "image/gif"),
        (b"RIFF\x24\x00\x00\x00WEBPVP8 ", "image/webp"),
    ],
    ids=["png", "jpeg", "gif87", "gif89", "webp"],
)
def test_the_first_bytes_name_the_format(
    start: bytes, media_type: str
) -> None:
    assert media_type_of(start) == media_type


@pytest.mark.parametrize(
    "start",
    [
        b"Order 4417 came with a broken mug.",
        # An iPhone photo: HEIC, which Anthropic's API does not read.
        b"\x00\x00\x00\x18ftypheic\x00\x00",
        b"RIFF\x24\x00\x00\x00WAVEfmt ",
        b"",
    ],
    ids=["text", "heic", "wave", "empty"],
)
def test_anything_else_is_not_an_image_the_api_reads(start: bytes) -> None:
    assert media_type_of(start) is None


def test_the_samples_are_png_images() -> None:
    assert media_type_of(SCREENSHOT.read_bytes()) == "image/png"
    assert media_type_of(CHART.read_bytes()) == "image/png"


def test_an_image_goes_as_base64_text() -> None:
    block = image_block(CHART)
    assert block["type"] == "image"
    source: Any = block["source"]
    assert source["type"] == "base64"
    assert source["media_type"] == "image/png"
    assert isinstance(source["data"], str)
    assert base64.b64decode(source["data"]) == CHART.read_bytes()


def test_the_bytes_decide_the_format_not_the_name(tmp_path: Path) -> None:
    photo = tmp_path / "photo.png"
    Image.new("RGB", (40, 30)).save(photo, "JPEG")
    source: Any = image_block(photo)["source"]
    assert source["media_type"] == "image/jpeg"


def test_a_file_that_is_no_image_is_refused(tmp_path: Path) -> None:
    notes = tmp_path / "notes.txt"
    notes.write_text("Order 4417 came with a broken mug.")
    with pytest.raises(ValueError) as caught:
        image_block(notes)
    assert str(caught.value) == (
        f"{notes} is not a PNG, JPEG, GIF or WebP image"
    )


def test_one_image_comes_before_the_question() -> None:
    blocks = with_images(QUESTION, [CHART])
    assert [block["type"] for block in blocks] == ["image", "text"]
    assert blocks[-1] == {"type": "text", "text": QUESTION}


def test_several_images_are_labelled_in_order() -> None:
    blocks = with_images("What changed?", [SCREENSHOT, CHART])
    texts = []
    for block in blocks:
        if block["type"] == "text":
            texts.append(block["text"])
        else:
            texts.append("<image>")
    assert texts == [
        "Image 1:",
        "<image>",
        "Image 2:",
        "<image>",
        "What changed?",
    ]


def test_ask_sends_the_image_before_the_question(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [answered()])
    main(["ask", QUESTION, "--image", str(CHART)])
    content = sent[0]["messages"][-1]["content"]
    assert [block["type"] for block in content] == ["image", "text"]
    assert content[0]["source"]["media_type"] == "image/png"
    assert base64.b64decode(content[0]["source"]["data"]) == (
        CHART.read_bytes()
    )
    assert content[1]["text"] == QUESTION
    assert capsys.readouterr().out.splitlines()[:3] == [
        ANSWER,
        "",
        "Model:         claude-haiku-5-5",
    ]


def test_ask_labels_two_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [answered()])
    main(
        ["ask", "What changed?", "--image", str(SCREENSHOT)]
        + [
            "--image",
            str(CHART),
        ]
    )
    content = sent[0]["messages"][-1]["content"]
    assert content[0] == {"type": "text", "text": "Image 1:"}
    assert content[2] == {"type": "text", "text": "Image 2:"}
    assert content[-1] == {"type": "text", "text": "What changed?"}


def test_ask_refuses_a_file_that_is_no_image_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("notes.txt").write_text("Order 4417 came with a broken mug.")
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [answered()])
    with pytest.raises(SystemExit) as stopped:
        main(["ask", QUESTION, "--image", "notes.txt"])
    assert stopped.value.code == (
        "askkit: notes.txt is not a PNG, JPEG, GIF or WebP image"
    )
    assert sent == []


def test_ask_says_which_image_it_cannot_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [answered()])
    with pytest.raises(SystemExit) as stopped:
        main(["ask", QUESTION, "--image", "chart.png"])
    assert stopped.value.code == (
        "askkit: cannot read chart.png: No such file or directory"
    )
    assert sent == []


def test_ask_keeps_images_out_of_a_saved_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    sent: list[dict[str, Any]] = []
    use_answers(monkeypatch, sent, [answered()])
    with pytest.raises(SystemExit) as stopped:
        main(["ask", QUESTION, "--image", str(CHART), "--history", "c.json"])
    assert stopped.value.code == (
        "askkit: --history keeps text only; ask about an image without"
        " --history"
    )
    assert sent == []
