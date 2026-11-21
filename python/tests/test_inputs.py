import io
import os
from pathlib import Path

import pytest
from PIL import Image

from askkit.cli import main
from askkit.images import (
    base64_length,
    image_fault,
    image_size,
    image_tokens,
    is_heic,
    resized_size,
)
from askkit.inputs import image_input, pdf_input

SAMPLES = Path(__file__).parent.parent / "samples"


def picture(width: int, height: int, kind: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height)).save(buffer, kind)
    return buffer.getvalue()


def test_an_image_costs_one_token_a_square() -> None:
    assert image_tokens(980, 252) == 315
    assert image_tokens(840, 560) == 600
    # A part of a square is a whole square: 29 px is two of them.
    assert image_tokens(29, 28) == 2


@pytest.mark.parametrize(
    ("size", "sent", "tokens"),
    [
        # The rows of the table in Anthropic's vision guide, on the
        # high-resolution tier, checked on 20 November 2026.
        ((200, 200), (200, 200), 64),
        ((1000, 1000), (1000, 1000), 1296),
        ((1092, 1092), (1092, 1092), 1521),
        ((1920, 1080), (1920, 1080), 2691),
        ((2000, 1500), (2000, 1500), 3888),
        ((3840, 2160), (2576, 1449), 4784),
        # A phone photo, on its side and upright.
        ((4032, 3024), (2212, 1659), 4740),
        ((3024, 4032), (1659, 2212), 4740),
        # A long, thin image stops at the 2,576 px edge, not the tokens.
        ((9000, 100), (2576, 29), 184),
    ],
)
def test_the_api_shrinks_what_does_not_fit(
    size: tuple[int, int], sent: tuple[int, int], tokens: int
) -> None:
    assert resized_size(*size) == sent
    assert image_tokens(*sent) == tokens


def test_the_size_is_read_from_the_header() -> None:
    assert image_size((SAMPLES / "chart.png").read_bytes()) == (840, 560)
    assert image_size(picture(4032, 3024, "JPEG")) == (4032, 3024)
    assert image_size(picture(30, 20, "GIF")) == (30, 20)
    assert image_size(picture(30, 20, "WEBP")) == (30, 20)


def test_base64_is_four_characters_for_three_bytes() -> None:
    assert base64_length(b"abc") == 4
    assert base64_length(b"abcd") == 8
    assert base64_length(b"") == 0


def test_an_image_within_the_limits_has_no_fault() -> None:
    assert image_fault(Path("photo.jpg"), picture(8000, 10, "PNG")) is None


def test_more_than_8000_px_a_side_is_refused() -> None:
    assert image_fault(Path("scan.png"), picture(8001, 10)) == (
        "scan.png is 8,001 x 10 px, over the 8,000 px a side the API takes;"
        " make it smaller"
    )


def test_more_than_10_mb_of_base64_is_refused() -> None:
    # Random pixels do not pack, so the PNG stays as large as its pixels.
    noise = Image.frombytes("RGB", (1700, 1700), os.urandom(1700 * 1700 * 3))
    buffer = io.BytesIO()
    noise.save(buffer, "PNG")
    fault = image_fault(Path("noise.png"), buffer.getvalue())
    assert fault is not None
    assert fault.startswith("noise.png is 11.")
    assert fault.endswith(
        " MB as base64, over the 10 MB an image may be; save it as JPEG,"
        " or make it smaller"
    )


def test_a_broken_header_is_refused() -> None:
    fault = image_fault(Path("photo.jpg"), b"\xff\xd8\xff\xe0\x00\x10JFIF")
    assert (
        fault == "photo.jpg starts like an image, but its size cannot be read"
    )


def test_an_image_input_says_what_the_api_will_bill() -> None:
    counted = image_input(Path("photo.jpg"), picture(4032, 3024, "JPEG"))
    assert counted.what == (
        "JPEG image, 4,032 x 3,024 px, shrunk to 2,212 x 1,659"
    )
    assert counted.tokens == 4740
    assert counted.how == "79 x 60 squares of 28 x 28 px"


def test_a_pdf_input_is_its_page_images_and_its_text() -> None:
    data = (SAMPLES / "invoice.pdf").read_bytes()
    counted = pdf_input(
        Path("invoice.pdf"), data, "claude-haiku-5-5", 1_000_000
    )
    assert counted.what == "PDF, 2 pages, 1,051 characters of text"
    # 1,051 characters at 2.7 a token are 390 tokens, and two page images
    # at most 1,568 each.
    assert counted.tokens == 390 + 2 * 1568
    assert counted.how == "at most: page images 3,136, text 390"


def test_estimate_prices_an_image(capsys: pytest.CaptureFixture[str]) -> None:
    main(["estimate", str(SAMPLES / "chart.png"), "--output", "100"])
    assert capsys.readouterr().out.splitlines() == [
        "Model:         claude-haiku-5-5",
        "Input:         PNG image, 840 x 560 px",
        "Input tokens:  600 (30 x 20 squares of 28 x 28 px)",
        "Output tokens: 100 (your guess)",
        "Window:        700 of 1,000,000 tokens",
        "One call:      $0.000110",
    ]


def test_estimate_prices_a_pdf(capsys: pytest.CaptureFixture[str]) -> None:
    main(["estimate", str(SAMPLES / "invoice.pdf"), "--output", "100"])
    assert capsys.readouterr().out.splitlines() == [
        "Model:         claude-haiku-5-5",
        "Input:         PDF, 2 pages, 1,051 characters of text",
        "Input tokens:  3,526 (at most: page images 3,136, text 390)",
        "Output tokens: 100 (your guess)",
        "Window:        3,626 of 1,000,000 tokens",
        "One call:      $0.000403",
    ]


def test_estimate_prices_a_recording_at_openai(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["estimate", str(SAMPLES / "voicemail.wav")])
    assert capsys.readouterr().out.splitlines() == [
        "Model:         claude-haiku-5-5",
        "Input:         WAV recording, 11.1 s",
        "Transcribe:    $0.000831 at OpenAI, gpt-transcribe",
        "Model reads:   the transcript; estimate it once it is written",
    ]


def test_estimate_names_the_model_that_reads_a_transcript(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The transcript goes to the model given with -m, whichever provider
    # runs it: the estimate must not say Claude reads it.
    main(["estimate", str(SAMPLES / "voicemail.wav"), "-m", "gpt-6-luna"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Model:         gpt-6-luna"
    assert not any("Claude" in line for line in lines)


def test_estimate_gives_no_length_for_a_packed_recording(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    message = tmp_path / "message.mp3"
    message.write_bytes(b"ID3\x04\x00\x00\x00\x00\x00\x00")
    main(["estimate", str(message)])
    assert capsys.readouterr().out.splitlines() == [
        "Model:         claude-haiku-5-5",
        "Input:         MP3 recording, length unknown",
        "Transcribe:    unknown: only a WAV gives its length",
        "Model reads:   the transcript; estimate it once it is written",
    ]


def test_estimate_stops_at_the_image_limit(tmp_path: Path) -> None:
    scan = tmp_path / "scan.png"
    scan.write_bytes(picture(9000, 10))
    with pytest.raises(SystemExit) as stopped:
        main(["estimate", str(scan)])
    assert stopped.value.code == (
        f"askkit: {scan} is 9,000 x 10 px, over the 8,000 px a side the API"
        " takes; make it smaller"
    )


def test_estimate_stops_at_the_recording_limit(tmp_path: Path) -> None:
    meeting = tmp_path / "meeting.mp3"
    meeting.write_bytes(b"ID3" + bytes(25 * 1024 * 1024))
    with pytest.raises(SystemExit) as stopped:
        main(["estimate", str(meeting)])
    assert stopped.value.code == (
        f"askkit: {meeting} is 25.0 MB, over the 25 MB a call takes; save"
        " it as MP3, or split it at a pause"
    )


def test_estimate_counts_images_for_claude_only() -> None:
    with pytest.raises(SystemExit) as stopped:
        main(
            [
                "estimate",
                str(SAMPLES / "chart.png"),
                "-m",
                "gemini-3.5-flash-lite",
            ]
        )
    assert stopped.value.code == (
        "askkit: AskKit counts image and PDF tokens for Claude only; each"
        " provider counts them its own way"
    )


def test_ask_refuses_an_image_over_the_limit_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    Path("scan.png").write_bytes(picture(8001, 10))
    with pytest.raises(SystemExit) as stopped:
        main(["ask", "What does it say?", "--image", "scan.png"])
    assert stopped.value.code == (
        "askkit: scan.png is 8,001 x 10 px, over the 8,000 px a side the API"
        " takes; make it smaller"
    )


def test_read_refuses_too_many_pages_before_any_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pypdf import PdfWriter

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    writer = PdfWriter()
    for _ in range(601):
        writer.add_blank_page(width=595, height=842)
    with open("report.pdf", "wb") as file:
        writer.write(file)
    with pytest.raises(SystemExit) as stopped:
        main(["read", "report.pdf"])
    assert stopped.value.code == (
        "askkit: report.pdf has 601 pages, over the 600 a request may hold;"
        " send it in parts"
    )


def test_estimate_tells_a_heic_photo_from_a_recording(tmp_path: Path) -> None:
    photo = tmp_path / "IMG_4417.HEIC"
    photo.write_bytes(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1")
    with pytest.raises(SystemExit) as stopped:
        main(["estimate", str(photo)])
    assert stopped.value.code == (
        f"askkit: {photo} is a HEIC photo, which Claude's API does not read;"
        " save it as JPEG"
    )


def test_an_m4a_recording_is_not_a_heic_photo() -> None:
    assert not is_heic(b"\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00")
    assert is_heic(b"\x00\x00\x00\x18ftypmif1\x00\x00\x00\x00")
