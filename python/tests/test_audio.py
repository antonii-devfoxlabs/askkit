import io
import struct
import wave
from pathlib import Path
from typing import Any

import httpx2
import pytest
from openai import OpenAI

from askkit.audio import (
    MAX_AUDIO_BYTES,
    PRICE_PER_MINUTE,
    SHOP_PROMPT,
    audio_fault,
    audio_format,
    audio_seconds,
    transcribe,
    transcription_cost,
)
from askkit.cli import main
from askkit.config import OPENAI_KEY_NAME, load_api_key

VOICEMAIL = Path(__file__).parent.parent / "samples" / "voicemail.wav"

# What a transcription model writes for the sample, as the stand-in says it.
HEARD = (
    "Hi, this is Maria. I'm calling about order 4417. The mug arrived"
    " broken, and the box was torn too. I'd like a new one, please, not a"
    " refund. Thank you."
)
TRANSCRIPTION: dict[str, Any] = {
    "text": HEARD,
    "usage": {
        "type": "tokens",
        "input_tokens": 154,
        "input_token_details": {"audio_tokens": 111, "text_tokens": 43},
        "output_tokens": 41,
        "total_tokens": 195,
    },
}


def wav_bytes(seconds: int, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(rate)
        recording.writeframes(b"\x00\x00" * rate * seconds)
    return buffer.getvalue()


def fake_openai(
    sent: list[httpx2.Request],
    status: int = 200,
    body: dict[str, Any] = TRANSCRIPTION,
    base_url: str | None = None,
) -> OpenAI:
    def answer(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(status, json=body)

    transport = httpx2.MockTransport(answer)
    return OpenAI(
        api_key="sk-proj-a-test-key-Cd34",
        base_url=base_url,
        max_retries=0,
        http_client=httpx2.Client(transport=transport),
    )


@pytest.fixture
def keyed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(OPENAI_KEY_NAME, "sk-proj-a-test-key-Cd34")


def use_fake_openai(
    monkeypatch: pytest.MonkeyPatch,
    sent: list[httpx2.Request],
    status: int = 200,
    body: dict[str, Any] = TRANSCRIPTION,
) -> None:
    def make(api_key: str) -> OpenAI:
        return fake_openai(sent, status, body)

    monkeypatch.setattr("askkit.cli.OpenAI", make)


@pytest.mark.parametrize(
    ("start", "kind"),
    [
        (b"RIFF\x24\x00\x00\x00WAVEfmt ", "WAV"),
        (b"ID3\x04\x00\x00\x00", "MP3"),
        (b"\xff\xfb\x90\x64\x00", "MP3"),
        (b"\xff\xfa\x90\x64\x00", "MP3"),
        (b"\x00\x00\x00\x20ftypM4A ", "M4A"),
        (b"OggS\x00\x02\x00\x00", "Ogg"),
        (b"fLaC\x00\x00\x00\x22", "FLAC"),
        (b"\x1a\x45\xdf\xa3\x9f\x42\x86", "WebM"),
    ],
)
def test_each_format_is_told_by_its_first_bytes(
    start: bytes, kind: str
) -> None:
    assert audio_format(start) == kind


@pytest.mark.parametrize(
    "start",
    [
        b"Order 4417 came broken",
        b"%PDF-1.7\n",
        # A WebP image is a RIFF file too, but not a WAVE one.
        b"RIFF\x24\x00\x00\x00WEBPVP8 ",
        b"",
    ],
)
def test_text_a_pdf_an_image_and_nothing_are_not_audio(start: bytes) -> None:
    assert audio_format(start) is None


def test_the_sample_is_a_wav_of_eleven_seconds() -> None:
    data = VOICEMAIL.read_bytes()
    assert audio_format(data) == "WAV"
    # 16,000 samples a second, two bytes each, after a 44-byte header.
    assert len(data) == 44 + 177_307 * 2
    assert audio_seconds(data) == 177_307 / 16_000


def test_only_a_wav_has_a_length() -> None:
    assert audio_seconds(wav_bytes(3)) == 3
    assert audio_seconds(b"ID3\x04\x00\x00\x00" + bytes(100)) is None


def test_a_wav_the_wave_module_cannot_read_has_no_length() -> None:
    # Format 3 is decimal samples: a real WAV, but not one wave reads.
    header = b"RIFF" + struct.pack("<I", 36) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 3, 1, 16000, 64000, 4, 32)
    header += b"data" + struct.pack("<I", 0)
    assert audio_format(header) == "WAV"
    assert audio_seconds(header) is None


def test_a_minute_costs_the_price_per_minute() -> None:
    assert transcription_cost(60) == PRICE_PER_MINUTE
    assert round(transcription_cost(177_307 / 16_000), 6) == 0.000831


def test_a_file_that_is_not_audio_is_a_fault() -> None:
    assert audio_fault(Path("notes.txt"), b"Order 4417") == (
        "notes.txt is not a WAV, MP3, M4A, Ogg, FLAC or WebM file"
    )


def test_a_file_over_25_mb_is_a_fault() -> None:
    data = b"ID3" + bytes(MAX_AUDIO_BYTES)
    assert audio_fault(Path("meeting.mp3"), data) == (
        "meeting.mp3 is 25.0 MB, over the 25 MB a call takes;"
        " save it as MP3, or split it at a pause"
    )
    assert audio_fault(Path("meeting.mp3"), data[:MAX_AUDIO_BYTES]) is None


def test_transcribe_sends_the_file_with_its_name_and_the_prompt() -> None:
    sent: list[httpx2.Request] = []
    data = VOICEMAIL.read_bytes()
    result = transcribe(fake_openai(sent), VOICEMAIL, data, "gpt-transcribe")
    assert result.text == HEARD
    request = sent[0]
    assert request.url.path == "/v1/audio/transcriptions"
    body = request.read()
    assert b'name="model"\r\n\r\ngpt-transcribe' in body
    assert b'filename="voicemail.wav"' in body
    assert SHOP_PROMPT.encode() in body
    assert data in body


def test_the_same_call_reaches_whisper_on_your_own_server() -> None:
    sent: list[httpx2.Request] = []
    client = fake_openai(sent, base_url="http://localhost:8000/v1")
    model = "openai/whisper-large-v3-turbo"
    transcribe(client, VOICEMAIL, VOICEMAIL.read_bytes(), model)
    assert str(sent[0].url) == "http://localhost:8000/v1/audio/transcriptions"
    assert f'name="model"\r\n\r\n{model}'.encode() in sent[0].read()


def test_the_openai_key_is_read_by_its_own_name(
    keyed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert load_api_key(OPENAI_KEY_NAME) == "sk-proj-a-test-key-Cd34"
    assert load_api_key() is None


def test_transcribe_prints_the_text_and_what_it_cost(
    keyed: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sent: list[httpx2.Request] = []
    use_fake_openai(monkeypatch, sent)
    main(["transcribe", str(VOICEMAIL)])
    assert capsys.readouterr().out == (
        f"{HEARD}\n"
        "\n"
        "Model:         gpt-transcribe\n"
        "Tokens:        154 in, 41 out\n"
        "Audio:         WAV, 11.1 s\n"
        "Cost:          $0.000831\n"
    )
    assert len(sent) == 1


def test_out_saves_the_text_for_extract(
    keyed: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    use_fake_openai(monkeypatch, [])
    out = tmp_path / "voicemail.txt"
    main(["transcribe", str(VOICEMAIL), "--out", str(out)])
    assert out.read_text(encoding="utf-8") == HEARD + "\n"
    assert capsys.readouterr().out.endswith(f"Saved:         {out}\n")


def test_an_mp3_has_no_length_and_no_cost(
    keyed: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    use_fake_openai(monkeypatch, [])
    path = tmp_path / "voicemail.mp3"
    path.write_bytes(b"ID3\x04\x00\x00\x00" + bytes(100))
    main(["transcribe", str(path)])
    out = capsys.readouterr().out
    assert "Audio:         MP3, length unknown\n" in out
    assert "Cost:          unknown: only a WAV gives its length\n" in out


def test_a_file_that_is_not_audio_stops_before_any_call(
    keyed: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[httpx2.Request] = []
    use_fake_openai(monkeypatch, sent)
    path = tmp_path / "voicemail.txt"
    path.write_text("Order 4417", encoding="utf-8")
    with pytest.raises(SystemExit) as stopped:
        main(["transcribe", str(path)])
    assert stopped.value.code == (
        f"askkit: {path} is not a WAV, MP3, M4A, Ogg, FLAC or WebM file"
    )
    assert sent == []


def test_a_missing_file_stops_before_any_call(
    keyed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    sent: list[httpx2.Request] = []
    use_fake_openai(monkeypatch, sent)
    with pytest.raises(SystemExit) as stopped:
        main(["transcribe", "missing.wav"])
    assert stopped.value.code == (
        "askkit: cannot read missing.wav: No such file or directory"
    )
    assert sent == []


def test_no_openai_key_says_where_it_goes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(OPENAI_KEY_NAME, raising=False)
    with pytest.raises(SystemExit) as stopped:
        main(["transcribe", str(VOICEMAIL)])
    assert stopped.value.code == (
        "askkit: OPENAI_API_KEY is not set (transcribe runs at OpenAI);"
        " put your OpenAI key after the = in .env"
    )


def test_a_rejected_key_is_said_in_one_line(
    keyed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = {"error": {"message": "Incorrect API key provided."}}
    use_fake_openai(monkeypatch, [], status=401, body=body)
    with pytest.raises(SystemExit) as stopped:
        main(["transcribe", str(VOICEMAIL)])
    assert stopped.value.code == (
        "askkit: OpenAI rejected the key; check OPENAI_API_KEY in .env"
    )


def test_another_rejection_names_its_status(
    keyed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = {"error": {"message": "Audio file might be corrupted."}}
    use_fake_openai(monkeypatch, [], status=400, body=body)
    with pytest.raises(SystemExit) as stopped:
        main(["transcribe", str(VOICEMAIL)])
    assert stopped.value.code == (
        "askkit: OpenAI rejected the request: Error code: 400 -"
        " Audio file might be corrupted."
    )
