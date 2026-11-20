import io
import wave
from pathlib import Path

import openai
from openai import OpenAI
from openai.types.audio import Transcription

# OpenAI's transcription model, recommended in its speech-to-text guide
# (checked on 20 November 2026). The same call reaches a Whisper model on
# your own vLLM server: only the client's base_url and the model change.
TRANSCRIBE_MODEL = "gpt-transcribe"
# Estimated by OpenAI's pricing page, per minute of audio (checked on 20
# November 2026). The page gives no rate per token for this model.
PRICE_PER_MINUTE = 0.0045
# The largest file the endpoint takes, from the same guide.
MAX_AUDIO_BYTES = 25 * 1024 * 1024

# An MP3 with no tag at its start begins with its first frame of sound,
# whose first two bytes are one of these.
MP3_FRAMES = (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")

# Tells the model what the callers talk about, and that order numbers are
# written as digits. A hint the model may follow, not a check.
SHOP_PROMPT = (
    "A customer leaves a voice message for a shop that sells mugs, teapots"
    " and jugs. Order numbers are four digits, written as digits: 4417."
)


def audio_format(data: bytes) -> str | None:
    # Told by the first bytes, as images and PDFs are. WAV and M4A hold
    # their name a few bytes in, so those bytes are read where they stand.
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "WAV"
    if data[:3] == b"ID3" or data[:2] in MP3_FRAMES:
        return "MP3"
    if data[4:8] == b"ftyp":
        return "M4A"
    if data[:4] == b"OggS":
        return "OGG"
    if data[:4] == b"fLaC":
        return "FLAC"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "WebM"
    return None


def audio_seconds(data: bytes) -> float | None:
    # Only a WAV says its own length in its header; the other formats are
    # packed, and only a decoder can tell. So no length, and no cost, for
    # them: guessing one from the size would print a wrong price.
    if audio_format(data) != "WAV":
        return None
    try:
        with wave.open(io.BytesIO(data), "rb") as recording:
            return recording.getnframes() / recording.getframerate()
    except wave.Error:
        # A WAV of a kind the wave module does not read, such as one of
        # decimal samples: the endpoint takes it, but its length is unknown.
        return None


def transcription_cost(seconds: float) -> float:
    return seconds / 60 * PRICE_PER_MINUTE


def audio_fault(path: Path, data: bytes) -> str | None:
    # Checked before the call: the endpoint would reject both, after the
    # whole file had gone over the network.
    if audio_format(data) is None:
        return f"{path} is not a WAV, MP3, M4A, OGG, FLAC or WebM file"
    if len(data) > MAX_AUDIO_BYTES:
        megabytes = len(data) / 1024 / 1024
        return (
            f"{path} is {megabytes:.1f} MB, over the 25 MB a call takes;"
            " save it as MP3, or split it at a pause"
        )
    return None


def transcribe(
    client: OpenAI, path: Path, data: bytes, model: str
) -> Transcription:
    # The bytes go as they are, with the file's name: the endpoint asks
    # for a name with its extension, such as voicemail.wav.
    return client.audio.transcriptions.create(
        model=model, file=(path.name, data), prompt=SHOP_PROMPT
    )


def openai_error(error: openai.APIError) -> str:
    # Most specific first: a timeout is a kind of connection error, and a
    # rejected key a kind of status error.
    if isinstance(error, openai.AuthenticationError):
        return "OpenAI rejected the key; check OPENAI_API_KEY in .env"
    if isinstance(error, openai.APIStatusError):
        # The SDK's message starts with the status: "Error code: 400 - ...".
        return f"OpenAI rejected the request: {error.message}"
    if isinstance(error, openai.APITimeoutError):
        return "OpenAI did not answer in time; try again"
    return "cannot reach OpenAI; check your connection"
