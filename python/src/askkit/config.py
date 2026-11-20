import os

from dotenv import find_dotenv, load_dotenv

KEY_NAME = "ANTHROPIC_API_KEY"
# Speech to text runs at OpenAI: Anthropic's API takes no audio.
OPENAI_KEY_NAME = "OPENAI_API_KEY"


def load_api_key(name: str = KEY_NAME) -> str | None:
    # Finds .env in this folder or the nearest one above it.
    load_dotenv(find_dotenv(usecwd=True))
    key = os.environ.get(name, "").strip()
    return key or None


def masked(key: str) -> str:
    # Enough to tell two keys apart, never enough to use one.
    if len(key) < 20:
        return "*" * len(key)
    return f"{key[:6]}...{key[-4:]}"
