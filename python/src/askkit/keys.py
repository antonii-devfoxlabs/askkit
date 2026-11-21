import os

# Every key AskKit can be given: Anthropic's for the model calls,
# OpenAI's for speech to text and moderation, Google's for the layer's
# Gemini models. A key is a password for a paid account, so none of
# them may be written into a log, however it got into the text.
KEY_NAMES = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY")
KEY_PLACEHOLDER = "[KEY]"


def known_keys() -> list[str]:
    # The keys this run holds. A name set to nothing is no key: replacing
    # an empty string would put the placeholder between every two
    # characters of the text.
    keys: list[str] = []
    for name in KEY_NAMES:
        key = os.environ.get(name, "").strip()
        if key:
            keys.append(key)
    return keys


def without_keys(text: str) -> str:
    # The text with every key this run holds replaced, word for word. A
    # pattern would guess at what a key looks like; the keys themselves
    # are known, so each one is found exactly, wherever it stands.
    for key in known_keys():
        text = text.replace(key, KEY_PLACEHOLDER)
    return text
