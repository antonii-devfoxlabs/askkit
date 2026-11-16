import math
from pathlib import Path

from tokenizers import Tokenizer

# Models whose tokenizer is published, with the Hugging Face repository
# that holds it: AskKit counts their tokens exactly.
TOKENIZERS: dict[str, str] = {
    "qwen3.8-27b": "Qwen/Qwen3.8-27B",
}

# The first count downloads a tokenizer and keeps a copy here, so every
# later count runs on this machine alone.
CACHE = Path.home() / ".cache" / "askkit"

# Every other model gets an estimate: English characters per token, by
# the start of the model ID, from each provider's docs, checked on
# 2026-11-16.
# Anthropic: about 3.5 characters before Claude Opus 4.7, whose newer
# tokenizer makes about 30 percent more tokens: 3.5 / 1.3 is about 2.7.
# Google: "a token is equivalent to about 4 characters".
# OpenAI: "1 token is approximately 4 characters" of English.
CHARACTERS_PER_TOKEN: dict[str, float] = {
    "claude-": 2.7,
    "gemini-": 4.0,
    "gpt-": 4.0,
}


def load_tokenizer(repository: str) -> Tokenizer:
    path = CACHE / (repository.replace("/", "--") + ".json")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        Tokenizer.from_pretrained(repository).save(str(path))
    return Tokenizer.from_file(str(path))


def count_tokens(model: str, text: str) -> tuple[int, bool]:
    """Return the tokens in text for model, and whether that is exact."""
    repository = TOKENIZERS.get(model)
    if repository is not None:
        tokenizer = load_tokenizer(repository)
        return len(tokenizer.encode(text).ids), True
    for start, characters in CHARACTERS_PER_TOKEN.items():
        if model.startswith(start):
            return math.ceil(len(text) / characters), False
    raise KeyError(f"no tokenizer or estimate for {model}")
