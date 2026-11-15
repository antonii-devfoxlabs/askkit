import math

# Anthropic's docs: on models before Claude 4.7 a token is about 3.5
# English characters, and Claude 4.7 and later models make about 30
# percent more tokens from the same text. 3.5 / 1.3 is about 2.7.
CHARACTERS_PER_TOKEN = 2.7


def estimate_tokens(text: str) -> int:
    # Rounded up: an estimate that errs high never surprises the bill.
    return math.ceil(len(text) / CHARACTERS_PER_TOKEN)
