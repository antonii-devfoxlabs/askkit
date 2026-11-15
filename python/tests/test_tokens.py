from askkit.tokens import estimate_tokens


def test_empty_text_has_no_tokens() -> None:
    assert estimate_tokens("") == 0


def test_estimate_rounds_up() -> None:
    assert estimate_tokens("a") == 1
    assert estimate_tokens("a" * 27) == 10
    assert estimate_tokens("a" * 28) == 11
