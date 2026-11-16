from pathlib import Path

import pytest
from tokenizers import Tokenizer

from askkit import tokens
from askkit.prices import PRICES
from askkit.tokens import TOKENIZERS, count_tokens

EMAIL = (
    "Summarise this support email in one line for the queue.\n"
    "\n"
    "Hi, my parcel for order 4417 arrived this morning with a torn box,\n"
    "and the mug inside was broken. I'd like a replacement, not a refund.\n"
    "Can you send one this week? I've attached a photo of the damage.\n"
    "Thanks, Maria\n"
)


def test_the_email_is_272_characters() -> None:
    assert len(EMAIL) == 272


def test_empty_text_has_no_tokens() -> None:
    assert count_tokens("claude-haiku-5-5", "") == (0, False)


def test_claude_is_estimated_at_2_7_characters_a_token() -> None:
    assert count_tokens("claude-haiku-5-5", "a") == (1, False)
    assert count_tokens("claude-haiku-5-5", "a" * 27) == (10, False)
    assert count_tokens("claude-haiku-5-5", "a" * 28) == (11, False)
    assert count_tokens("claude-haiku-5-5", EMAIL) == (101, False)


def test_gemini_and_gpt_are_estimated_at_4_characters_a_token() -> None:
    assert count_tokens("gemini-3.5-flash-lite", EMAIL) == (68, False)
    assert count_tokens("gpt-6-luna", EMAIL) == (68, False)


def test_qwen_is_counted_by_its_own_tokenizer() -> None:
    # The real tokenizer, downloaded from Hugging Face on the first run.
    assert count_tokens("qwen3.8-27b", EMAIL) == (73, True)


def test_every_priced_model_can_be_counted() -> None:
    for model in PRICES:
        if model not in TOKENIZERS:
            assert count_tokens(model, "abcd")[1] is False


def test_an_unknown_model_is_refused() -> None:
    with pytest.raises(KeyError):
        count_tokens("llama-9", "abcd")


def test_a_tokenizer_is_downloaded_once_then_read_from_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tokens, "CACHE", tmp_path)
    assert count_tokens("qwen3.8-27b", EMAIL) == (73, True)
    assert (tmp_path / "Qwen--Qwen3.8-27B.json").exists()

    def no_network(repository: str) -> Tokenizer:
        raise AssertionError(f"downloaded {repository} again")

    monkeypatch.setattr(Tokenizer, "from_pretrained", no_network)
    assert count_tokens("qwen3.8-27b", EMAIL) == (73, True)
