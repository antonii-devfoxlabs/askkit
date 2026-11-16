# Tokens per call, each from its provider's own model page, checked on
# 2026-11-16. When limits change, change this file only.
# Anthropic: https://platform.claude.com/docs/en/about-claude/models/overview
# Google: https://aistudio.google.com/docs/models/gemini-3.5-flash-lite
# OpenAI: https://developers.openai.com/api/docs/models/gpt-6-luna
# Qwen: https://www.qwencloud.com/models/qwen3.8-27b
# context_window: input and answer together. max_input: the input alone,
# lower than the window on some models. max_output: the longest answer.
LIMITS: dict[str, dict[str, int]] = {
    "claude-haiku-5-5": {
        "context_window": 1_000_000,
        "max_input": 1_000_000,
        "max_output": 128_000,
    },
    # Google gives an input limit and an output limit; AskKit takes the
    # input limit as the window, so it never says a call fits when it
    # may not.
    "gemini-3.5-flash-lite": {
        "context_window": 1_048_576,
        "max_input": 1_048_576,
        "max_output": 65_536,
    },
    "gpt-6-luna": {
        "context_window": 1_050_000,
        "max_input": 922_000,
        "max_output": 128_000,
    },
    # Qwen Cloud rounds to thousands: 1M, 983K and 131K. 983K is the
    # input limit with thinking, which Qwen3.8 does by default (991K
    # without).
    "qwen3.8-27b": {
        "context_window": 1_000_000,
        "max_input": 983_000,
        "max_output": 131_000,
    },
}
