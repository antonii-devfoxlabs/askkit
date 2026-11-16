# US dollars per million tokens, standard prices, each from its
# provider's own pricing page, checked on 2026-11-16:
#   Anthropic  https://platform.claude.com/docs/en/about-claude/pricing
#   Google     https://aistudio.google.com/docs/pricing
#   OpenAI     https://developers.openai.com/api/docs/pricing
#   Qwen       https://www.qwencloud.com/models/qwen3.8-27b
# When prices change, change this file only.
PRICES: dict[str, dict[str, float]] = {
    "claude-fable-5-2": {"input": 8.00, "output": 40.00},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gpt-6-luna": {"input": 0.10, "output": 0.50},
    "qwen3.8-27b": {"input": 0.50, "output": 3.00},
}

# Some models cost more for the whole call when a prompt is long.
LONG_PROMPT_PRICES: dict[str, dict[str, float]] = {
    "claude-haiku-5-5": {"over": 100_000, "input": 0.50, "output": 2.50},
    "gpt-6-luna": {"over": 272_000, "input": 0.20, "output": 0.75},
}
