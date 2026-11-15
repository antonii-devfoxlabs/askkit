# US dollars per million tokens, from
# https://platform.claude.com/docs/en/about-claude/pricing
# checked on 2026-11-16. When prices change, change this file only.
PRICES: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
}

# Claude Haiku 5.5 costs more when a prompt is over 100,000 tokens.
LONG_PROMPT_PRICES: dict[str, dict[str, float]] = {
    "claude-haiku-5-5": {"over": 100_000, "input": 0.50, "output": 2.50},
}
