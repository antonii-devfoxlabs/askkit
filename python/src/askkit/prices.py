# US dollars per million tokens, standard prices, each from its
# provider's own pricing page, checked on 2026-11-16:
# Anthropic: https://platform.claude.com/docs/en/about-claude/pricing
# Google: https://aistudio.google.com/docs/pricing
# OpenAI: https://developers.openai.com/api/docs/pricing
# Qwen: https://www.qwencloud.com/models/qwen3.8-27b
# When prices change, change this file only.
# The first four are the high-volume model of each provider, the ones
# askkit compare sets side by side. claude-sonnet-5-5 is the larger model
# a request is routed to when the small one is not enough (lesson 10.3),
# checked on 2026-11-20.
PRICES: dict[str, dict[str, float]] = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gpt-6-luna": {"input": 0.10, "output": 0.50},
    "qwen3.8-27b": {"input": 0.50, "output": 3.00},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
}

# Some models cost more for the whole call when a prompt is long.
LONG_PROMPT_PRICES: dict[str, dict[str, float]] = {
    "claude-haiku-5-5": {"over": 100_000, "input": 0.50, "output": 2.50},
    "gpt-6-luna": {"over": 272_000, "input": 0.20, "output": 0.75},
}

# Prompt caching, the same Anthropic page, checked on 2026-11-20: a write
# to the 5-minute cache (1.25 times the input price) and a read from it
# (0.1 times). Over the long prompt step, both from the higher prices.
# Claude Sonnet 5.5 reads at 0.05 times its input price, not 0.1 (the
# same page, its footnote 2, checked on 2026-11-20).
# GPT-6 Luna, the model askkit answer can fall back to at OpenAI (lesson
# 10.5), from OpenAI's pricing page, checked on 2026-11-20: a write to its
# 30-minute cache and a read; over 272,000 input tokens, the long-context
# prices.
# Gemini 3.5 Flash-Lite, the model askkit answer can switch to at Google
# (lesson 13.2), from Google's pricing page, checked on 2026-11-21: a read
# from its cache at $0.03. Its implicit cache is on by default and has
# no write price (the caching page, the same day), so a write, which the
# layer never reports, would be priced as plain input.
CACHE_PRICES: dict[str, dict[str, float]] = {
    "claude-haiku-5-5": {"write": 0.125, "read": 0.01},
    "claude-sonnet-5-5": {"write": 2.50, "read": 0.10},
    "gemini-3.5-flash-lite": {"write": 0.30, "read": 0.03},
    "gpt-6-luna": {"write": 0.125, "read": 0.01},
}
LONG_PROMPT_CACHE_PRICES: dict[str, dict[str, float]] = {
    "claude-haiku-5-5": {"write": 0.625, "read": 0.05},
    "gpt-6-luna": {"write": 0.25, "read": 0.02},
}

# US dollars per search with Anthropic's web search tool, on top of the
# tokens its results add: $10 per 1,000 searches, checked on 2026-11-18
# on the same pricing page. A search that fails is not billed.
SEARCH_PRICE = 10 / 1000

# The Message Batches API bills every token of a batch request, cache
# writes and reads included, at this share of the prices above: "All
# usage is charged at 50 percent of the standard API prices" (the
# batch processing page and the pricing page, checked on 2026-11-20).
BATCH_SHARE = 0.5
