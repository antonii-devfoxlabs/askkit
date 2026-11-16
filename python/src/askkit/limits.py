# Tokens per call, from
# https://platform.claude.com/docs/en/about-claude/models/overview
# checked on 2026-11-16. When limits change, change this file only.
LIMITS: dict[str, dict[str, int]] = {
    "claude-fable-5-2": {"context_window": 1_000_000, "max_output": 128_000},
    "claude-opus-5-5": {"context_window": 1_000_000, "max_output": 128_000},
    "claude-sonnet-5-5": {"context_window": 1_000_000, "max_output": 128_000},
    "claude-haiku-5-5": {"context_window": 1_000_000, "max_output": 128_000},
}
