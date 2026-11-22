"""A bigram language model: a table of word-pair counts, in NumPy."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

# The full stop ends a reply, and a reply starts after one, so its row
# of the table holds how replies begin.
END = "."


def read_replies(path: Path) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.split() for line in lines if line.strip()]


def split_replies(
    replies: list[list[str]],
) -> tuple[list[list[str]], list[list[str]]]:
    # Every tenth reply is held out, so the replies kept apart come from
    # the whole file rather than from its end.
    training = [
        reply for number, reply in enumerate(replies) if number % 10 != 9
    ]
    held_out = replies[9::10]
    return training, held_out


def vocabulary(replies: list[list[str]]) -> list[str]:
    words = {word for reply in replies for word in reply}
    words.discard(END)
    return [END, *sorted(words)]


def pairs(
    replies: list[list[str]], words: list[str]
) -> tuple[list[int], list[int]]:
    position = {word: number for number, word in enumerate(words)}
    firsts: list[int] = []
    seconds: list[int] = []
    for reply in replies:
        tokens = [END, *reply]
        for word, next_word in zip(tokens[:-1], tokens[1:], strict=True):
            firsts.append(position[word])
            seconds.append(position[next_word])
    return firsts, seconds


def count_pairs(
    replies: list[list[str]], words: list[str]
) -> NDArray[np.int64]:
    size = len(words)
    counts = np.zeros((size, size), dtype=np.int64)
    firsts, seconds = pairs(replies, words)
    for first, second in zip(firsts, seconds, strict=True):
        counts[first, second] += 1
    return counts


def probabilities(
    counts: NDArray[np.int64], smoothing: float
) -> NDArray[np.float64]:
    smoothed = counts + smoothing
    totals = smoothed.sum(axis=1, keepdims=True)
    return smoothed / totals


def sample_reply(
    table: NDArray[np.float64],
    words: list[str],
    generator: np.random.Generator,
    max_words: int = 30,
) -> str:
    reply: list[str] = []
    current = 0
    while len(reply) < max_words:
        current = int(generator.choice(len(words), p=table[current]))
        reply.append(words[current])
        if current == 0:
            break
    return " ".join(reply)


def average_loss(
    table: NDArray[np.float64],
    replies: list[list[str]],
    words: list[str],
) -> float:
    firsts, seconds = pairs(replies, words)
    chances = table[firsts, seconds]
    return float(-np.log(chances).mean())
