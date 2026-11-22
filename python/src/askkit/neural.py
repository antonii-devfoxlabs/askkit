"""A neural bigram model: an embedding table and an output table, trained
by gradient descent with NumPy, in place of the bigram model's counts."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class Tables:
    embeddings: NDArray[np.float64]
    outputs: NDArray[np.float64]


def start_tables(
    size: int, width: int, generator: np.random.Generator
) -> Tables:
    # Every number starts small and different from the others, so no two
    # words start as twins that would get the same slopes for ever.
    return Tables(
        embeddings=generator.uniform(-0.1, 0.1, (size, width)),
        outputs=generator.uniform(-0.1, 0.1, (width, size)),
    )


def softmax(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    # Taking each row's largest score away changes no probability, and
    # keeps np.exp from overflowing to infinity on a large score.
    powers = np.exp(scores - scores.max(axis=1, keepdims=True))
    return powers / powers.sum(axis=1, keepdims=True)


def table_of(tables: Tables) -> NDArray[np.float64]:
    return softmax(tables.embeddings @ tables.outputs)


def slopes(tables: Tables, firsts: list[int], seconds: list[int]) -> Tables:
    rows = tables.embeddings[firsts]
    misses = softmax(rows @ tables.outputs)
    # A probability is its own miss, except the real next word's, which
    # is its probability minus 1; the loss is the average over the pairs.
    misses[np.arange(len(seconds)), seconds] -= 1.0
    misses /= len(seconds)
    output_slopes = rows.T @ misses
    row_slopes = misses @ tables.outputs.T
    embedding_slopes = np.zeros_like(tables.embeddings)
    # A word read in many pairs gets the slopes of every one of them;
    # embedding_slopes[firsts] += row_slopes would keep only one.
    np.add.at(embedding_slopes, firsts, row_slopes)
    return Tables(embeddings=embedding_slopes, outputs=output_slopes)


def train_step(
    tables: Tables,
    firsts: list[int],
    seconds: list[int],
    learning_rate: float,
) -> None:
    found = slopes(tables, firsts, seconds)
    tables.embeddings -= learning_rate * found.embeddings
    tables.outputs -= learning_rate * found.outputs
