import math
from pathlib import Path

import numpy as np
import pytest

from askkit.bigram import (
    average_loss,
    count_pairs,
    pairs,
    probabilities,
    read_replies,
    sample_reply,
    split_replies,
    vocabulary,
)
from askkit.cli import main

# Lesson 1.3's ten replies, the toy every bigram test counts.
TEN_REPLIES = [
    reply.split()
    for reply in [
        "your order has shipped .",
        "your order has shipped .",
        "your order has shipped .",
        "your order is late .",
        "your order is late .",
        "your refund has been sent .",
        "your refund has been sent .",
        "your refund is ready .",
        "your parcel is late .",
        "your mug has shipped .",
    ]
]
SAMPLES = Path(__file__).parent.parent / "samples" / "replies.txt"


def followers(word: str) -> dict[str, int]:
    words = vocabulary(TEN_REPLIES)
    counts = count_pairs(TEN_REPLIES, words)
    row = counts[words.index(word)]
    return {words[column]: int(row[column]) for column in np.nonzero(row)[0]}


def test_the_full_stop_comes_first_then_the_words_in_order() -> None:
    assert vocabulary(TEN_REPLIES) == [
        ".",
        "been",
        "has",
        "is",
        "late",
        "mug",
        "order",
        "parcel",
        "ready",
        "refund",
        "sent",
        "shipped",
        "your",
    ]


def test_your_is_followed_as_lesson_1_3_counted() -> None:
    assert followers("your") == {
        "order": 5,
        "refund": 3,
        "parcel": 1,
        "mug": 1,
    }


def test_the_full_stop_row_counts_how_replies_start() -> None:
    assert followers(".") == {"your": 10}


def test_every_pair_is_counted_once() -> None:
    words = vocabulary(TEN_REPLIES)
    counts = count_pairs(TEN_REPLIES, words)
    words_in_replies = sum(len(reply) for reply in TEN_REPLIES)
    assert counts.sum() == words_in_replies
    assert counts.shape == (13, 13)


def test_pairs_start_each_reply_at_the_full_stop() -> None:
    words = vocabulary(TEN_REPLIES)
    firsts, seconds = pairs(TEN_REPLIES[:1], words)
    assert [words[number] for number in firsts] == [
        ".",
        "your",
        "order",
        "has",
        "shipped",
    ]
    assert [words[number] for number in seconds] == [
        "your",
        "order",
        "has",
        "shipped",
        ".",
    ]


def test_each_row_of_probabilities_adds_up_to_one() -> None:
    words = vocabulary(TEN_REPLIES)
    table = probabilities(count_pairs(TEN_REPLIES, words), 0.01)
    assert np.allclose(table.sum(axis=1), 1.0)


def test_without_smoothing_your_gives_lesson_1_3_shares() -> None:
    words = vocabulary(TEN_REPLIES)
    table = probabilities(count_pairs(TEN_REPLIES, words), 0.0)
    row = table[words.index("your")]
    assert row[words.index("order")] == pytest.approx(0.5)
    assert row[words.index("refund")] == pytest.approx(0.3)
    assert row[words.index("late")] == 0.0


def test_smoothing_gives_an_unseen_pair_a_small_probability() -> None:
    words = vocabulary(TEN_REPLIES)
    table = probabilities(count_pairs(TEN_REPLIES, words), 1.0)
    row = table[words.index("your")]
    # 0 + 1 over the row's 10 counts plus 13 added ones
    assert row[words.index("late")] == pytest.approx(1 / 23)


def test_a_sample_starts_after_the_full_stop_and_ends_on_one() -> None:
    words = vocabulary(TEN_REPLIES)
    table = probabilities(count_pairs(TEN_REPLIES, words), 0.0)
    reply = sample_reply(table, words, np.random.default_rng(4417))
    assert reply.split()[0] == "your"
    assert reply.endswith(" .")


def test_the_same_seed_writes_the_same_replies() -> None:
    words = vocabulary(TEN_REPLIES)
    table = probabilities(count_pairs(TEN_REPLIES, words), 0.0)
    first = np.random.default_rng(7)
    second = np.random.default_rng(7)
    assert [sample_reply(table, words, first) for _ in range(5)] == [
        sample_reply(table, words, second) for _ in range(5)
    ]


def test_a_sample_stops_at_max_words() -> None:
    words = vocabulary(TEN_REPLIES)
    size = len(words)
    # Never the full stop: only max_words can end it.
    table = np.zeros((size, size))
    table[:, words.index("your")] = 1.0
    reply = sample_reply(table, words, np.random.default_rng(1), 4)
    assert reply == "your your your your"


def test_a_uniform_table_scores_the_log_of_the_vocabulary() -> None:
    words = vocabulary(TEN_REPLIES)
    size = len(words)
    table = np.full((size, size), 1 / size)
    loss = average_loss(table, TEN_REPLIES, words)
    assert loss == pytest.approx(math.log(13))


def test_an_unseen_pair_without_smoothing_scores_infinity() -> None:
    words = vocabulary(TEN_REPLIES)
    table = probabilities(count_pairs(TEN_REPLIES, words), 0.0)
    unseen = [["your", "late", "."]]
    with pytest.warns(RuntimeWarning, match="divide by zero"):
        loss = average_loss(table, unseen, words)
    assert loss == math.inf


def test_every_tenth_reply_is_held_out() -> None:
    replies = [[str(number), "."] for number in range(25)]
    training, held_out = split_replies(replies)
    assert held_out == [["9", "."], ["19", "."]]
    assert len(training) == 23


def test_the_samples_file_holds_no_word_the_model_never_learned() -> None:
    training, held_out = split_replies(read_replies(SAMPLES))
    words = set(vocabulary(training))
    assert all(word in words for reply in held_out for word in reply)


def test_bigram_prints_the_table_the_loss_and_samples(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["bigram", str(SAMPLES), "--samples", "2"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[:4] == [
        "Replies:       183 to learn from, 20 held out",
        "Vocabulary:    148 words",
        "Table:         148 by 148 = 21,904 cells, 449 not zero",
        "Held out:      2.04 a word (a blind guess: 5.00)",
    ]
    assert lines[4] == "Samples:       seed 4417, smoothing 0.01"
    assert len(lines) == 7
    assert all(line.endswith(" .") for line in lines[5:])


def test_bigram_refuses_a_smoothing_below_zero() -> None:
    with pytest.raises(SystemExit, match="give 0 or more"):
        main(["bigram", str(SAMPLES), "--smoothing", "-1"])
