import math

import numpy as np
import pytest
from test_bigram import SAMPLES, TEN_REPLIES

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
from askkit.neural import (
    Tables,
    slopes,
    softmax,
    start_tables,
    table_of,
    train_step,
)


def trained_on_ten_replies(steps: int) -> tuple[Tables, list[str]]:
    words = vocabulary(TEN_REPLIES)
    firsts, seconds = pairs(TEN_REPLIES, words)
    tables = start_tables(len(words), 3, np.random.default_rng(4417))
    for _ in range(steps):
        train_step(tables, firsts, seconds, 1.0)
    return tables, words


def trained_on_samples() -> tuple[Tables, list[str], list[list[str]]]:
    training, held_out = split_replies(read_replies(SAMPLES))
    words = vocabulary(training)
    firsts, seconds = pairs(training, words)
    tables = start_tables(len(words), 16, np.random.default_rng(4417))
    for _ in range(2000):
        train_step(tables, firsts, seconds, 1.0)
    return tables, words, training


def test_each_row_of_softmax_adds_up_to_one() -> None:
    scores = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    result = softmax(scores)
    assert np.allclose(result.sum(axis=1), 1.0)
    assert result[0] == pytest.approx([0.665, 0.245, 0.090], abs=0.001)


def test_softmax_does_not_overflow_on_a_large_score() -> None:
    result = softmax(np.array([[1000.0, 0.0]]))
    assert result[0] == pytest.approx([1.0, 0.0])


def test_small_random_tables_start_near_a_blind_guess() -> None:
    words = vocabulary(TEN_REPLIES)
    tables = start_tables(len(words), 3, np.random.default_rng(4417))
    loss = average_loss(table_of(tables), TEN_REPLIES, words)
    assert loss == pytest.approx(math.log(13), abs=0.01)


def test_no_two_starting_numbers_are_the_same() -> None:
    tables = start_tables(13, 3, np.random.default_rng(4417))
    numbers = np.concatenate([tables.embeddings, tables.outputs.T])
    assert len(np.unique(numbers)) == numbers.size


def test_backpropagation_finds_the_probes_slopes() -> None:
    # Lesson A1.1's probe: move one weight a little, run the whole model
    # again, put the weight back. Backpropagation must agree with it.
    words = vocabulary(TEN_REPLIES)
    firsts, seconds = pairs(TEN_REPLIES, words)
    tables = start_tables(len(words), 3, np.random.default_rng(4417))
    found = slopes(tables, firsts, seconds)
    probe = 0.0001
    for name in ("embeddings", "outputs"):
        weights = getattr(tables, name)
        measured = np.zeros_like(weights)
        before = average_loss(table_of(tables), TEN_REPLIES, words)
        for place in np.ndindex(weights.shape):
            weights[place] += probe
            after = average_loss(table_of(tables), TEN_REPLIES, words)
            weights[place] -= probe
            measured[place] = (after - before) / probe
        np.testing.assert_allclose(
            getattr(found, name), measured, atol=0.0001
        )


def test_a_training_step_lowers_the_loss() -> None:
    words = vocabulary(TEN_REPLIES)
    firsts, seconds = pairs(TEN_REPLIES, words)
    tables = start_tables(len(words), 3, np.random.default_rng(4417))
    before = average_loss(table_of(tables), TEN_REPLIES, words)
    train_step(tables, firsts, seconds, 1.0)
    after = average_loss(table_of(tables), TEN_REPLIES, words)
    assert after < before


def test_trained_on_ten_replies_your_gets_lesson_1_3_shares() -> None:
    tables, words = trained_on_ten_replies(3000)
    row = table_of(tables)[words.index("your")]
    shares = [row[words.index(word)] for word in ["order", "refund"]]
    assert shares == pytest.approx([0.5, 0.3], abs=0.03)
    assert row[words.index("parcel")] == pytest.approx(0.1, abs=0.03)
    assert row[words.index("mug")] == pytest.approx(0.1, abs=0.03)


def test_the_trained_model_beats_the_table_on_held_out_replies() -> None:
    tables, words, training = trained_on_samples()
    held_out = split_replies(read_replies(SAMPLES))[1]
    counted = probabilities(count_pairs(training, words), 0.01)
    neural_loss = average_loss(table_of(tables), held_out, words)
    table_loss = average_loss(counted, held_out, words)
    assert neural_loss == pytest.approx(1.98, abs=0.005)
    assert neural_loss < table_loss


def test_what_follows_three_is_shared_with_five() -> None:
    # "allow three" is in the training replies, "allow five" is not;
    # three and five come after the same words ("in three", "in five").
    tables, words, training = trained_on_samples()
    counted = probabilities(count_pairs(training, words), 0.01)
    allow, five = words.index("allow"), words.index("five")
    neural = table_of(tables)[allow, five]
    assert neural > 10 * counted[allow, five]


def test_the_same_seed_trains_and_writes_the_same_replies() -> None:
    tables, words = trained_on_ten_replies(100)
    again, _ = trained_on_ten_replies(100)
    first, second = np.random.default_rng(7), np.random.default_rng(7)
    assert [sample_reply(table_of(tables), words, first)] == [
        sample_reply(table_of(again), words, second)
    ]


def test_neural_prints_the_weights_the_losses_and_samples(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(["neural", str(SAMPLES), "--samples", "2"])
    lines = capsys.readouterr().out.splitlines()
    assert lines[:3] == [
        "Replies:       183 to learn from, 20 held out",
        "Vocabulary:    148 words",
        "Weights:       148 by 16 + 16 by 148 = 4,736",
    ]
    assert lines[3] == "Step      0:   training 5.00, held out 5.00"
    assert lines[7] == "Step   2000:   training 1.47, held out 1.98"
    assert lines[8] == (
        "Held out:      1.98 a word (the table: 2.04, a blind guess: 5.00)"
    )
    assert lines[9] == (
        "Samples:       seed 4417, 2000 steps at learning rate 1.0"
    )
    assert len(lines) == 12
    assert all(line.endswith(" .") for line in lines[10:])


def test_neural_refuses_a_learning_rate_of_zero() -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["neural", str(SAMPLES), "--learning-rate", "0"])
    assert stopped.value.code == "askkit: --learning-rate must be above 0"


def test_neural_refuses_a_width_of_zero() -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["neural", str(SAMPLES), "--width", "0"])
    assert stopped.value.code == (
        "askkit: --width must be 1 or more and --steps 0 or more"
    )
