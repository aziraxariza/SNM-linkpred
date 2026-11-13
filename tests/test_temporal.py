import pandas as pd
from src.temporal import first_appearance, chronological_split, negative_pairs


def test_first_appearance_collapses_repeated_and_reverse_edges():
    frame = pd.DataFrame([
        (1, 2, 10), (1, 2, 20), (2, 1, 30), (2, 3, 15)
    ], columns=["src", "dst", "timestamp"])
    dyads = first_appearance(frame)
    assert len(dyads) == 2
    assert list(dyads.timestamp) == [10, 15]


def test_chronological_split_is_ordered():
    dyads = pd.DataFrame({
        "u": range(20), "v": range(1, 21), "timestamp": range(20)
    })
    train, val, test, cuts = chronological_split(dyads, .6, .2)
    assert train.timestamp.max() < val.timestamp.min()
    assert val.timestamp.max() < test.timestamp.min()
    assert cuts["train_end"] < cuts["validation_end"]


def test_negative_pairs_not_observed():
    nodes = [0, 1, 2, 3, 4]
    known = {(0, 1), (1, 2)}
    neg = negative_pairs(nodes, known, 3, 42)
    assert len(neg) == 3
    assert not (set(neg) & known)
