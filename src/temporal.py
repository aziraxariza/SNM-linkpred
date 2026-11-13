"""Chronological first-appearance link prediction for Email-Eu-core-temporal.

A dyad is positive at the first timestamp at which it appears. This avoids
counting repeated messages on an already-observed pair as newly formed links.
"""
from pathlib import Path
import gzip
import random
from urllib.request import urlretrieve
import networkx as nx
import pandas as pd


TEMPORAL_URL = "https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz"


def load_temporal(path="data/raw/email-Eu-core-temporal.txt.gz", url=TEMPORAL_URL):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urlretrieve(url, path)
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            parts = line.split()
            if len(parts) >= 3 and not parts[0].startswith("#"):
                u, v, ts = int(parts[0]), int(parts[1]), int(parts[2])
                if u != v:
                    rows.append((u, v, ts))
    frame = pd.DataFrame(rows, columns=["src", "dst", "timestamp"])
    if frame.empty:
        raise ValueError("Temporal dataset contains no valid three-column records")
    return frame


def first_appearance(frame):
    pairs = frame[frame.src != frame.dst].copy()
    pairs["u"] = pairs[["src", "dst"]].min(axis=1)
    pairs["v"] = pairs[["src", "dst"]].max(axis=1)
    return (pairs.groupby(["u", "v"], as_index=False)["timestamp"]
            .min().sort_values("timestamp").reset_index(drop=True))


def chronological_split(dyads, train_fraction=.70, validation_fraction=.15):
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train plus validation fraction must be below one")
    ordered = dyads.sort_values(["timestamp", "u", "v"]).reset_index(drop=True)
    n = len(ordered)
    i = max(1, int(n * train_fraction))
    j = max(i + 1, int(n * (train_fraction + validation_fraction)))
    if j >= n:
        raise ValueError("Not enough unique dyads for chronological train/validation/test")
    t_train = int(ordered.iloc[i - 1].timestamp)
    t_val = int(ordered.iloc[j - 1].timestamp)
    train = ordered[ordered.timestamp <= t_train]
    val = ordered[(ordered.timestamp > t_train) & (ordered.timestamp <= t_val)]
    test = ordered[ordered.timestamp > t_val]
    return train, val, test, {"train_end": t_train, "validation_end": t_val}


def negative_pairs(nodes, known_dyads, count, seed):
    rng = random.Random(seed)
    nodes = list(nodes)
    known = set(known_dyads)
    negatives = set()
    limit = max(10000, count * 100)
    attempts = 0
    while len(negatives) < count and attempts < limit:
        u, v = rng.sample(nodes, 2)
        pair = (min(u, v), max(u, v))
        attempts += 1
        if pair not in known:
            negatives.add(pair)
    if len(negatives) < count:
        raise RuntimeError(f"Could sample only {len(negatives)} of {count} negatives")
    return sorted(negatives)


def build_train_graph(train, nodes):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(train[["u", "v"]].itertuples(index=False, name=None))
    return graph
