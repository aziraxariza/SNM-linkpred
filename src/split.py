import random
import networkx as nx


def split_edges(graph, test_fraction, validation_fraction, seed):
    edges = [tuple(sorted(e)) for e in graph.edges()]
    rng = random.Random(seed)
    rng.shuffle(edges)
    nt = max(1, round(len(edges) * test_fraction))
    test = edges[:nt]
    rest = edges[nt:]
    nv = max(1, round(len(rest) * validation_fraction))
    val, train = rest[:nv], rest[nv:]
    train_graph = nx.Graph()
    train_graph.add_nodes_from(graph.nodes(data=True))
    train_graph.add_edges_from(train)
    return train_graph, train, val, test


def stratum(pair, labels):
    u, v = pair
    return "intra" if labels[u] == labels[v] else "inter"


def sample_negatives(full_graph, labels, positives, ratio, seed, max_attempt_factor=100):
    """Sample absent pairs by department stratum, excluding withheld positives."""
    rng = random.Random(seed)
    targets = {"intra": 0, "inter": 0}
    for edge in positives:
        targets[stratum(edge, labels)] += ratio
    picked = {"intra": set(), "inter": set()}
    nodes = list(full_graph.nodes())
    remaining = sum(targets.values())
    attempts = 0
    limit = max(10000, remaining * max_attempt_factor)

    while attempts < limit and any(len(picked[k]) < targets[k] for k in targets):
        u, v = rng.sample(nodes, 2)
        pair = tuple(sorted((u, v)))
        attempts += 1
        if u not in labels or v not in labels or full_graph.has_edge(*pair):
            continue
        kind = stratum(pair, labels)
        if len(picked[kind]) < targets[kind]:
            picked[kind].add(pair)

    for kind in targets:
        if len(picked[kind]) < targets[kind]:
            raise RuntimeError(
                f"Could sample only {len(picked[kind])} of {targets[kind]} {kind} negatives"
            )
    return {kind: sorted(pairs) for kind, pairs in picked.items()}


def split_by_stratum(edges, negatives, labels):
    return {
        kind: ([edge for edge in edges if stratum(edge, labels) == kind], negatives[kind])
        for kind in ("intra", "inter")
    }
