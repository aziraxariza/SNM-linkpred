import networkx as nx
from src.predict import score_pairs
from src.split import split_edges
from src.communities import nmi_score


def test_similarity_scores():
    g = nx.Graph([(0, 2), (1, 2), (0, 3), (1, 3)])
    s = score_pairs(g, [(0, 1)])
    assert s["common_neighbors"] == [2.0]
    assert s["jaccard"] == [1.0]
    assert s["preferential_attachment"] == [4.0]
    assert s["adamic_adar"][0] > 0


def test_edge_splits_disjoint():
    g = nx.cycle_graph(30)
    tg, tr, va, te = split_edges(g, .2, .2, 42)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    assert tg.number_of_edges() == len(tr)


def test_nmi_identical():
    p = {0: 0, 1: 0, 2: 1, 3: 1}
    assert nmi_score(p, p) == 1.0
