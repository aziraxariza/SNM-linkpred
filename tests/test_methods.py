import networkx as nx
from src.predict import score_pairs, METHODS
from src.split import split_edges
from src.communities import nmi_score


def test_similarity_scores():
    g = nx.Graph([(0, 2), (1, 2), (0, 3), (1, 3)])
    scores = score_pairs(g, [(0, 1)])
    assert scores["common_neighbors"] == [2.0]
    assert scores["jaccard"] == [1.0]
    assert scores["preferential_attachment"] == [4.0]
    assert scores["resource_allocation"] == [1.0]
    assert scores["leicht_holme_newman"] == [0.5]
    assert scores["adamic_adar"][0] > 0
    assert set(scores) == set(METHODS)


def test_edge_splits_disjoint():
    graph = nx.cycle_graph(30)
    train_graph, train, val, test = split_edges(graph, .2, .2, 42)
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    assert train_graph.number_of_edges() == len(train)


def test_nmi_identical():
    partition = {0: 0, 1: 0, 2: 1, 3: 1}
    assert nmi_score(partition, partition) == 1.0
