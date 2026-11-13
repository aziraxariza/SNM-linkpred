import networkx as nx
import community as community_louvain
from sklearn.metrics import normalized_mutual_info_score


def louvain_partition(graph, resolution=1.0, seed=0):
    if graph.number_of_edges() == 0:
        return {n: i for i, n in enumerate(graph.nodes())}
    return community_louvain.best_partition(graph, resolution=resolution, random_state=seed)


def nmi_score(truth, predicted):
    nodes = sorted(set(truth) & set(predicted))
    return float(normalized_mutual_info_score([truth[n] for n in nodes],
                                              [predicted[n] for n in nodes]))


def modularity_score(graph, partition):
    groups = {}
    for node, cid in partition.items():
        groups.setdefault(cid, set()).add(node)
    return float(nx.community.modularity(graph, list(groups.values()))) if graph.number_of_edges() else 0.0
