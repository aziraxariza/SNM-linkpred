import math


METHODS = ("common_neighbors", "jaccard", "adamic_adar", "preferential_attachment")


def score_pairs(graph, pairs):
    deg = dict(graph.degree())
    neigh = {n: set(graph.neighbors(n)) for n in graph.nodes()}
    out = {m: [] for m in METHODS}
    for u, v in pairs:
        common = neigh[u] & neigh[v]
        union = neigh[u] | neigh[v]
        out["common_neighbors"].append(float(len(common)))
        out["jaccard"].append(float(len(common) / len(union)) if union else 0.0)
        out["adamic_adar"].append(float(sum(1 / math.log(deg[w]) for w in common if deg[w] > 1)))
        out["preferential_attachment"].append(float(deg[u] * deg[v]))
    return out
