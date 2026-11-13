import math

METHODS = (
    "common_neighbors",
    "jaccard",
    "adamic_adar",
    "preferential_attachment",
    "resource_allocation",
    "leicht_holme_newman",
)


def score_pairs(graph, pairs):
    deg = dict(graph.degree())
    neigh = {n: set(graph.neighbors(n)) for n in graph.nodes()}
    out = {method: [] for method in METHODS}
    for u, v in pairs:
        common = neigh[u] & neigh[v]
        union = neigh[u] | neigh[v]
        du, dv = deg[u], deg[v]
        out["common_neighbors"].append(float(len(common)))
        out["jaccard"].append(float(len(common) / len(union)) if union else 0.0)
        out["adamic_adar"].append(float(
            sum(1 / math.log(deg[w]) for w in common if deg[w] > 1)))
        out["preferential_attachment"].append(float(du * dv))
        out["resource_allocation"].append(float(
            sum(1 / deg[w] for w in common if deg[w] > 0)))
        out["leicht_holme_newman"].append(
            float(len(common) / (du * dv)) if du and dv else 0.0)
    return out
