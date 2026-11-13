from pathlib import Path
import gzip
from urllib.request import urlretrieve
import networkx as nx


def fetch(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        urlretrieve(url, dest)
    return dest


def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8")


def load_email_eu_core(edge_url, label_url, raw_dir):
    ep = fetch(edge_url, Path(raw_dir) / "email-Eu-core.txt.gz")
    lp = fetch(label_url, Path(raw_dir) / "email-Eu-core-department-labels.txt.gz")
    edges, labels = [], {}
    with open_text(ep) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2 and not p[0].startswith("#"):
                edges.append((int(p[0]), int(p[1])))
    with open_text(lp) as f:
        for line in f:
            p = line.split()
            if len(p) >= 2 and not p[0].startswith("#"):
                labels[int(p[0])] = int(p[1])
    dg = nx.DiGraph()
    dg.add_edges_from(edges)
    dg.remove_edges_from(nx.selfloop_edges(dg))
    dg.add_nodes_from(labels)
    ug = nx.Graph()
    ug.add_nodes_from(dg.nodes())
    ug.add_edges_from((u, v) for u, v in dg.edges() if u != v)
    nx.set_node_attributes(dg, labels, "department")
    nx.set_node_attributes(ug, labels, "department")
    return dg, ug, labels


def describe(dg, ug, labels):
    comps = list(nx.connected_components(ug))
    return {
        "directed_nodes": dg.number_of_nodes(),
        "directed_edges": dg.number_of_edges(),
        "undirected_nodes": ug.number_of_nodes(),
        "undirected_edges": ug.number_of_edges(),
        "department_count": len(set(labels.values())),
        "density": nx.density(ug),
        "average_clustering": nx.average_clustering(ug),
        "connected_components": len(comps),
        "largest_component_size": max(map(len, comps), default=0),
        "isolates": nx.number_of_isolates(ug)
    }
