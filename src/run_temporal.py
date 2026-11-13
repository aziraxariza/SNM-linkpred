"""Run a chronological first-appearance experiment on Email-Eu-core-temporal."""
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from src.temporal import load_temporal, first_appearance, chronological_split, negative_pairs, build_train_graph
from src.communities import louvain_partition
from src.predict import score_pairs, METHODS


def main():
    cfg = json.loads(Path("configs/default.json").read_text())
    events = load_temporal(cfg["temporal_path"])
    dyads = first_appearance(events)
    train, val, test, cutoffs = chronological_split(
        dyads, cfg["temporal_train_fraction"], cfg["temporal_validation_fraction"])
    graph = build_train_graph(train, set(events.src) | set(events.dst))
    partition = louvain_partition(graph, cfg["louvain_resolution"], seed=17)

    # Candidate negatives are dyads never observed in the full event file.
    # This is a closed-world benchmark; the limitation is documented.
    all_observed = set(map(tuple, dyads[["u", "v"]].itertuples(index=False, name=None)))
    rows = []
    for split_name, positives in [("validation", val), ("test", test)]:
        pos = list(map(tuple, positives[["u", "v"]].itertuples(index=False, name=None)))
        neg = negative_pairs(graph.nodes(), all_observed, len(pos), seed=1701 if split_name == "validation" else 1702)
        pairs = pos + neg
        y = [1] * len(pos) + [0] * len(neg)
        for method, scores in score_pairs(graph, pairs).items():
            rows.append({"split": split_name, "stratum": "all", "method": method,
                         "roc_auc": roc_auc_score(y, scores),
                         "average_precision": average_precision_score(y, scores),
                         "n_positive": len(pos), "n_negative": len(neg)})
            for group in ("same_detected_community", "different_detected_community"):
                keep = [i for i, (u, v) in enumerate(pairs)
                        if (partition.get(u) == partition.get(v)) == (group == "same_detected_community")]
                if not keep or len({y[i] for i in keep}) < 2:
                    continue
                rows.append({"split": split_name, "stratum": group, "method": method,
                             "roc_auc": roc_auc_score([y[i] for i in keep], [scores[i] for i in keep]),
                             "average_precision": average_precision_score([y[i] for i in keep], [scores[i] for i in keep]),
                             "n_positive": sum(y[i] for i in keep),
                             "n_negative": len(keep) - sum(y[i] for i in keep)})
    out = Path(cfg["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "temporal_metrics.csv", index=False)
    summary = {
        "source": "SNAP Email-Eu-core-temporal",
        "event_rows": int(len(events)),
        "unique_first_appearance_dyads": int(len(dyads)),
        "train_dyads": int(len(train)), "validation_dyads": int(len(val)),
        "test_dyads": int(len(test)), "cutoffs_unix_seconds": cutoffs,
        "detected_communities": len(set(partition.values())),
        "evaluation": "chronological first-appearance prediction; undirected dyads",
        "negative_sampling": "equal-count uniformly sampled dyads absent from complete observed dyad set"
    }
    (out / "temporal_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved {out / 'temporal_metrics.csv'}")


if __name__ == "__main__":
    main()
