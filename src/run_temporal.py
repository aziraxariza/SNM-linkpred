"""Chronological first-appearance experiment on SNAP Email-Eu-core-temporal."""
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from src.temporal import load_temporal, first_appearance, chronological_split, negative_pairs, build_train_graph
from src.communities import louvain_partition
from src.predict import score_pairs


def main():
    cfg = json.loads(Path("configs/default.json").read_text())
    events = load_temporal(cfg["temporal_path"])
    dyads = first_appearance(events)
    train, val, test, cutoffs = chronological_split(
        dyads, cfg["temporal_train_fraction"], cfg["temporal_validation_fraction"])
    graph = build_train_graph(train, set(events.src) | set(events.dst))
    partition = louvain_partition(graph, cfg["louvain_resolution"], seed=17)
    # Closed-world negatives: dyads absent from the complete observed event file.
    observed = set(map(tuple, dyads[["u", "v"]].itertuples(index=False, name=None)))
    rows, score_rows = [], []
    for split_name, positive_frame, seed in [
        ("validation", val, 1701), ("test", test, 1702)
    ]:
        positives = list(map(tuple, positive_frame[["u", "v"]].itertuples(index=False, name=None)))
        negatives = negative_pairs(graph.nodes(), observed, len(positives), seed)
        pairs = positives + negatives
        y = [1] * len(positives) + [0] * len(negatives)
        for method, scores in score_pairs(graph, pairs).items():
            rows.append({
                "split": split_name, "stratum": "all", "method": method,
                "roc_auc": roc_auc_score(y, scores),
                "average_precision": average_precision_score(y, scores),
                "n_positive": len(positives), "n_negative": len(negatives),
                "positive_prevalence": len(positives) / (len(positives) + len(negatives))
            })
            for (u, v), label, score in zip(pairs, y, scores):
                score_rows.append({"split": split_name, "stratum": "all", "method": method,
                                   "u": u, "v": v, "label": label, "score": float(score)})
            for same in (True, False):
                ids = [i for i, (u, v) in enumerate(pairs)
                       if (partition.get(u) == partition.get(v)) == same]
                if ids and len({y[i] for i in ids}) == 2:
                    yy = [y[i] for i in ids]
                    ss = [scores[i] for i in ids]
                    stratum = "same_detected_community" if same else "different_detected_community"
                    rows.append({
                        "split": split_name, "stratum": stratum, "method": method,
                        "roc_auc": roc_auc_score(yy, ss),
                        "average_precision": average_precision_score(yy, ss),
                        "n_positive": sum(yy), "n_negative": len(yy) - sum(yy),
                        "positive_prevalence": sum(yy) / len(yy)
                    })
                    for i in ids:
                        u, v = pairs[i]
                        score_rows.append({"split": split_name, "stratum": stratum,
                                           "method": method, "u": u, "v": v,
                                           "label": y[i], "score": float(scores[i])})

    out = Path(cfg["results_dir"])
    figures = Path(cfg["figures_dir"])
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    predictions = pd.DataFrame(score_rows)
    metrics.to_csv(out / "temporal_metrics.csv", index=False)
    predictions.to_csv(out / "temporal_prediction_scores.csv", index=False)

    # Test-set ROC and PR curves for all temporal candidate strata.
    test_scores = predictions[predictions.split == "test"]
    for stratum in ("all", "same_detected_community", "different_detected_community"):
        sub = test_scores[test_scores.stratum == stratum]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        for method, group in sub.groupby("method"):
            if group.label.nunique() < 2:
                continue
            fpr, tpr, _ = roc_curve(group.label, group.score)
            ax.plot(fpr, tpr, label=method.replace("_", " "))
        ax.plot([0, 1], [0, 1], "--", label="Random")
        ax.set(title=f"Temporal first-appearance ROC: {stratum}",
               xlabel="False positive rate", ylabel="True positive rate")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f"21_temporal_roc_{stratum}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 5))
        for method, group in sub.groupby("method"):
            if group.label.nunique() < 2:
                continue
            precision, recall, _ = precision_recall_curve(group.label, group.score)
            ax.plot(recall, precision, label=method.replace("_", " "))
        prevalence = float(sub.label.mean())
        ax.axhline(prevalence, linestyle="--", label="Candidate prevalence")
        ax.set(title=f"Temporal first-appearance PR: {stratum}",
               xlabel="Recall", ylabel="Precision")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f"22_temporal_pr_{stratum}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    summary = {
        "dataset": "SNAP Email-Eu-core temporal",
        "event_rows": int(len(events)),
        "unique_first_appearance_dyads": int(len(dyads)),
        "train_dyads": int(len(train)),
        "validation_dyads": int(len(val)),
        "test_dyads": int(len(test)),
        "cutoffs_unix_seconds": cutoffs,
        "detected_communities": len(set(partition.values())),
        "task": "chronological first-appearance prediction on undirected dyads",
        "negative_sampling": "equal-count random dyads absent from all observed events (closed-world)"
    }
    (out / "temporal_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
