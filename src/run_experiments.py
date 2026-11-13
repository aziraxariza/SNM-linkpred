import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.data import load_email_eu_core, describe
from src.split import split_edges, sample_negatives, split_by_stratum
from src.predict import METHODS, score_pairs
from src.communities import louvain_partition, nmi_score, modularity_score
from src.metrics import evaluate
from src.plots import generate_all


def pair_dataset(graph, positives, negatives, labels, top_k):
    pairs = positives + negatives
    y = np.r_[np.ones(len(positives), dtype=int), np.zeros(len(negatives), dtype=int)]
    scores = score_pairs(graph, pairs)
    rows = {m: evaluate(y, s, top_k) for m, s in scores.items()}
    return rows, y, scores


def tune_bonus(graph, val_groups, partition, grid, top_k):
    best = {}
    for method in METHODS:
        best[method] = (float("-inf"), 0.0)
        for lam in grid:
            all_y, all_s = [], []
            for _, (pos, neg) in val_groups.items():
                pairs = pos + neg
                y = np.r_[np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)]
                base = score_pairs(graph, pairs)[method]
                adjusted = [s + (lam if partition.get(u) == partition.get(v) else 0.0)
                            for (u, v), s in zip(pairs, base)]
                all_y.extend(y.tolist())
                all_s.extend(adjusted)
            if len(set(all_y)) < 2:
                continue
            score = evaluate(all_y, all_s, top_k)["average_precision"]
            if np.isfinite(score) and score > best[method][0]:
                best[method] = (score, lam)
    return {m: {"validation_ap": v[0], "lambda": v[1]} for m, v in best.items()}


def main(config_file):
    cfg = json.loads(Path(config_file).read_text(encoding="utf-8"))
    directed, graph, labels = load_email_eu_core(
        cfg["edge_url"], cfg["label_url"], cfg["raw_dir"])
    results = Path(cfg["results_dir"])
    figures = Path(cfg["figures_dir"])
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    (results / "network_summary.json").write_text(
        json.dumps(describe(directed, graph, labels), indent=2), encoding="utf-8")

    metric_rows, community_rows, split_rows, tuning_rows, prediction_rows = [], [], [], [], []
    nmi_rows, modularity_rows = [], []

    for rep, seed in enumerate(cfg["seeds"][:cfg["repetitions"]], start=1):
        train_graph, train_edges, val_edges, test_edges = split_edges(
            graph, cfg["test_fraction"], cfg["validation_fraction"], seed)
        partition = louvain_partition(train_graph, cfg["louvain_resolution"], seed)
        nmi_rows.append({"rep": rep, "seed": seed, "nmi": nmi_score(labels, partition),
                         "detected_communities": len(set(partition.values()))})
        modularity_rows.append({"rep": rep, "seed": seed,
                                "modularity": modularity_score(train_graph, partition)})

        for split_name, edges in [("train", train_edges), ("validation", val_edges), ("test", test_edges)]:
            counts = {"intra": 0, "inter": 0}
            for u, v in edges:
                counts["intra" if labels[u] == labels[v] else "inter"] += 1
            for kind, count in counts.items():
                split_rows.append({"rep": rep, "split": split_name, "stratum": kind, "count": count})

        val_neg = sample_negatives(graph, labels, val_edges, cfg["negative_ratio"], seed + 10000)
        test_neg = sample_negatives(graph, labels, test_edges, cfg["negative_ratio"], seed + 20000)
        val_groups = split_by_stratum(val_edges, val_neg, labels)
        test_groups = split_by_stratum(test_edges, test_neg, labels)
        tuned = tune_bonus(train_graph, val_groups, partition, cfg["community_bonus_grid"], cfg["top_k"])
        for method, item in tuned.items():
            tuning_rows.append({"rep": rep, "method": method, **item})

        for stratum, (pos, neg) in test_groups.items():
            if not pos or not neg:
                continue
            rows, y, scores = pair_dataset(train_graph, pos, neg, labels, cfg["top_k"])
            pairs = pos + neg
            for method, values in rows.items():
                metric_rows.append({"rep": rep, "seed": seed, "stratum": stratum,
                                    "method": method, "n_positive": len(pos),
                                    "n_negative": len(neg),
                                    "positive_prevalence": len(pos) / (len(pos) + len(neg)), **values})
                base = scores[method]
                for (u, v), yy, ss in zip(pairs, y, base):
                    prediction_rows.append({"rep": rep, "stratum": stratum, "method": method,
                                            "label": int(yy), "score": float(ss)})
                lam = tuned[method]["lambda"]
                adjusted = [s + (lam if partition.get(u) == partition.get(v) else 0.0)
                            for (u, v), s in zip(pairs, base)]
                community_values = evaluate(y, adjusted, cfg["top_k"])
                metric_rows.append({"rep": rep, "seed": seed, "stratum": stratum,
                                    "method": method + "_community_aware",
                                    "n_positive": len(pos), "n_negative": len(neg),
                                    "positive_prevalence": len(pos) / (len(pos) + len(neg)),
                                    **community_values})

        # Community-wise within-department metrics; small/empty groups are omitted and recorded by absence.
        pos, neg = test_groups["intra"]
        for dept in sorted(set(labels.values())):
            p = [e for e in pos if labels[e[0]] == dept]
            n = [e for e in neg if labels[e[0]] == dept]
            if not p or not n:
                continue
            y = np.r_[np.ones(len(p), dtype=int), np.zeros(len(n), dtype=int)]
            pairs = p + n
            for method, values in score_pairs(train_graph, pairs).items():
                m = evaluate(y, values, cfg["top_k"])
                community_rows.append({"rep": rep, "department": dept,
                    "community_size": sum(v == dept for v in labels.values()),
                    "method": method, "average_precision": m["average_precision"],
                    "roc_auc": m["roc_auc"]})

    metrics = pd.DataFrame(metric_rows)
    communities = pd.DataFrame(community_rows)
    splits = pd.DataFrame(split_rows)
    tuning = pd.DataFrame(tuning_rows)
    nmi = pd.DataFrame(nmi_rows)
    modularity = pd.DataFrame(modularity_rows)

    metrics.to_csv(results / "test_metrics_by_run.csv", index=False)
    communities.to_csv(results / "community_metrics_by_run.csv", index=False)
    splits.to_csv(results / "split_stratum_counts.csv", index=False)
    tuning.to_csv(results / "validation_tuning.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(results / "test_prediction_scores.csv", index=False)
    nmi.to_csv(results / "louvain_nmi.csv", index=False)
    modularity.to_csv(results / "louvain_modularity.csv", index=False)

    summary = metrics.groupby(["method", "stratum"])[
        ["roc_auc", "average_precision", "precision_at_10"]].agg(["mean", "std", "count"])
    summary.to_csv(results / "metric_summary.csv")

    # Paired per-seed comparison: detected-community adjustment vs baseline.
    paired_rows = []
    for base_method in METHODS:
        for stratum in ("intra", "inter"):
            base = metrics[(metrics.method == base_method) & (metrics.stratum == stratum)]
            adjusted = metrics[(metrics.method == base_method + "_community_aware") &
                               (metrics.stratum == stratum)]
            merged = base.merge(adjusted, on=["rep", "seed", "stratum"],
                                suffixes=("_base", "_community"))
            for metric in ("roc_auc", "average_precision", "precision_at_10"):
                a = merged[metric + "_community"].dropna()
                b = merged.loc[a.index, metric + "_base"]
                if len(a) < 2:
                    continue
                diff = (a - b).to_numpy()
                if np.allclose(diff, 0):
                    stat, pvalue = 0.0, 1.0
                else:
                    test = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
                    stat, pvalue = float(test.statistic), float(test.pvalue)
                paired_rows.append({"method": base_method, "stratum": stratum,
                                    "metric": metric, "n_pairs": len(a),
                                    "baseline_mean": float(b.mean()),
                                    "community_mean": float(a.mean()),
                                    "mean_difference": float(diff.mean()),
                                    "median_difference": float(np.median(diff)),
                                    "wilcoxon_statistic": stat, "p_value": pvalue})
    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        # Benjamini-Hochberg correction across the reported tests.
        order = np.argsort(paired["p_value"].to_numpy())
        pvals = paired["p_value"].to_numpy()
        qvals = np.empty(len(pvals), dtype=float)
        running = 1.0
        for rank_idx in range(len(order) - 1, -1, -1):
            original_idx = order[rank_idx]
            running = min(running, pvals[original_idx] * len(pvals) / (rank_idx + 1))
            qvals[original_idx] = min(1.0, running)
        paired["p_value_bh"] = qvals
    paired.to_csv(results / "paired_comparisons.csv", index=False)
    generate_all(graph, labels, metrics, communities, splits, nmi, modularity, tuning,
                 pd.DataFrame(prediction_rows), figures)
    print(f"Completed {len(nmi)} seeded repetitions.")
    print(f"Tables: {results.resolve()}")
    print(f"Figures: {figures.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Community-stratified link prediction experiments")
    parser.add_argument("--config", default="configs/default.json")
    main(parser.parse_args().config)
