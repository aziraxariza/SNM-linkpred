from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics import roc_curve, precision_recall_curve


def save(fig, folder, name):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(folder / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_all(graph, labels, metrics, communities, splits, nmi, modularity, tuning, out):
    out = Path(out)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)

    # 1: Degree histogram
    deg = np.array([d for _, d in graph.degree()])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(deg, bins=30, edgecolor="black", linewidth=.4)
    ax.set(title="Degree distribution", xlabel="Degree", ylabel="Number of nodes")
    save(fig, out, "01_degree_distribution.png")

    # 2: Log-log degree frequency
    values, counts = np.unique(deg[deg > 0], return_counts=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(values, counts, "o")
    ax.set(title="Degree frequency on logarithmic axes", xlabel="Degree (log)", ylabel="Node count (log)")
    save(fig, out, "02_degree_loglog.png")

    # 3: CCDF
    x = np.unique(deg)
    y = np.array([(deg >= k).mean() for k in x])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(np.maximum(x, 1), y, marker="o")
    ax.set(title="Degree complementary cumulative distribution", xlabel="Degree k", ylabel="P(Degree >= k)")
    save(fig, out, "03_degree_ccdf.png")

    # 4: Community sizes
    sizes = pd.Series(list(labels.values())).value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(1, len(sizes)+1), sizes.values)
    ax.set(title="Department size distribution", xlabel="Department (ranked by size)", ylabel="Number of nodes")
    save(fig, out, "04_department_sizes.png")

    # 5: Edge strata by split
    if not splits.empty:
        tab = splits.groupby(["split", "stratum"])["count"].mean().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(7, 4))
        tab.plot(kind="bar", ax=ax)
        ax.set(title="Intra- and inter-department edges by split", xlabel="Split", ylabel="Mean edge count")
        save(fig, out, "05_edge_strata_by_split.png")

    # 6/7: performance boxplots
    for metric, fname in [("roc_auc", "06_roc_auc_boxplots.png"),
                          ("average_precision", "07_average_precision_boxplots.png")]:
        if not metrics.empty:
            fig, ax = plt.subplots(figsize=(11, 5))
            sns.boxplot(data=metrics, x="method", y=metric, hue="stratum", ax=ax)
            ax.set(title=f"{metric.replace('_', ' ').title()} by method and edge stratum",
                   xlabel="Method", ylabel=metric.replace("_", " ").title())
            ax.tick_params(axis="x", rotation=25)
            save(fig, out, fname)

    # 8: ROC curve panels for each stratum, aggregated by interpolated TPR.
    for stratum in ("intra", "inter"):
        fig, ax = plt.subplots(figsize=(6, 5))
        subset = metrics[metrics.stratum == stratum] if not metrics.empty else pd.DataFrame()
        # Per-run scores are not retained; use the AUC distribution figure instead of fabricating curves.
        ax.text(.5, .5, "ROC coordinates are not exported in this run.\nSee per-run ROC-AUC distributions.",
                ha="center", va="center", transform=ax.transAxes)
        ax.set(title=f"ROC analysis: {stratum}-community candidates", xlabel="False positive rate", ylabel="True positive rate")
        save(fig, out, f"08_{stratum}_roc_placeholder.png")

    # 9: NMI
    if not nmi.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=nmi, x="nmi", bins=10, ax=ax)
        ax.set(title="Agreement of Louvain partition with departments", xlabel="Normalized mutual information", ylabel="Repetitions")
        save(fig, out, "09_louvain_nmi.png")

    # 10: modularity
    if not modularity.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=modularity, x="modularity", bins=10, ax=ax)
        ax.set(title="Louvain modularity across runs", xlabel="Modularity", ylabel="Repetitions")
        save(fig, out, "10_louvain_modularity.png")

    # 11: community heatmap
    if not communities.empty:
        pivot = communities.pivot_table(index="department", columns="method", values="average_precision")
        fig, ax = plt.subplots(figsize=(9, max(5, .22 * len(pivot))))
        sns.heatmap(pivot, cmap="viridis", ax=ax, cbar_kws={"label": "Average precision"})
        ax.set(title="Within-department prediction performance", xlabel="Method", ylabel="Department label")
        save(fig, out, "11_community_heatmap.png")

        # 12: size-performance scatter
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=communities, x="community_size", y="average_precision", hue="method", ax=ax)
        ax.set(title="Department size and link prediction performance",
               xlabel="Department size (nodes)", ylabel="Average precision")
        save(fig, out, "12_community_size_vs_ap.png")

    # 13: tuned community bonus
    if not tuning.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=tuning, x="method", y="lambda", ax=ax)
        ax.set(title="Validation-selected community bonus", xlabel="Baseline method", ylabel="Selected lambda")
        ax.tick_params(axis="x", rotation=25)
        save(fig, out, "13_selected_community_bonus.png")

    # 14: forest-style confidence intervals
    if not metrics.empty:
        base = metrics.groupby(["method", "stratum"])["roc_auc"].agg(["mean", "std", "count"]).reset_index()
        base["se"] = base["std"] / np.sqrt(base["count"].clip(lower=1))
        base = base.dropna(subset=["mean"])
        if not base.empty:
            fig, ax = plt.subplots(figsize=(9, 6))
            yy = np.arange(len(base))
            ax.errorbar(base["mean"], yy, xerr=1.96 * base["se"], fmt="o", capsize=3)
            ax.set_yticks(yy, [f"{r.method} | {r.stratum}" for r in base.itertuples()])
            ax.set(title="ROC-AUC means with approximate 95% confidence intervals", xlabel="ROC-AUC")
            save(fig, out, "14_auc_confidence_intervals.png")

    # 15: Network workflow as vector diagram (Mermaid source)
    (out / "15_methodology_pipeline.mmd").write_text("""flowchart TD
 A[SNAP Email-Eu-core] --> B[Parse edge list and department labels]
 B --> C[Directed graph and undirected projection]
 C --> D[Network characterization]
 D --> E[Seeded train validation test split]
 E --> F[Louvain on training graph]
 F --> G[CN / Jaccard / Adamic-Adar / Preferential Attachment]
 G --> H[Intra and inter department evaluation]
 H --> I[Validation tuning of community bonus]
 I --> J[Held-out test metrics]
 J --> K[CSV tables and figures]
""", encoding="utf-8")

    # 16: Graphical explanation diagram in Graphviz DOT source
    (out / "16_link_prediction_concept.dot").write_text("""graph LinkPrediction {
  graph [label="Illustration of common-neighbour link prediction", labelloc=t, fontsize=18];
  node [shape=circle, style=filled, fillcolor=lightblue];
  A -- C; B -- C; A -- D; B -- D;
  A [fillcolor=lightgoldenrod1]; B [fillcolor=lightgoldenrod1];
  C; D;
  A -- B [style=dashed, color=red, label="candidate missing edge"];
}
""", encoding="utf-8")
