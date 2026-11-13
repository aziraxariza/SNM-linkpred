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


def generate_all(graph, labels, metrics, communities, splits, nmi, modularity,
                 tuning, predictions, out):
    out = Path(out)
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)
    degree = np.array([d for _, d in graph.degree()])

    # 01. Degree distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(degree, bins=30, edgecolor="black", linewidth=.4)
    ax.set(title="Degree distribution", xlabel="Degree", ylabel="Number of nodes")
    save(fig, out, "01_degree_distribution.png")

    # 02. Log-log degree frequency
    values, counts = np.unique(degree[degree > 0], return_counts=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(values, counts, "o")
    ax.set(title="Degree frequency (log-log)", xlabel="Degree", ylabel="Node frequency")
    save(fig, out, "02_degree_loglog.png")

    # 03. Degree complementary CDF
    x = np.unique(degree)
    y = np.array([(degree >= k).mean() for k in x])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(np.maximum(x, 1), y, marker="o")
    ax.set(title="Degree complementary cumulative distribution",
           xlabel="Degree k", ylabel="P(Degree >= k)")
    save(fig, out, "03_degree_ccdf.png")

    # 04. Department sizes
    sizes = pd.Series(list(labels.values())).value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(1, len(sizes) + 1), sizes.values)
    ax.set(title="Department size distribution", xlabel="Department (ranked)",
           ylabel="Number of nodes")
    save(fig, out, "04_department_sizes.png")

    # 05. Intra/inter edge counts per split
    if not splits.empty:
        tab = splits.groupby(["split", "stratum"])["count"].mean().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(7, 4))
        tab.plot(kind="bar", ax=ax)
        ax.set(title="Edge strata across data splits", xlabel="Split", ylabel="Mean edge count")
        save(fig, out, "05_edge_strata_by_split.png")

    # 06-07. Repeated-run metric distributions
    if not metrics.empty:
        for metric, filename in [
            ("roc_auc", "06_roc_auc_boxplots.png"),
            ("average_precision", "07_average_precision_boxplots.png"),
            ("precision_at_10", "08_precision_at_10.png"),
        ]:
            if metric not in metrics.columns:
                continue
            fig, ax = plt.subplots(figsize=(11, 5))
            sns.boxplot(data=metrics, x="method", y=metric, hue="stratum", ax=ax)
            ax.set(title=f"{metric.replace('_', ' ').title()} by method and stratum",
                   xlabel="Method", ylabel=metric.replace("_", " ").title())
            ax.tick_params(axis="x", rotation=25)
            save(fig, out, filename)

    # 09-12. ROC and PR curves; mean curves across independent repetitions.
    if not predictions.empty:
        for stratum in ("intra", "inter"):
            sub = predictions[predictions.stratum == stratum]
            fig, ax = plt.subplots(figsize=(6.5, 5.2))
            grid = np.linspace(0, 1, 201)
            for method, group in sub.groupby("method"):
                curves = []
                for _, run in group.groupby("rep"):
                    if run.label.nunique() < 2:
                        continue
                    fpr, tpr, _ = roc_curve(run.label, run.score)
                    curves.append(np.interp(grid, fpr, tpr))
                if curves:
                    ax.plot(grid, np.mean(curves, axis=0), label=method.replace("_", " "))
            ax.plot([0, 1], [0, 1], "--", label="Random")
            ax.set(title=f"ROC curve: {stratum}-department candidates",
                   xlabel="False positive rate", ylabel="True positive rate")
            ax.legend(fontsize=8)
            save(fig, out, f"09_{stratum}_roc_curve.png")

            fig, ax = plt.subplots(figsize=(6.5, 5.2))
            for method, group in sub.groupby("method"):
                curves = []
                for _, run in group.groupby("rep"):
                    if run.label.nunique() < 2:
                        continue
                    precision, recall, _ = precision_recall_curve(run.label, run.score)
                    curves.append(np.interp(grid, recall[::-1], precision[::-1]))
                if curves:
                    ax.plot(grid, np.mean(curves, axis=0), label=method.replace("_", " "))
            prevalence = float(sub.label.mean()) if len(sub) else 0
            ax.axhline(prevalence, linestyle="--", label="Candidate prevalence")
            ax.set(title=f"Precision-recall curve: {stratum}-department candidates",
                   xlabel="Recall", ylabel="Precision")
            ax.legend(fontsize=8)
            save(fig, out, f"10_{stratum}_precision_recall.png")

    # 13. Louvain agreement with department metadata
    if not nmi.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=nmi, x="nmi", bins=10, ax=ax)
        ax.set(title="Louvain vs department-label agreement",
               xlabel="Normalized mutual information", ylabel="Repetitions")
        save(fig, out, "11_louvain_nmi.png")

    # 14. Modularity distribution
    if not modularity.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=modularity, x="modularity", bins=10, ax=ax)
        ax.set(title="Louvain modularity across runs", xlabel="Modularity", ylabel="Repetitions")
        save(fig, out, "12_louvain_modularity.png")

    # 15-16. Department heatmap and size-performance relationship
    if not communities.empty:
        pivot = communities.pivot_table(index="department", columns="method",
                                        values="average_precision")
        fig, ax = plt.subplots(figsize=(9, max(5, .22 * len(pivot))))
        sns.heatmap(pivot, cmap="viridis", ax=ax,
                    cbar_kws={"label": "Average precision"})
        ax.set(title="Within-department prediction performance",
               xlabel="Method", ylabel="Department label")
        save(fig, out, "13_community_heatmap.png")

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=communities, x="community_size", y="average_precision",
                        hue="method", ax=ax)
        ax.set(title="Department size and prediction performance",
               xlabel="Department size (nodes)", ylabel="Average precision")
        save(fig, out, "14_community_size_vs_ap.png")

    # 17. Validation-selected community bonus
    if not tuning.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=tuning, x="method", y="lambda", ax=ax)
        ax.set(title="Validation-selected community bonus",
               xlabel="Baseline method", ylabel="Selected lambda")
        ax.tick_params(axis="x", rotation=25)
        save(fig, out, "15_selected_community_bonus.png")

    # 18. Mean AUC with t-based approximate intervals
    if not metrics.empty:
        from scipy.stats import t
        base = metrics.groupby(["method", "stratum"])["roc_auc"].agg(
            ["mean", "std", "count"]).reset_index()
        base = base.dropna(subset=["mean"])
        if not base.empty:
            se = base["std"] / np.sqrt(base["count"].clip(lower=1))
            crit = [t.ppf(.975, max(int(n) - 1, 1)) if n > 1 else np.nan
                    for n in base["count"]]
            err = se * np.asarray(crit)
            fig, ax = plt.subplots(figsize=(9, 6))
            yy = np.arange(len(base))
            ax.errorbar(base["mean"], yy, xerr=err, fmt="o", capsize=3)
            ax.set_yticks(yy, [f"{r.method} | {r.stratum}" for r in base.itertuples()])
            ax.set(title="ROC-AUC mean and approximate 95% t intervals", xlabel="ROC-AUC")
            save(fig, out, "16_auc_confidence_intervals.png")

    # 19. Department-labelled graph sample for visual explanation
    largest = max(nx.connected_components(graph), key=len)
    sample_nodes = sorted(largest, key=lambda n: graph.degree(n), reverse=True)[:120]
    sg = graph.subgraph(sample_nodes)
    pos = nx.spring_layout(sg, seed=17, k=0.35)
    fig, ax = plt.subplots(figsize=(10, 8))
    dept_values = [labels.get(n, -1) for n in sg.nodes()]
    nx.draw_networkx_edges(sg, pos, ax=ax, alpha=.15, width=.45)
    nx.draw_networkx_nodes(sg, pos, ax=ax, node_size=18, node_color=dept_values,
                           cmap="tab20", linewidths=0)
    ax.set(title="Illustrative induced subgraph: 120 highest-degree nodes")
    ax.axis("off")
    save(fig, out, "17_network_sample.png")

    # 20. Methodology diagram (Mermaid source)
    (out / "18_methodology_pipeline.mmd").write_text("""flowchart TD
 A[SNAP Email-Eu-core] --> B[Parse directed edges and department labels]
 B --> C[Build undirected projection]
 C --> D[Network characterization]
 D --> E[Seeded train validation test split]
 E --> F[Louvain on training graph only]
 F --> G[Structural link scores]
 G --> H[Intra/inter department candidate evaluation]
 H --> I[Validation tuning]
 I --> J[Held-out test metrics]
 J --> K[Tables, confidence intervals and figures]
""", encoding="utf-8")

    # 21. Candidate scoring concept (Graphviz source)
    (out / "19_link_prediction_concept.dot").write_text("""graph LinkPrediction {
 graph [label="Common-neighbour link prediction", labelloc=t, fontsize=18];
 node [shape=circle, style=filled, fillcolor=lightblue];
 A -- C; B -- C; A -- D; B -- D;
 A [fillcolor=lightgoldenrod1]; B [fillcolor=lightgoldenrod1];
 C; D;
 A -- B [style=dashed, color=red, label="candidate missing edge"];
}
""", encoding="utf-8")
