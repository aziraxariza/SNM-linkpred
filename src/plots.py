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

    # Render clean, labelled methodology flowchart.
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7); ax.axis("off")
    boxes = [
        (0.4, 5.2, 2.2, 1.0, "SNAP datasets", "Static + timestamped"),
        (3.1, 5.2, 2.2, 1.0, "Preprocessing", "Deduplicate, label, project"),
        (5.8, 5.2, 2.2, 1.0, "Split design", "Random holdout / chronological"),
        (8.5, 5.2, 2.2, 1.0, "Training graph", "No validation/test edges"),
        (8.5, 3.1, 2.2, 1.0, "Louvain", "Detected groups from train"),
        (5.8, 3.1, 2.2, 1.0, "Link scores", "CN, Jaccard, AA, PA"),
        (3.1, 3.1, 2.2, 1.0, "Validation", "Tune community bonus"),
        (0.4, 3.1, 2.2, 1.0, "Test evaluation", "AUC, AP, Precision@K"),
        (3.1, 0.8, 2.2, 1.0, "Analysis", "Paired tests + uncertainty"),
        (6.0, 0.8, 2.2, 1.0, "Figures + tables", "CSV, PNG, diagram sources"),
        (8.9, 0.8, 2.2, 1.0, "Manuscript", "Results only from outputs"),
    ]
    centers = {}
    for x0, y0, w, h, title, sub in boxes:
        patch = FancyBboxPatch((x0,y0),w,h,boxstyle="round,pad=0.08,rounding_size=0.12",
                               facecolor="#edf3f8",edgecolor="#36566f",linewidth=1.4)
        ax.add_patch(patch); centers[title]=(x0+w/2,y0+h/2)
        ax.text(x0+w/2,y0+.64,title,ha="center",va="center",fontsize=10,weight="bold")
        ax.text(x0+w/2,y0+.31,sub,ha="center",va="center",fontsize=7.7,wrap=True)
    links=[("SNAP datasets","Preprocessing"),("Preprocessing","Split design"),
           ("Split design","Training graph"),("Training graph","Louvain"),
           ("Louvain","Link scores"),("Link scores","Validation"),
           ("Validation","Test evaluation"),("Test evaluation","Analysis"),
           ("Analysis","Figures + tables"),("Figures + tables","Manuscript")]
    for a,b in links:
        x1,y1=centers[a]; x2,y2=centers[b]
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle="-|>",mutation_scale=13,
                                     color="#526b7a",linewidth=1.3))
    ax.set_title("Research workflow: static reconstruction and temporal first appearance",
                 fontsize=14,weight="bold",pad=12)
    save(fig,out,"18_methodology_pipeline.png")

    # Common-neighbour explainer.
    fig, ax = plt.subplots(figsize=(8,5))
    pos={"A":(0,0),"B":(4,0),"C":(1.3,2),"D":(2.7,2)}
    edges=[("A","C"),("B","C"),("A","D"),("B","D")]
    demo=nx.Graph(edges)
    nx.draw_networkx_edges(demo,pos,ax=ax,width=2,edge_color="#607d8b")
    nx.draw_networkx_nodes(demo,pos,nodelist=["A","B"],node_color="#f4c66a",
                           node_size=1100,ax=ax,edgecolors="#654d18")
    nx.draw_networkx_nodes(demo,pos,nodelist=["C","D"],node_color="#9cc9df",
                           node_size=850,ax=ax,edgecolors="#31576b")
    nx.draw_networkx_labels(demo,pos,ax=ax,font_size=12,font_weight="bold")
    ax.plot([0,4],[0,0],linestyle="--",color="#c0392b",linewidth=2)
    ax.text(2,-.35,"candidate pair (A, B)",ha="center",color="#a52a2a",fontsize=10)
    ax.text(2,2.55,"2 shared neighbours: CN(A,B) = 2",ha="center",fontsize=11)
    ax.set_title("How common-neighbour link prediction works",fontsize=13,weight="bold")
    ax.axis("off")
    save(fig,out,"19_common_neighbour_explainer.png")

    # Department-strata explainer.
    fig, ax = plt.subplots(figsize=(9,5))
    ax.set_xlim(0,10); ax.set_ylim(0,5); ax.axis("off")
    for x0,title,color in [(0.5,"Department 1","#e5f1f7"),(6.0,"Department 2","#f8eee2")]:
        ax.add_patch(FancyBboxPatch((x0,.7),3.5,3.5,boxstyle="round,pad=.12",
                                    facecolor=color,edgecolor="#526b7a",linewidth=1.4))
        ax.text(x0+1.75,3.9,title,ha="center",fontsize=11,weight="bold")
    left=[(1.2,2.5),(2.1,1.6),(3.1,2.5)]
    right=[(6.8,2.5),(7.8,1.6),(8.8,2.5)]
    for group in (left,right):
        for i in range(3):
            for j in range(i+1,3):
                ax.plot([group[i][0],group[j][0]],[group[i][1],group[j][1]],color="#8799a5",lw=1.3)
        for x,y0 in group:
            ax.scatter(x,y0,s=230,c="#6baed6",edgecolors="#31576b",zorder=3)
    ax.plot([3.1,6.8],[2.5,2.5],linestyle="--",color="#c0392b",lw=2)
    ax.text(5,2.85,"inter-department candidate",ha="center",color="#a52a2a",fontsize=9)
    ax.text(2.25,.25,"intra-department pairs",ha="center",fontsize=9)
    ax.text(7.75,.25,"intra-department pairs",ha="center",fontsize=9)
    ax.set_title("Candidate-pair strata in the static experiment",fontsize=13,weight="bold")
    save(fig,out,"20_department_strata_explainer.png")
