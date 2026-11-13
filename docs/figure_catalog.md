# Figure catalogue

The static pipeline creates 20+ publication and explanatory assets after execution.

| Output | Content |
|---|---|
| 01 | Degree distribution |
| 02 | Log-log degree frequency |
| 03 | Degree complementary cumulative distribution |
| 04 | Department size distribution |
| 05 | Intra/inter department edge counts across data splits |
| 06 | ROC-AUC boxplots by method and stratum |
| 07 | Average-precision boxplots |
| 08 | Precision@10 distributions |
| 09 | Intra-department ROC curve |
| 10 | Intra-department precision-recall curve |
| 09/10 for inter | Inter-department ROC and precision-recall curves |
| 11 | Louvain/department NMI distribution |
| 12 | Louvain modularity distribution |
| 13 | Department-level average-precision heatmap |
| 14 | Department size versus average precision |
| 15 | Validation-selected community bonus |
| 16 | ROC-AUC mean and approximate 95% t intervals |
| 17 | Illustrative induced subgraph of high-degree nodes coloured by department |
| 18 | Rendered methodology flowchart |
| 19 | Common-neighbour explainer diagram |
| 20 | Intra-/inter-department candidate strata diagram |

The pipeline also writes Mermaid and Graphviz source diagrams. Generated assets are saved in `paper/figures/`. Temporal outputs are `results/temporal_metrics.csv` and `results/temporal_summary.json`; no temporal plots are claimed by the current pipeline.

All empirical plots are generated from run outputs. The conceptual diagrams are explicitly illustrative and must not be presented as measured network results.
