# Figure catalogue

The static pipeline generates data-derived figures after execution:

1. Degree distribution
2. Log-log degree frequency
3. Degree complementary cumulative distribution
4. Department size distribution
5. Intra-/inter-department edge counts across train, validation and test
6. ROC-AUC distributions by method and stratum
7. Average-precision distributions by method and stratum
8. Precision@10 distributions
9. Intra-department ROC curve
10. Intra-department precision-recall curve
11. Inter-department ROC curve
12. Inter-department precision-recall curve
13. Louvain/department NMI distribution
14. Louvain modularity distribution
15. Department-level average-precision heatmap
16. Department size versus average precision
17. Validation-selected community bonus
18. ROC-AUC means with approximate 95% t intervals
19. An illustrative induced subgraph of high-degree nodes coloured by department
20. Mermaid methodology diagram source
21. Graphviz common-neighbour prediction diagram source

Generated figures are stored in `paper/figures/`. The temporal pipeline writes `results/temporal_metrics.csv` and `results/temporal_summary.json`; temporal plotting can be added after reviewing those results. Do not create curves or tables by hand from expected outcomes.
