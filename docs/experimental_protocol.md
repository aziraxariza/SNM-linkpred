# Experimental protocol

## Static Email-Eu-core task
The SNAP Email-Eu-core edge list is directed and department labels assign each node to one of 42 departments. The primary heuristic task uses an undirected simple projection: an undirected tie exists if either direction was observed. Remove self-loops and collapse duplicate ties. Retain the directed graph for descriptive counts.

## Static split and leakage control
For each fixed seed, shuffle the observed undirected positive edges and partition them into train, validation and test subsets. The training graph contains training positives only. Negative candidates must be absent from the complete observed graph, so a withheld positive is never sampled as a negative. Sample negatives separately for intra- and inter-department strata, without replacement, at the configured ratio. The run aborts if it cannot obtain the requested stratum count.

Louvain is run on the training graph only. Tune the community bonus on validation average precision, then evaluate once on the test subset. The bonus uses detected Louvain membership; supplied department labels are used to define evaluation strata and compare partitions, not as the predictor's input.

## Temporal Email-Eu-core task
The temporal dataset contains directed timestamped email events. Convert each event to an unordered dyad and retain the earliest timestamp for that dyad. This first-appearance representation prevents repeated emails on an already-existing dyad from being counted as new links. Sort unique dyads by first timestamp and split chronologically into train, validation and test periods. Fit Louvain on the training graph only.

The temporal dataset's node identifiers do not match those of the static Email-Eu-core graph. Do not transfer department labels from the static dataset. Temporal strata are based on detected training communities.

Temporal negatives are sampled from dyads absent from the complete observed temporal dataset. This closed-world definition is appropriate only for a retrospective benchmark and uses information from the full observation window to identify never-observed pairs. Report it explicitly; do not describe it as a live prospective negative sampler.

## Metrics
Report ROC-AUC, average precision and Precision@K for static runs, separately by intra-/inter-department stratum. Include positive/negative counts and candidate prevalence. Temporal output reports ROC-AUC and average precision for the full candidate set and detected-community strata. Report means, standard deviations and approximate confidence intervals across static seeds. Compare each baseline with its detected-community variant using paired Wilcoxon signed-rank tests on matched seeds, and apply Benjamini--Hochberg correction across the reported comparisons. Do not compare average precision across strata without showing prevalence.

## Reproducibility
The seeds and split fractions are in `configs/default.json`. Keep raw data and generated output out of Git by default. Record data source, retrieval date, file checksum, software versions and configuration for a publication run. Use `pytest -q` before experiments.

## Interpretation limits
Static random edge holdout measures missing-edge reconstruction, not prediction of future edges. The temporal experiment measures first appearance of undirected dyads, not message volume or repeated interaction. The undirected projections discard direction. One organizational network cannot establish generality. Department labels and Louvain communities represent different notions of grouping.
