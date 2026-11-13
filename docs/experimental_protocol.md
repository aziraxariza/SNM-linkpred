# Experimental protocol

## Dataset and graph representation
Download the edge list and department labels from https://snap.stanford.edu/data/email-Eu-core.html. Keep the original directed graph for descriptive counts. The primary heuristic analysis uses an undirected simple projection: an undirected edge exists if either direction is present. Remove self-loops and collapse repeated ties.

## Splitting and leakage
For each fixed seed, shuffle observed positive edges and partition them into train, validation and test sets. The training graph contains only training positives. Negative candidates must be absent from the **complete observed graph**, so a withheld positive edge is never sampled as a negative. Sample negatives separately for intra- and inter-department strata, at the configured ratio. Candidate sampling is without replacement.

Louvain is run on the training graph only. The community bonus parameter is selected on validation data and evaluated once on test data. Department-label-based community membership is an oracle analysis, since it relies on external labels.

## Metrics
Report ROC-AUC and average precision separately within each stratum. Report Precision@K on the sampled candidate population and include candidate counts and positive prevalence. Use repeated seeds and report mean, standard deviation and confidence intervals. Metrics are conditional on the candidate-sampling procedure.

## Interpretation
The benchmark measures static missing-edge reconstruction. It does not show that a predicted edge will form in the future. Department labels may not match interaction-based communities. The undirected projection removes directionality. These limitations belong in the paper.
