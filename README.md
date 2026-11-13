# Community Structure and Link Prediction

Reproducible experiments on how community membership relates to missing-edge reconstruction in the SNAP Email-Eu-core network.

**Status:** implementation scaffold. No empirical results are claimed until the pipeline is run and its outputs inspected.

## Research questions
- Does link prediction differ for intra-department and inter-department candidate pairs?
- How do Common Neighbors, Jaccard, Adamic-Adar and Preferential Attachment compare?
- Are community size and density associated with prediction difficulty?
- Does a community-membership feature improve ranking?
- How closely do Louvain communities match the supplied department labels?

## Dataset
SNAP Email-Eu-core: https://snap.stanford.edu/data/email-Eu-core.html

The source is a directed edge list with department labels. The primary heuristic experiment uses an undirected simple projection: an undirected edge exists if either direction was observed. Direction is therefore discarded for the baseline analysis. The static edge-holdout task is **missing-edge reconstruction**, not temporal future-link forecasting.

## Setup
Python 3.10+:
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m src.run_experiments --config configs/default.json
pytest -q
```

The pipeline downloads the SNAP files, runs repeated seeded splits, writes CSV/JSON outputs to `results/`, and creates publication figures in `paper/figures/`. Raw data and generated outputs are not committed.

## Method
Positive edges are split into train/validation/test sets. Negative candidates are pairs absent from the complete observed graph, avoiding false negatives from withheld positive edges. Negatives are sampled separately for intra- and inter-department strata. Louvain is run on the training graph only. The community bonus is tuned on validation data; the supplied-label variant is an oracle analysis because it uses external metadata.

## Limitations to report
- One organizational network does not establish generality.
- Undirected projection discards communication direction.
- Random edge holdout measures reconstruction, not future edge formation.
- Negative sampling defines the candidate population and affects metrics.
- Ground-truth department labels are not necessarily available in deployment.

See `docs/experimental_protocol.md` and `paper/overleaf/` for details and the manuscript scaffold.
