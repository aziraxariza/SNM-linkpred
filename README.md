# Community Structure and Link Prediction

Reproducible static and temporal experiments on link prediction in organizational networks.

## Research questions
- Does link prediction differ for intra- and inter-department candidate pairs?
- How do Common Neighbors, Jaccard, Adamic-Adar, Preferential Attachment, Resource Allocation and LHN compare?
- How closely do Louvain communities match the supplied department labels?
- Does a validation-tuned detected-community bonus improve held-out ranking?
- Do patterns persist under chronological first-appearance prediction?

## Datasets
- **Static primary:** SNAP Email-Eu-core, directed communication edges plus 42 department labels: https://snap.stanford.edu/data/email-Eu-core.html
- **Temporal companion:** SNAP Email-Eu-core temporal, timestamped communication events: https://snap.stanford.edu/data/email-Eu-core-temporal.html
- Additional candidate datasets and their intended roles are catalogued in [datasets/README.md](datasets/README.md).

The static baseline projects the directed network to an undirected simple graph. The temporal experiment separately predicts first appearances of undirected dyads using chronological cutoffs. Do not conflate these tasks.

## Setup

Python 3.10+:
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Run the static repeated-split experiment:
```bash
python -m src.run_experiments --config configs/default.json
```

Run the chronological temporal experiment:
```bash
python -m src.run_temporal
```

The first run downloads the relevant SNAP files. Outputs are written to `results/`; figures are written to `paper/figures/`. Raw data and generated outputs are excluded from Git by default.\n\nTo run both experiments on GitHub, open **Actions → Full research experiments → Run workflow**. The manual workflow installs dependencies, runs tests, downloads both datasets, executes both experiments, and uploads the CSV tables and PNG figures as `snm-linkpred-experiment-results`. Static results include paired Wilcoxon comparisons and Benjamini--Hochberg adjusted p-values; `run_metadata.json` records software versions and compressed dataset checksums.

## Evaluation design
Static positives are split into train, validation and test sets. Negative candidates are absent from the complete observed static graph and sampled separately by department stratum. Louvain is fitted on the training graph only. The community bonus is tuned on validation data and evaluated on test data.

The temporal experiment collapses repeated and reverse-direction events to an undirected dyad and retains its first timestamp. It uses chronological train/validation/test cutoffs. Temporal negatives are pairs absent from the complete event file, a closed-world retrospective definition described in the paper.

## Manuscript and literature
- [Overleaf LaTeX source](paper/overleaf/)
- [30-paper literature review](literature/reading_list.md)
- [Experimental protocol](docs/experimental_protocol.md)
- [Figure catalogue](docs/figure_catalog.md)
- [Dataset catalogue](datasets/README.md)

## Reproducibility and limitations
Fixed seeds and configuration are stored in the repository. Do not write Results or claim a method wins until the scripts have run and the generated CSVs have been inspected. Static random edge holdout measures missing-edge reconstruction, not future formation. The temporal task is first-appearance prediction, but its closed-world negative sampling still uses the full observation window. The study uses one organization as its primary labelled network; external datasets are needed before claiming generality.
