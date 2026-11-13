# Dataset catalogue

| Dataset | Research use | Available labels/time | Source |
|---|---|---|---|
| Email-Eu-core | Primary static network; department-stratified evaluation | 1,005 nodes, 25,571 directed edges, 42 department labels | https://snap.stanford.edu/data/email-Eu-core.html |
| Email-Eu-core temporal | Chronological future-link prediction | 986 nodes, 332,334 timestamped edges, 803-day span | https://snap.stanford.edu/data/email-Eu-core-temporal.html |
| EU email network | Larger external network | 265,214 nodes, 420,045 directed edges | https://snap.stanford.edu/data/email-EuAll.html |
| ca-GrQc | Co-authorship external validation | Static graph | https://snap.stanford.edu/data/ca-GrQc.html |
| ca-HepTh | Co-authorship external validation | Static graph | https://snap.stanford.edu/data/ca-HepTh.html |
| Wiki-Vote | Directed network / directionality sensitivity | Static directed edges | https://snap.stanford.edu/data/wiki-Vote.html |
| CollegeMsg | Temporal messaging network | Timestamped interactions | https://snap.stanford.edu/data/CollegeMsg.html |
| Bitcoin OTC | Signed trust network extension | Timestamped signed edges | https://snap.stanford.edu/data/soc-sign-bitcoin-otc.html |
| Bitcoin Alpha | Signed trust network extension | Timestamped signed edges | https://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html |
| Facebook ego networks | Social graph structural validation | Ego-network files and circles | https://snap.stanford.edu/data/ego-Facebook.html |
| OGB link prediction | Standardized benchmark comparison | Fixed benchmark splits/evaluators | https://ogb.stanford.edu/docs/linkprop/ |
| Temporal Graph Benchmark (TGB) | Temporal benchmark suite | Chronological splits and benchmark evaluators | https://tgb.complexdatalab.com/ |

## Recommended use

1. Email-Eu-core is the primary dataset because it provides department labels.
2. Email-Eu-core temporal is a separate chronological experiment. Never randomly split timestamped events for a future-link claim.
3. ca-GrQc and ca-HepTh can test whether findings hold in co-authorship networks. Detect communities on the training graph; do not imply that department labels exist.
4. Vary the negative-sampling ratio and include hard negatives in a robustness experiment.
5. Treat Bitcoin datasets separately because positive/negative trust signs have meaning. Do not collapse signed edges into ordinary links without a stated modelling choice.

## Data handling

Download data from the original providers. Record the source URL, retrieval date, version and SHA-256 checksum. Keep raw files out of Git unless the data license explicitly permits redistribution. Cite the provider's recommended papers.
