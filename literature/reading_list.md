# Literature review: 30 papers

The list combines older foundational papers with work directly about link prediction, community structure and evaluation.

| # | Paper | Relevance |
|---:|---|---|
| 1 | Watts & Strogatz (1998), *Collective dynamics of small-world networks*, Nature. https://doi.org/10.1038/30918 | Small-world structure |
| 2 | Barabási & Albert (1999), *Emergence of scaling in random networks*, Science. https://doi.org/10.1126/science.286.5439.509 | Preferential attachment |
| 3 | Newman (2001), *Clustering and preferential attachment in growing networks*, PRE. https://doi.org/10.1103/PhysRevE.64.025102 | Network growth |
| 4 | Girvan & Newman (2002), *Community structure in social and biological networks*, PNAS. https://doi.org/10.1073/pnas.122653799 | Foundational community detection |
| 5 | Adamic & Adar (2003), *Friends and neighbors on the Web*, Social Networks. https://doi.org/10.1016/S0378-8733(03)00009-1 | Adamic-Adar index |
| 6 | Newman (2004), *Finding community structure in very large networks*, PRE. https://doi.org/10.1103/PhysRevE.69.066133 | Modularity |
| 7 | Palla et al. (2005), *Uncovering the overlapping community structure of complex networks in nature and society*, Nature. https://doi.org/10.1038/nature03607 | Overlapping communities |
| 8 | Leicht, Holme & Newman (2006), *Vertex similarity in networks*, PRE. https://doi.org/10.1103/PhysRevE.73.026120 | Similarity indices |
| 9 | Newman (2006), *Modularity and community structure in networks*, PNAS. https://doi.org/10.1073/pnas.0601602103 | Modularity measure |
| 10 | Liben-Nowell & Kleinberg (2007), *The link-prediction problem for social networks*, JASIST. https://doi.org/10.1002/asi.20591 | Core link-prediction framework |
| 11 | Raghavan et al. (2007), *Near linear time algorithm to detect community structures in large-scale networks*, PRE. https://doi.org/10.1103/PhysRevE.76.036106 | Label propagation |
| 12 | Leskovec et al. (2007), *Graph evolution: Densification and shrinking diameters*, ACM TKDD. https://doi.org/10.1145/1217299.1217301 | Network evolution |
| 13 | Blondel et al. (2008), *Fast unfolding of communities in large networks*, JSTAT. https://doi.org/10.1088/1742-5468/2008/10/P10008 | Louvain |
| 14 | Lancichinetti et al. (2008), *Benchmark graphs for testing community detection algorithms*, PRE. https://doi.org/10.1103/PhysRevE.78.046110 | Community benchmarks |
| 15 | Zhou et al. (2009), *Predicting missing links via local information*, EPJ B. https://doi.org/10.1140/epjb/e2009-00335-8 | Resource allocation and local scores |
| 16 | Fortunato (2010), *Community detection in graphs*, Physics Reports. https://doi.org/10.1016/j.physrep.2009.11.002 | Community-detection review |
| 17 | Lü & Zhou (2011), *Link prediction in complex networks: A survey*, Physica A. https://doi.org/10.1016/j.physa.2010.11.027 | Link-prediction survey |
| 18 | Soundarajan & Hopcroft (2012), *Using community information to improve the precision of link prediction methods*, WWW Companion. https://doi.org/10.1145/2187980.2188020 | Directly relevant community-aware prediction |
| 19 | Yang & Leskovec (2012), *Defining and evaluating network communities based on ground-truth*, KDD Workshop. https://doi.org/10.1145/2350190.2350193 | Ground-truth evaluation |
| 20 | Perozzi et al. (2014), *DeepWalk: Online learning of social representations*, KDD. https://doi.org/10.1145/2623330.2623732 | Embedding methods |
| 21 | Grover & Leskovec (2016), *node2vec: Scalable feature learning for networks*, KDD. https://doi.org/10.1145/2939672.2939754 | Node embeddings |
| 22 | Kipf & Welling (2016), *Variational graph auto-encoders*. https://arxiv.org/abs/1611.07308 | Graph autoencoder |
| 23 | Hamilton et al. (2017), *Inductive representation learning on large graphs*, NeurIPS. | GraphSAGE |
| 24 | Yin et al. (2017), *Local higher-order graph clustering*, KDD. | Email-Eu-core source reference |
| 25 | Benson et al. (2018), *Simplicial closure and higher-order link prediction*, PNAS. https://doi.org/10.1073/pnas.1800683115 | Higher-order interactions |
| 26 | Zhang & Chen (2018), *Link prediction based on graph neural networks*, NeurIPS. | SEAL |
| 27 | Li et al. (2023), *Evaluating Graph Neural Networks for Link Prediction: Current Pitfalls and New Benchmarking*, NeurIPS D&B. https://arxiv.org/abs/2306.10453 | Negative-sampling and evaluation pitfalls |
| 28 | Kumar et al. (2024), *Community-enhanced link prediction in dynamic networks*, ACM TWeb. https://doi.org/10.1145/3580513 | Modern community-aware prediction |
| 29 | Nooribakhsh et al. (2024), *Community detection in social networks using machine learning: A systematic mapping study*, KAIS. https://doi.org/10.1007/s10115-024-02201-8 | Recent review |
| 30 | Cornell et al. (2025), *Are We Really Measuring Progress? ... Temporal Link Prediction*. https://arxiv.org/abs/2506.12588 | Temporal evaluation issues |

## How to position the paper

The most directly relevant antecedents are Liben-Nowell & Kleinberg (2007), Lü & Zhou (2011), Soundarajan & Hopcroft (2012), Zhou et al. (2009), Kumar et al. (2024), and Li et al. (2023).

Do not claim that community-aware link prediction is new. Earlier work already studies community information for link prediction. A defensible contribution is a transparent, reproducible comparison of intra- and inter-department prediction on an organizational communication network, with detected-community comparison and explicit split/negative-sampling sensitivity analysis.

This is a curated reading list, not a claim that every full text has been read. Check each final bibliographic field against the publisher/proceedings record before submission.
