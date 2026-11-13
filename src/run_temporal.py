"""Chronological first-appearance experiment on SNAP Email-Eu-core-temporal."""
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
from src.temporal import load_temporal,first_appearance,chronological_split,negative_pairs,build_train_graph
from src.communities import louvain_partition
from src.predict import score_pairs

def main():
    cfg=json.loads(Path("configs/default.json").read_text())
    events=load_temporal(cfg["temporal_path"]); dyads=first_appearance(events)
    train,val,test,cutoffs=chronological_split(dyads,cfg["temporal_train_fraction"],cfg["temporal_validation_fraction"])
    graph=build_train_graph(train,set(events.src)|set(events.dst))
    partition=louvain_partition(graph,cfg["louvain_resolution"],seed=17)
    # Closed-world negatives: dyads absent from the complete observed event file.
    observed=set(map(tuple,dyads[["u","v"]].itertuples(index=False,name=None)))
    rows=[]
    for split_name,pos_frame,seed in [("validation",val,1701),("test",test,1702)]:
        pos=list(map(tuple,pos_frame[["u","v"]].itertuples(index=False,name=None)))
        neg=negative_pairs(graph.nodes(),observed,len(pos),seed); pairs=pos+neg; y=[1]*len(pos)+[0]*len(neg)
        for method,scores in score_pairs(graph,pairs).items():
            rows.append({"split":split_name,"stratum":"all","method":method,
                         "roc_auc":roc_auc_score(y,scores),"average_precision":average_precision_score(y,scores),
                         "n_positive":len(pos),"n_negative":len(neg)})
            for same in (True,False):
                ids=[i for i,(u,v) in enumerate(pairs) if (partition.get(u)==partition.get(v))==same]
                if ids and len({y[i] for i in ids})==2:
                    yy=[y[i] for i in ids]; ss=[scores[i] for i in ids]
                    rows.append({"split":split_name,"stratum":"same_detected_community" if same else "different_detected_community",
                                 "method":method,"roc_auc":roc_auc_score(yy,ss),
                                 "average_precision":average_precision_score(yy,ss),
                                 "n_positive":sum(yy),"n_negative":len(yy)-sum(yy)})
    out=Path(cfg["results_dir"]); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"temporal_metrics.csv",index=False)
    summary={"dataset":"SNAP Email-Eu-core temporal","event_rows":len(events),
             "unique_first_appearance_dyads":len(dyads),"train_dyads":len(train),
             "validation_dyads":len(val),"test_dyads":len(test),"cutoffs_unix_seconds":cutoffs,
             "detected_communities":len(set(partition.values())),
             "task":"chronological first-appearance prediction on undirected dyads",
             "negative_sampling":"equal-count random dyads absent from all observed events (closed-world)"}
    (out/"temporal_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
