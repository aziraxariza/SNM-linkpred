"""Chronological first-appearance link prediction for Email-Eu-core-temporal."""
from pathlib import Path
import gzip, random
from urllib.request import urlretrieve
import networkx as nx
import pandas as pd

TEMPORAL_URL = "https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz"

def load_temporal(path="data/raw/email-Eu-core-temporal.txt.gz", url=TEMPORAL_URL):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists(): urlretrieve(url,path)
    opener=gzip.open if str(path).endswith(".gz") else open
    rows=[]
    with opener(path,"rt",encoding="utf-8") as stream:
        for line in stream:
            p=line.split()
            if len(p)>=3 and not p[0].startswith("#"):
                u,v,t=map(int,p[:3])
                if u!=v: rows.append((u,v,t))
    frame=pd.DataFrame(rows,columns=["src","dst","timestamp"])
    if frame.empty: raise ValueError("No valid temporal edge rows")
    return frame

def first_appearance(frame):
    x=frame.copy()
    x["u"]=x[["src","dst"]].min(axis=1); x["v"]=x[["src","dst"]].max(axis=1)
    return x.groupby(["u","v"],as_index=False)["timestamp"].min().sort_values("timestamp").reset_index(drop=True)

def chronological_split(dyads,train_fraction=.70,validation_fraction=.15):
    if train_fraction<=0 or validation_fraction<=0 or train_fraction+validation_fraction>=1:
        raise ValueError("Invalid chronological split fractions")
    ordered=dyads.sort_values(["timestamp","u","v"]).reset_index(drop=True)
    n=len(ordered); i=max(1,int(n*train_fraction)); j=max(i+1,int(n*(train_fraction+validation_fraction)))
    if j>=n: raise ValueError("Not enough unique dyads")
    t1=int(ordered.iloc[i-1].timestamp); t2=int(ordered.iloc[j-1].timestamp)
    train=ordered[ordered.timestamp<=t1]; val=ordered[(ordered.timestamp>t1)&(ordered.timestamp<=t2)]
    test=ordered[ordered.timestamp>t2]
    return train,val,test,{"train_end":t1,"validation_end":t2}

def negative_pairs(nodes,known_dyads,count,seed):
    rng=random.Random(seed); nodes=list(nodes); known=set(known_dyads); neg=set()
    attempts=0; limit=max(10000,count*100)
    while len(neg)<count and attempts<limit:
        u,v=rng.sample(nodes,2); pair=(min(u,v),max(u,v)); attempts+=1
        if pair not in known: neg.add(pair)
    if len(neg)<count: raise RuntimeError(f"Only sampled {len(neg)} of {count} negatives")
    return sorted(neg)

def build_train_graph(train,nodes):
    g=nx.Graph(); g.add_nodes_from(nodes)
    g.add_edges_from(train[["u","v"]].itertuples(index=False,name=None))
    return g
