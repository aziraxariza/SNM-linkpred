import pandas as pd
from src.temporal import first_appearance,chronological_split,negative_pairs

def test_first_appearance_collapses_repeated_and_reverse_edges():
    f=pd.DataFrame([(1,2,10),(1,2,20),(2,1,30),(2,3,15)],columns=["src","dst","timestamp"])
    d=first_appearance(f)
    assert len(d)==2 and list(d.timestamp)==[10,15]

def test_chronological_split_is_ordered():
    d=pd.DataFrame({"u":range(20),"v":range(1,21),"timestamp":range(20)})
    tr,va,te,_=chronological_split(d,.6,.2)
    assert tr.timestamp.max()<va.timestamp.min()<te.timestamp.min()

def test_negative_pairs_are_not_observed():
    known={(0,1),(1,2)}
    neg=negative_pairs([0,1,2,3,4],known,3,42)
    assert len(neg)==3 and not(set(neg)&known)
