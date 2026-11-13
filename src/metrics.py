import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate(y_true, scores, top_k=(10, 25, 50, 100)):
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    result = {
        "roc_auc": float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan"),
        "average_precision": float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else float("nan")
    }
    order = np.argsort(-s)
    for k in top_k:
        kk = min(int(k), len(y))
        result[f"precision_at_{k}"] = float(y[order[:kk]].mean()) if kk else float("nan")
    return result
