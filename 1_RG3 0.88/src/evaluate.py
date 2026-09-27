"""평가 지표와 fold 단위 평가."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score)

from train import SEED, fit_with_thresholds, make_model

K_LIST = (0.05, 0.10, 0.20)
RULES = ("F1max", "Recall80")
TEAM_COLS = ["model", "product", "f1_mean", "f1_std", "recall_mean", "recall_std",
             "precision_mean", "precision_std", "prauc_mean", "prauc_std", "threshold"]


def recall_at_k(y, p, k, seed=SEED):
    """확률 상위 k% 를 검사했을 때 잡히는 불량 비율. 동점은 시드 고정 무작위로 푼다."""
    n = int(np.ceil(k * len(y)))
    order = np.random.default_rng(seed).permutation(len(y))
    top = order[np.argsort(-p[order], kind="stable")][:n]
    return y[top].sum() / max(y.sum(), 1)


def score(y, p, thr=None):
    out = {"prauc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
           "prevalence": y.mean(), "n_test": len(y), "n_pos_test": int(y.sum())}
    for k in K_LIST:
        out[f"recall_at_{int(k*100)}"] = recall_at_k(y, p, k)
    for rule, t in (thr or {}).items():
        pred = (p >= t).astype(int)
        out[f"thr_{rule}"] = t
        out[f"f1_{rule}"] = f1_score(y, pred, zero_division=0)
        out[f"recall_{rule}"] = recall_score(y, pred, zero_division=0)
        out[f"precision_{rule}"] = precision_score(y, pred, zero_division=0)
        out[f"flag_rate_{rule}"] = pred.mean()
    return out


def run_fold(algo, imb, X, y, tr, te, fold, perm_feats=None):
    """한 outer fold: 임계값 탐색·적합은 train에서만, 평가는 test에서만."""
    model, thr = fit_with_thresholds(make_model(algo, imb), X[tr], y[tr])
    p = model.predict_proba(X[te])[:, 1]
    rec = {"algo": algo, "imbalance": imb, "fold": fold, **score(y[te], p, thr)}
    imp = None
    if perm_feats is not None:
        r = permutation_importance(model, X[te], y[te], scoring="average_precision",
                                   n_repeats=5, random_state=SEED)
        imp = pd.DataFrame({"algo": algo, "imbalance": imb, "fold": fold,
                            "feature": perm_feats, "importance": r.importances_mean})
    return rec, (te, p), imp


def config_name(algo, imb, rule=None):
    return f"{algo}|{imb}" + (f"|thr={rule}" if rule else "")


def team_table(folds, product, rules=RULES):
    """팀 공통 형식: 설정×임계값 규칙마다 한 행."""
    rows = []
    for (algo, imb), g in folds.groupby(["algo", "imbalance"], sort=False):
        for rule in rules:
            rows.append({
                "model": config_name(algo, imb, rule), "product": product,
                "f1_mean": g[f"f1_{rule}"].mean(), "f1_std": g[f"f1_{rule}"].std(),
                "recall_mean": g[f"recall_{rule}"].mean(), "recall_std": g[f"recall_{rule}"].std(),
                "precision_mean": g[f"precision_{rule}"].mean(), "precision_std": g[f"precision_{rule}"].std(),
                "prauc_mean": g.prauc.mean(), "prauc_std": g.prauc.std(),
                "threshold": g[f"thr_{rule}"].mean(),
            })
    return pd.DataFrame(rows)[TEAM_COLS]


def extended_table(folds):
    num = [c for c in folds.columns if c not in ("algo", "imbalance", "fold")]
    g = folds.groupby(["algo", "imbalance"], sort=False)[num]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std"))
    return out[sorted(out.columns)].reset_index()
