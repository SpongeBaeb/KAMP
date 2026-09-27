"""Phase 2 결과 요약: 계열별 정리표 (순위 없음) + 무작위 기준선 행.

실행: python src/phase2_summary.py   (phase2_models → phase2b → phase2c → phase2e 다음에 실행)
산출물: outputs/tables/p2_summary_by_family.csv, outputs/tables/p2_null_test_extra.csv
"""
from pathlib import Path
import random

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from preprocess import LABEL, load_raw, merge_pairs
from train import SEED, make_model

random.seed(SEED)
np.random.seed(SEED)

TAB = Path(__file__).resolve().parents[1] / "outputs" / "tables"
N_PERM = 30
EXTRA = [("HGB", "ROS"), ("RF", "SMOTE")]  # Phase 2 null_test에 없던 설정


def _perm(algo, imb, X, y_perm):
    auc, ap = [], []
    for tr, te in RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED).split(X, y_perm):
        p = make_model(algo, imb).fit(X[tr], y_perm[tr]).predict_proba(X[te])[:, 1]
        auc.append(roc_auc_score(y_perm[te], p))
        ap.append(average_precision_score(y_perm[te], p))
    return np.mean(auc), np.mean(ap)


def extra_null(ext2):
    raw, _, feats = load_raw()
    d = merge_pairs(raw, feats)
    X, y = d[feats].to_numpy(), d[LABEL].to_numpy()
    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(y) for _ in range(N_PERM)]
    jobs = [(a, i, k) for a, i in EXTRA for k in range(N_PERM)]
    res = Parallel(n_jobs=-1)(delayed(_perm)(a, i, X, perms[k]) for a, i, k in jobs)
    null = pd.DataFrame([{"algo": a, "imbalance": i, "roc_auc": r[0], "prauc": r[1]} for (a, i, _), r in zip(jobs, res)])
    e = ext2.set_index(["algo", "imbalance"])
    rows = []
    for (a, i), g in null.groupby(["algo", "imbalance"], sort=False):
        oa, op = e.loc[(a, i), "roc_auc_mean"], e.loc[(a, i), "prauc_mean"]
        rows.append({"algo": a, "imbalance": i, "observed_roc_auc": oa, "null_roc_auc_mean": g.roc_auc.mean(),
                     "null_roc_auc_std": g.roc_auc.std(), "p_roc_auc_greater": (1 + (g.roc_auc >= oa).sum()) / (N_PERM + 1),
                     "observed_prauc": op, "null_prauc_mean": g.prauc.mean(), "null_prauc_std": g.prauc.std(),
                     "p_prauc_greater": (1 + (g.prauc >= op).sum()) / (N_PERM + 1), "n_perm": N_PERM})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "p2_null_test_extra.csv", index=False, encoding="utf-8-sig")
    return out


def row(family, model, e, p):
    return {"family": family, "model": model,
            "prauc_mean": e["prauc_mean"], "prauc_std": e["prauc_std"],
            "f1_F1max": e["f1_F1max_mean"], "precision_F1max": e["precision_F1max_mean"],
            "recall_F1max": e["recall_F1max_mean"], "recall_at_20": e["recall_at_20_mean"],
            "roc_auc": e["roc_auc_mean"], "perm_p_prauc": p}


def main():
    ext2 = pd.read_csv(TAB / "p2_model_comparison_extended.csv").set_index(["algo", "imbalance"])
    ext2b = pd.read_csv(TAB / "p2b_model_comparison_extended.csv").set_index("algo")
    ext2c = pd.read_csv(TAB / "p2c_model_comparison_extended.csv").set_index("algo")
    n2 = pd.read_csv(TAB / "p2_null_test.csv").set_index(["algo", "imbalance"])
    n2x = extra_null(ext2.reset_index()).set_index(["algo", "imbalance"])
    n2b = pd.read_csv(TAB / "p2b_null_test.csv").set_index("algo")
    n2c = pd.read_csv(TAB / "p2c_null_test.csv").set_index("algo")
    null = pd.concat([n2, n2x])

    rows = [{"family": "기준선", "model": "무작위 (불량률 / 무작위 검사)", "prauc_mean": 0.042, "prauc_std": np.nan,
             "f1_F1max": np.nan, "precision_F1max": np.nan, "recall_F1max": np.nan, "recall_at_20": 0.20,
             "roc_auc": 0.5, "perm_p_prauc": np.nan}]
    fam = "지도학습 · class_weight"
    for a in ("LogReg", "RF", "HGB"):
        rows.append(row(fam, f"{a} | class_weight", ext2.loc[(a, "class_weight")], null.loc[(a, "class_weight"), "p_prauc_greater"]))
    fam = "지도학습 · 오버샘플링"
    for a, i in EXTRA:
        rows.append(row(fam, f"{a} | {i}", ext2.loc[(a, i)], null.loc[(a, i), "p_prauc_greater"]))
    fam = "앙상블 · 준지도학습"
    for a, lab in (("SoftVote(LogReg+RF+HGB)", "소프트보팅 (LogReg+RF+HGB)"), ("SelfTrain+SoftVote", "자기학습 + 소프트보팅")):
        rows.append(row(fam, lab, ext2b.loc[a], n2b.loc[a, "p_prauc_greater"]))
    fam = "이상탐지 · 정상만 학습"
    for a, lab in (("IsolationForest", "Isolation Forest"), ("Mahalanobis", "마할라노비스 거리"),
                   ("OneClassSVM", "One-Class SVM"), ("LOF", "LOF")):
        rows.append(row(fam, lab, ext2c.loc[a], n2c.loc[a, "p_prauc_greater"]))
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "p2_summary_by_family.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    print(n2x.round(3).to_string())
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
