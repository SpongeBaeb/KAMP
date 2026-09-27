"""Phase 2-B. 추가 후보: 소프트보팅 앙상블 + 준지도학습(self-training) + 비용 기반 임계값.

실행: python src/phase2b_ensemble_ssl.py   (phase2_models.py 다음에 실행)
산출물: outputs/tables/model_comparison_RG3.csv 에 행 추가 (팀 공통 형식),
        outputs/tables/p2b_*.csv, outputs/figures/p2b_*.png

준지도학습은 결정 ②(비라벨 미사용)의 예외로 둔 실험 후보다. 비라벨 데이터는 순위 변환(quantile
mapping)으로만 옮길 수 있고, 그 한계는 p2_unlabeled_rank_evidence.csv에 정리되어 있다.
"""
from pathlib import Path
import random

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from preprocess import LABEL, PRODUCT, load_raw, merge_pairs
from train import ALGOS, COST_RATIOS, SEED, SelfTraining, SoftVote, fit_with_thresholds, make_model
from evaluate import RULES, TEAM_COLS, extended_table, score, team_table
from viz import ALGO_COLOR, C_PASS, INK2, plt

random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
N_SPLITS, N_REPEATS = 5, 10
N_POOL = 5000  # self-training에 쓰는 비라벨 풀 크기 (시드 고정 무작위 추출)
N_PERM = 30
RULES_B = RULES + tuple(f"Cost{r}" for r in COST_RATIOS)
VOTE = "SoftVote(LogReg+RF+HGB)"
SSL = "SelfTrain+SoftVote"
C_VOTE, C_SSL = "#eda100", "#4a3aa7"


def save(df, name):
    df.to_csv(TAB / name, index=False, encoding="utf-8-sig")


def unlabeled_pool(unl, feats):
    """전체 비라벨 배치 안에서의 변수별 분위를 구한 뒤 풀을 추출한다."""
    u = np.column_stack([(unl[c].rank(method="average").to_numpy() - 0.5) / len(unl) for c in feats])
    return u[np.random.default_rng(SEED).choice(len(unl), N_POOL, replace=False)]


def make_candidate(name, pool):
    vote = SoftVote([make_model(a, "class_weight") for a in ALGOS])
    return vote if name == VOTE else SelfTraining(vote, pool)


def run_fold_b(name, pool, X, y, tr, te, fold):
    model, thr = fit_with_thresholds(make_candidate(name, pool), X[tr], y[tr], cost=True)
    p = model.predict_proba(X[te])[:, 1]
    rec = {"algo": name, "imbalance": "class_weight", "fold": fold, **score(y[te], p, thr)}
    yt, n = y[te], len(te)
    for r in COST_RATIOS:  # 제품 100개당 총비용 (FP 1, FN r) vs 단순 정책
        pred = p >= thr[f"Cost{r}"]
        rec[f"cost100_Cost{r}"] = 100 * (r * ((~pred) & (yt == 1)).sum() + (pred & (yt == 0)).sum()) / n
        rec[f"cost100_inspect_all_r{r}"] = 100 * (yt == 0).sum() / n
        rec[f"cost100_inspect_none_r{r}"] = 100 * r * (yt == 1).sum() / n
    return rec


def _perm_vote(X, y_perm):
    auc, ap = [], []
    for tr, te in RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                          random_state=SEED).split(X, y_perm):
        p = make_candidate(VOTE, None).fit(X[tr], y_perm[tr]).predict_proba(X[te])[:, 1]
        auc.append(roc_auc_score(y_perm[te], p))
        ap.append(average_precision_score(y_perm[te], p))
    return np.mean(auc), np.mean(ap)


def main():
    raw, unl, feats = load_raw()
    data = merge_pairs(raw, feats)
    X, y = data[feats].to_numpy(), data[LABEL].to_numpy()
    pool = unlabeled_pool(unl, feats)
    splits = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                          random_state=SEED).split(X, y))

    recs = Parallel(n_jobs=-1)(delayed(run_fold_b)(name, pool, X, y, tr, te, f)
                               for name in (VOTE, SSL) for f, (tr, te) in enumerate(splits))
    folds = pd.DataFrame(recs)
    save(folds, "p2b_cv_fold_metrics.csv")
    ext = extended_table(folds)
    save(ext, "p2b_model_comparison_extended.csv")

    # 팀 공통 결과표에 행 추가 (재실행해도 중복되지 않도록 같은 모델 행은 교체)
    new = team_table(folds, PRODUCT, RULES_B)
    path = TAB / f"model_comparison_{PRODUCT}.csv"
    old = pd.read_csv(path)
    old = old[~old.model.str.startswith((VOTE, SSL))]
    merged = pd.concat([old, new], ignore_index=True)[TEAM_COLS]
    merged.to_csv(path, index=False, encoding="utf-8-sig")

    # 소프트보팅 앙상블의 라벨 순열 귀무분포
    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(y) for _ in range(N_PERM)]
    null = Parallel(n_jobs=-1)(delayed(_perm_vote)(X, yp) for yp in perms)
    null = pd.DataFrame(null, columns=["roc_auc", "prauc"])
    e = ext.set_index("algo")
    nt = pd.DataFrame([{
        "algo": a, "observed_roc_auc": e.loc[a, "roc_auc_mean"], "observed_prauc": e.loc[a, "prauc_mean"],
        "null_model": VOTE, "null_roc_auc_mean": null.roc_auc.mean(), "null_roc_auc_std": null.roc_auc.std(),
        "p_roc_auc_greater": (1 + (null.roc_auc >= e.loc[a, "roc_auc_mean"]).sum()) / (N_PERM + 1),
        "null_prauc_mean": null.prauc.mean(), "null_prauc_std": null.prauc.std(),
        "p_prauc_greater": (1 + (null.prauc >= e.loc[a, "prauc_mean"]).sum()) / (N_PERM + 1), "n_perm": N_PERM}
        for a in (VOTE, SSL)])
    save(nt, "p2b_null_test.csv")

    plot(ext, nt, y.mean())

    pd.set_option("display.width", 250)
    print("[추가 행]\n", new.to_string(index=False))
    cols = ["algo", "prauc_mean", "roc_auc_mean", "recall_at_5_mean", "recall_at_10_mean", "recall_at_20_mean"]
    print(ext[cols].to_string(index=False))
    fr = [c for c in ext.columns if c.startswith(("flag_rate_", "cost100_")) and c.endswith("_mean")]
    print(ext[["algo", *fr]].T.to_string())
    print("\n[순열 검정]\n", nt.to_string(index=False))
    print(f"\n결과표 총 {len(merged)}행 → {path}")


def plot(ext, nt, prev):
    p2 = pd.read_csv(TAB / "p2_model_comparison_extended.csv")
    p2 = p2[p2.imbalance == "class_weight"]
    rows = [(r.algo, r, ALGO_COLOR[r.algo]) for r in p2.itertuples()] + \
           [(a, r, c) for a, c in ((VOTE, C_VOTE), (SSL, C_SSL)) for r in ext[ext.algo == a].itertuples()]
    null_pr = nt.null_prauc_mean.iloc[0]

    # (1) 순위 성능: Phase 2 단일 모델 vs 2-B 후보
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, m, refs, ttl in ((axes[0], "prauc", ((prev, "불량률"), (null_pr, "순열 기준선")), "PR-AUC"),
                             (axes[1], "recall_at_20", ((0.2, "무작위 검사"),), "Recall@상위 20%")):
        for k, (name, r, c) in enumerate(rows):
            ax.errorbar(getattr(r, f"{m}_mean"), k, xerr=getattr(r, f"{m}_std"), fmt="o", color=c, ms=7, lw=1.5,
                        mec="white", mew=1)
        for v, lab in refs:
            ax.axvline(v, color=INK2 if lab != "순열 기준선" else C_PASS, lw=1, ls="--")
            ax.text(v, -0.75, f" {lab}", fontsize=7.5, color=INK2, va="center")
        ax.set_title(ttl, loc="left"); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(rows)), [f"{n} | class_weight" for n, _, _ in rows]); axes[0].set_ylim(len(rows) - 0.5, -1.0)
    fig.suptitle("Phase 2 단일 모델 vs Phase 2-B 후보 (병합 591행, 5-fold×10, 평균 ± 표준편차)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "p2b_ranking_vs_phase2.png"); plt.close(fig)

    # (2) 임계값 규칙별 Recall / Precision / 경보 비율
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, m, ttl in zip(axes, ("recall", "precision", "flag_rate"), ("Recall", "Precision", "경보 비율 (검사 대상 비율)")):
        for j, (a, c) in enumerate(((VOTE, C_VOTE), (SSL, C_SSL))):
            r = ext[ext.algo == a].iloc[0]
            v = [r[f"{m}_{rule}_mean"] for rule in RULES_B]
            s = [r[f"{m}_{rule}_std"] for rule in RULES_B]
            ax.errorbar(v, np.arange(len(RULES_B)) + (j - 0.5) * 0.3, xerr=s, fmt="o", color=c, ms=6, lw=1.3,
                        mec="white", label=a)
        if m == "precision":
            ax.axvline(prev, color=INK2, ls="--", lw=1)
        ax.set_xlim(-0.03, 1.03); ax.set_title(ttl, loc="left"); ax.grid(axis="y", visible=False)
    labels = {"F1max": "F1 최대", "Recall80": "Recall ≥ 0.8", "Cost5": "비용 FN:FP=5:1",
              "Cost10": "비용 FN:FP=10:1", "Cost20": "비용 FN:FP=20:1"}
    axes[0].set_yticks(range(len(RULES_B)), [labels[r] for r in RULES_B]); axes[0].invert_yaxis()
    axes[2].legend(loc="center right")
    fig.suptitle("Phase 2-B 후보의 임계값 규칙별 결과 (Precision 점선 = 불량률)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "p2b_threshold_rules.png"); plt.close(fig)


if __name__ == "__main__":
    main()
