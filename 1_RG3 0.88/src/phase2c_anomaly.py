"""Phase 2 추가 후보: 이상탐지 계열 (Isolation Forest, 마할라노비스 거리, One-Class SVM, LOF).

실행: python src/phase2c_anomaly.py   (phase2_models.py, phase2b_ensemble_ssl.py 다음에 실행)
산출물: outputs/tables/model_comparison_RG3.csv 에 행 추가 (팀 공통 형식),
        outputs/tables/p2c_*.csv, outputs/figures/p2c_*.png

- 쌍 병합 데이터, 팀 공통 5-fold×10 분할을 그대로 쓴다.
- 각 학습 fold의 정상 샷만으로 모델을 적합한다 (불량 라벨은 적합에 쓰지 않음).
- 평가는 평가 fold = 학습에 쓰지 않은 정상 샷 + 불량 샷. fold별 기준선(불량 비율)을 함께 기록한다.
- 임계값은 학습 fold 내부 3-fold로 정한다 (내부 학습 정상 샷으로 적합 → 내부 검증 정상+불량으로 선택).
"""
from pathlib import Path
import random

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from preprocess import IDX, LABEL, PRODUCT, load_raw, merge_pairs
from train import ANOMALY, SEED, NoveltyDetector, fit_with_thresholds
from evaluate import TEAM_COLS, extended_table, score, team_table
from viz import ALGO_COLOR, C_FAIL, C_L, C_PASS, INK2, plt

random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
N_SPLITS, N_REPEATS = 5, 10
N_PERM = 30
IMB = "normal_only"
DET_COLOR = {"IsolationForest": "#eda100", "Mahalanobis": "#e87ba4", "OneClassSVM": "#008300", "LOF": "#4a3aa7"}


def save(df, name):
    df.to_csv(TAB / name, index=False, encoding="utf-8-sig")


# ================================================================ 이상탐지 CV
def run_fold_c(method, X, y, tr, te, fold):
    model, thr = fit_with_thresholds(NoveltyDetector(method), X[tr], y[tr])
    fit_rows = tr[y[tr] == 0]
    assert not set(fit_rows) & set(te), "학습 정상 샷이 평가 세트에 섞임"
    assert model.n_train_normals_ == len(fit_rows), "정상 샷 외 데이터가 학습에 쓰임"
    p = model.predict_proba(X[te])[:, 1]
    return {"algo": method, "imbalance": IMB, "fold": fold, "n_train_normals": len(fit_rows),
            "n_test_normals": int((y[te] == 0).sum()), **score(y[te], p, thr)}


def _perm_detector(method, X, y_perm):
    auc, ap = [], []
    for tr, te in RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                          random_state=SEED).split(X, y_perm):
        p = NoveltyDetector(method).fit(X[tr], y_perm[tr]).predict_proba(X[te])[:, 1]
        auc.append(roc_auc_score(y_perm[te], p))
        ap.append(average_precision_score(y_perm[te], p))
    return np.mean(auc), np.mean(ap)


# ================================================================ 최근접 정상 거리
def nearest_normal(data, feats):
    """각 샷에서 자기 자신을 뺀 가장 가까운 정상 샷까지의 거리 (정상 샷 기준 표준화 공간)."""
    live = [c for c in feats if data[c].nunique() > 1]
    y = data[LABEL].to_numpy()
    Z = StandardScaler().fit(data.loc[y == 0, live]).transform(data[live])
    normal_idx = np.flatnonzero(y == 0)
    nn = NearestNeighbors(n_neighbors=6).fit(Z[normal_idx])
    dist, ind = nn.kneighbors(Z)
    rows = []
    for i in range(len(Z)):
        d, j = dist[i], normal_idx[ind[i]]
        keep = j != i  # 정상 샷은 자기 자신 제외
        d, j = d[keep][:5], j[keep][:5]
        rows.append({IDX: data[IDX].iloc[i], LABEL: y[i], "nn1_dist": d[0], "nn5_mean_dist": d.mean(),
                     "nn1_idx": data[IDX].iloc[j[0]],
                     "nn1_idx_gap": abs(int(data[IDX].iloc[j[0]]) - int(data[IDX].iloc[i]))})
    per = pd.DataFrame(rows)
    save(per, "p2c_nearest_normal_distance.csv")

    norm, fail = per[per[LABEL] == 0], per[per[LABEL] == 1]
    summ = []
    for m in ("nn1_dist", "nn5_mean_dist", "nn1_idx_gap"):
        mw = stats.mannwhitneyu(fail[m], norm[m], alternative="two-sided")
        summ.append({"metric": m, "normal_median": norm[m].median(), "defect_median": fail[m].median(),
                     "normal_p90": norm[m].quantile(.9), "defect_p90": fail[m].quantile(.9),
                     "auc_defect_gt_normal": mw.statistic / (len(fail) * len(norm)), "mannwhitney_p": mw.pvalue,
                     "ks_p": stats.ks_2samp(fail[m], norm[m]).pvalue})
    summ = pd.DataFrame(summ)
    summ["defect_share_above_normal_p95_nn1"] = (fail.nn1_dist > norm.nn1_dist.quantile(.95)).mean()
    summ["share_nn1_within_5shots_normal"] = (norm.nn1_idx_gap <= 5).mean()
    summ["share_nn1_within_5shots_defect"] = (fail.nn1_idx_gap <= 5).mean()
    save(summ, "p2c_nearest_normal_summary.csv")

    s = summ.set_index("metric")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    ax = axes[0]
    for d, c, lab in ((norm.nn1_dist, C_L, f"정상 샷 (n={len(norm)})"), (fail.nn1_dist, C_FAIL, f"불량 샷 (n={len(fail)})")):
        v = np.sort(d.to_numpy())
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, lw=2, label=lab)
    ax.set_xlabel("최근접 정상 샷까지 거리 (표준화 유클리드)"); ax.set_ylabel("누적 비율")
    ax.set_title(f"누적분포 — Mann-Whitney p={s.loc['nn1_dist', 'mannwhitney_p']:.2f}, "
                 f"AUC={s.loc['nn1_dist', 'auc_defect_gt_normal']:.2f}", loc="left")
    ax.legend(loc="lower right")

    ax = axes[1]
    rng = np.random.default_rng(SEED)
    for k, (d, c) in enumerate(((norm.nn1_dist, C_L), (fail.nn1_dist, C_FAIL))):
        ax.boxplot(d, positions=[k], widths=0.5, showfliers=False, medianprops={"color": INK2},
                   boxprops={"color": C_PASS}, whiskerprops={"color": C_PASS}, capprops={"color": C_PASS})
        ax.scatter(k + rng.uniform(-0.18, 0.18, len(d)), d, s=10 if k == 0 else 26, color=c,
                   alpha=0.35 if k == 0 else 0.9, edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(norm.nn1_dist.quantile(.95), color=INK2, lw=0.8, ls="--")
    ax.text(1.45, norm.nn1_dist.quantile(.95), "정상 95%", fontsize=7.5, color=INK2, va="bottom", ha="right")
    ax.set_xticks([0, 1], ["정상", "불량"]); ax.grid(axis="x", visible=False)
    ax.set_ylabel("최근접 정상 샷까지 거리")
    ax.set_title(f"불량 중 정상 95% 초과: {s.defect_share_above_normal_p95_nn1.iloc[0]:.0%} (무작위면 5%)", loc="left")

    ax = axes[2]
    for d, c, lab in ((norm.nn1_idx_gap, C_L, "정상"), (fail.nn1_idx_gap, C_FAIL, "불량")):
        v = np.sort(d.clip(lower=1).to_numpy())
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=c, lw=2, label=lab)
    ax.set_xscale("log"); ax.set_ylabel("누적 비율")
    ax.set_xlabel("최근접 정상 샷과의 인덱스 간격 (log)")
    ax.set_title(f"이웃이 5샷 이내인 비율 — 정상 {s.share_nn1_within_5shots_normal.iloc[0]:.0%}, "
                 f"불량 {s.share_nn1_within_5shots_defect.iloc[0]:.0%}", loc="left")
    ax.legend(loc="lower right")
    fig.suptitle("불량 샷 vs 정상 샷의 최근접 정상 샷 거리 (병합 591행, 정상 샷 기준 표준화, 자기 자신 제외)",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "p2c_nearest_normal_distance.png"); plt.close(fig)
    return summ


# ================================================================ 그래프: Phase 2 후보와 비교
def plot_compare(ext, nt, prev_mean):
    p2 = pd.read_csv(TAB / "p2_model_comparison_extended.csv")
    p2 = p2[p2.imbalance == "class_weight"]
    n2 = pd.read_csv(TAB / "p2_null_test.csv").set_index("algo")
    ntc = nt.set_index("algo")
    rows = [(f"{r.algo} | class_weight", r, ALGO_COLOR[r.algo], n2.loc[r.algo]) for r in p2.itertuples()] + \
           [(f"{a} | 정상만 학습", ext[ext.algo == a].iloc[0], DET_COLOR[a], ntc.loc[a]) for a in ANOMALY]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    panels = ((axes[0], "prauc", "null_prauc_mean", ((prev_mean, "평가 세트 불량 비율"),), "PR-AUC"),
              (axes[1], "roc_auc", "null_roc_auc_mean", ((0.5, "0.5"),), "ROC-AUC (참고)"),
              (axes[2], "recall_at_20", None, ((0.2, "무작위 검사"),), "Recall@상위 20%"))
    for ax, m, nullcol, refs, ttl in panels:
        for k, (lab, r, c, nr) in enumerate(rows):
            mean, sd = (getattr(r, f"{m}_mean"), getattr(r, f"{m}_std")) if not isinstance(r, pd.Series) \
                else (r[f"{m}_mean"], r[f"{m}_std"])
            ax.errorbar(mean, k, xerr=sd, fmt="o", color=c, ms=7, lw=1.5, mec="white", mew=1)
            if nullcol:
                ax.plot(nr[nullcol], k, marker="|", color=INK2, ms=14, mew=2, zorder=4,
                        label="라벨 순열 기준선" if k == 0 else None)
        for v, t in refs:
            ax.axvline(v, color=INK2, lw=1, ls="--")
            ax.text(v, -0.8, f" {t}", fontsize=7.5, color=INK2, va="center")
        ax.set_title(ttl, loc="left"); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(rows)), [r[0] for r in rows]); axes[0].set_ylim(len(rows) - 0.5, -1.2)
    axes[0].legend(loc="lower right")
    fig.suptitle("지도학습 vs 이상탐지 (병합 591행, 5-fold×10, 평균 ± 표준편차, 세로 막대 = 모델별 순열 기준선)",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "p2c_ranking_vs_phase2.png"); plt.close(fig)


def main():
    raw, _, feats = load_raw()
    data = merge_pairs(raw, feats)
    X, y = data[feats].to_numpy(), data[LABEL].to_numpy()
    splits = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                          random_state=SEED).split(X, y))

    recs = Parallel(n_jobs=-1)(delayed(run_fold_c)(m, X, y, tr, te, f)
                               for m in ANOMALY for f, (tr, te) in enumerate(splits))
    folds = pd.DataFrame(recs)
    save(folds, "p2c_cv_fold_metrics.csv")
    ext = extended_table(folds)
    save(ext, "p2c_model_comparison_extended.csv")

    # 평가 세트 기준선 (분할이 모든 모델에 공통이므로 한 모델 기준으로 요약)
    f0 = folds[folds.algo == ANOMALY[0]]
    base = pd.DataFrame([{
        "n_folds": len(f0), "n_train_normals_mean": f0.n_train_normals.mean(),
        "n_test_mean": f0.n_test.mean(), "n_test_normals_mean": f0.n_test_normals.mean(),
        "n_test_defects_mean": f0.n_pos_test.mean(), "n_test_defects_min": f0.n_pos_test.min(),
        "n_test_defects_max": f0.n_pos_test.max(),
        "baseline_defect_rate_mean": f0.prevalence.mean(), "baseline_defect_rate_std": f0.prevalence.std(),
        "baseline_defect_rate_min": f0.prevalence.min(), "baseline_defect_rate_max": f0.prevalence.max(),
        "note": "평가 세트 = 학습에 쓰지 않은 정상 샷 + 불량 샷. PR-AUC 무작위 기준선 = 이 불량 비율",
    }])
    save(base, "p2c_eval_baseline.csv")

    new = team_table(folds, PRODUCT)
    path = TAB / f"model_comparison_{PRODUCT}.csv"
    old = pd.read_csv(path)
    old = old[~old.model.str.startswith(tuple(f"{a}|" for a in ANOMALY))]
    merged = pd.concat([old, new], ignore_index=True)[TEAM_COLS]
    merged.to_csv(path, index=False, encoding="utf-8-sig")

    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(y) for _ in range(N_PERM)]
    res = Parallel(n_jobs=-1)(delayed(_perm_detector)(m, X, perms[i]) for m in ANOMALY for i in range(N_PERM))
    null = pd.DataFrame([{"algo": m, "perm": i, "roc_auc": r[0], "prauc": r[1]}
                         for (m, i), r in zip([(m, i) for m in ANOMALY for i in range(N_PERM)], res)])
    save(null, "p2c_null_distribution.csv")
    e = ext.set_index("algo")
    nt = pd.DataFrame([{
        "algo": m, "observed_roc_auc": e.loc[m, "roc_auc_mean"], "observed_prauc": e.loc[m, "prauc_mean"],
        "null_roc_auc_mean": g.roc_auc.mean(), "null_roc_auc_std": g.roc_auc.std(),
        "p_roc_auc_greater": (1 + (g.roc_auc >= e.loc[m, "roc_auc_mean"]).sum()) / (N_PERM + 1),
        "null_prauc_mean": g.prauc.mean(), "null_prauc_std": g.prauc.std(),
        "p_prauc_greater": (1 + (g.prauc >= e.loc[m, "prauc_mean"]).sum()) / (N_PERM + 1), "n_perm": N_PERM}
        for m, g in null.groupby("algo", sort=False)])
    save(nt, "p2c_null_test.csv")

    nn = nearest_normal(data, feats)
    plot_compare(ext, nt, base.baseline_defect_rate_mean.iloc[0])

    pd.set_option("display.width", 250)
    print("[평가 세트 기준선]\n", base.T.to_string())
    print("\n[추가 행]\n", new.to_string(index=False))
    cols = ["algo", "prauc_mean", "prauc_std", "roc_auc_mean", "recall_at_5_mean", "recall_at_10_mean",
            "recall_at_20_mean", "flag_rate_F1max_mean", "flag_rate_Recall80_mean"]
    print(ext[cols].to_string(index=False))
    print("\n[순열 검정]\n", nt.to_string(index=False))
    print("\n[최근접 정상 거리]\n", nn.to_string(index=False))
    print(f"\n결과표 총 {len(merged)}행 → {path}")


if __name__ == "__main__":
    main()
