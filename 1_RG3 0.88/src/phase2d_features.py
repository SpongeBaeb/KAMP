"""Phase 2 추가 분석: 상관관계 + 변수 정리 + 파생변수, 변수 세트별 모델 재비교.

실행: python src/phase2d_features.py
산출물: outputs/tables/p2d_*.csv, outputs/figures/p2d_*.png

- 변수 정리·파생변수는 라벨을 쓰지 않는다 (불량 25건으로 라벨 기반 선택을 하면 과적합).
- fold 내부에서 상수·준상수·|ρ|≥0.95 중복 변수를 한 번 더 제거한다 (RedundancyFilter).
- 모델은 LogReg / RF / HGB (class_weight), 팀 공통 5-fold×10, 임계값은 학습 fold 내부에서 결정.
"""
from pathlib import Path
import random

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from preprocess import IDX, LABEL, PRODUCT, REGIME_START_IDX, build_feature_sets, load_raw, merge_pairs
from train import ALGOS, SEED, fit_with_thresholds, make_model_fs
from evaluate import RULES, TEAM_COLS, extended_table, score, team_table
from viz import ALGO_COLOR, C_FAIL, C_L, C_PASS, C_U, INK2, plt

random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
N_SPLITS, N_REPEATS = 5, 10
N_PERM = 30
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#eb6834"])


def save(df, name):
    df.to_csv(TAB / name, index=False, encoding="utf-8-sig")


def code(set_name):
    return set_name.split()[0]


# ================================================================ 상관관계
def correlation(d, sets, feats):
    live = [c for c in feats if d[c].nunique() > 1]
    sp = d[live].corr("spearman")
    sp.to_csv(TAB / "p2d_feature_correlation_spearman.csv", encoding="utf-8-sig")
    pe = d[live].corr()
    iu = np.triu_indices(len(live), 1)
    pairs = pd.DataFrame([{"feature_a": live[i], "feature_b": live[j], "spearman": sp.values[i, j],
                           "pearson": pe.values[i, j]} for i, j in zip(*iu)])
    pairs = pairs.reindex(pairs.spearman.abs().sort_values(ascending=False).index)
    save(pairs[pairs.spearman.abs() >= 0.7], "p2d_high_corr_pairs.csv")

    # 히트맵 (|ρ| 기준 계층 군집 순서)
    dist = 1 - np.abs(sp.values)
    np.fill_diagonal(dist, 0)
    order = leaves_list(linkage(squareform(dist, checks=False), "average"))
    m = sp.values[np.ix_(order, order)]
    names = [live[i] for i in order]
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(m, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(names)), names, fontsize=7.5)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and abs(m[i, j]) >= 0.7:
                ax.text(j, i, f"{m[i, j]:.2f}".replace("0.", "."), ha="center", va="center", fontsize=6,
                        color="white" if abs(m[i, j]) > 0.85 else "#0b0b0b")
    cb = fig.colorbar(im, ax=ax, shrink=0.7)
    cb.set_label("Spearman ρ", fontsize=8)
    ax.set_title("공정 변수 간 상관 (병합 591행, 비슷한 변수끼리 군집 정렬, |ρ| ≥ 0.7만 숫자 표시)", loc="left")
    fig.savefig(FIG / "p2d_correlation_heatmap.png"); plt.close(fig)

    # 불량과의 상관 (원 변수 + 파생변수)
    full = sets["S3 S2+시간 파생"]
    y = d[LABEL].to_numpy()
    before = (d[IDX] < REGIME_START_IDX).to_numpy()
    cands = [c for c in live] + [c for c in full.columns if c not in feats and full[c].nunique() > 1]
    n = len(y)
    rows = []
    for c in cands:
        x = full[c] if c in full.columns else d[c]
        r, p = stats.pointbiserialr(y, x)
        z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
        rb = stats.pointbiserialr(y[before], x[before])[0] if x[before].nunique() > 1 else np.nan
        ra = stats.pointbiserialr(y[~before], x[~before])[0] if x[~before].nunique() > 1 else np.nan
        rows.append({"feature": c, "type": "원 변수" if c in feats else "파생", "r": r,
                     "ci_low": np.tanh(z - 1.96 * se), "ci_high": np.tanh(z + 1.96 * se), "p": p,
                     "r_before_1891": rb, "r_after_1891": ra})
    lc = pd.DataFrame(rows)
    lc["p_bonferroni"] = np.minimum(lc.p * len(lc), 1.0)
    lc = lc.reindex(lc.r.abs().sort_values(ascending=False).index)
    save(lc, "p2d_label_correlation.csv")

    tcrit = stats.t.ppf(1 - 0.025 / len(lc), n - 2)
    rcrit = tcrit / np.sqrt(n - 2 + tcrit ** 2)
    o = lc.sort_values("r")
    fig, ax = plt.subplots(figsize=(8.5, 10))
    ypos = np.arange(len(o))
    for k, r in enumerate(o.itertuples()):
        c = C_L if r.type == "원 변수" else C_U
        ax.plot([r.ci_low, r.ci_high], [k, k], color=c, lw=1.5, alpha=0.6)
        ax.plot(r.r, k, "o", color=c, ms=6, mec="white")
    for s in (-1, 1):
        ax.axvline(s * rcrit, color=C_FAIL, lw=1, ls="--")
    ax.text(rcrit, len(o) - 0.5, f" 다중검정 보정 유의선 |r|={rcrit:.3f}", fontsize=7.5, color=INK2, va="bottom")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(ypos, o.feature, fontsize=7.5); ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.8, len(o) + 0.8)
    ax.plot([], [], "o", color=C_L, label="원 변수"); ax.plot([], [], "o", color=C_U, label="파생변수")
    ax.legend(loc="lower right")
    ax.set_xlabel("불량(1)과의 점이연 상관계수 r (선 = 95% 신뢰구간)")
    ax.set_title(f"불량과 각 변수의 상관 ({len(lc)}개 변수, 병합 591행, 불량 25건)", loc="left")
    fig.savefig(FIG / "p2d_label_correlation.png"); plt.close(fig)
    return pairs, lc, rcrit


# ================================================================ 변수 세트별 CV
def run_fold_fs(set_name, algo, X, y, tr, te, fold):
    model, thr = fit_with_thresholds(make_model_fs(algo), X[tr], y[tr])
    p = model.predict_proba(X[te])[:, 1]
    return {"set": set_name, "model_name": algo, "algo": algo, "imbalance": f"class_weight|feat={code(set_name)}",
            "fold": fold, "n_features_used": len(model.named_steps["filter"].support_), **score(y[te], p, thr)}


def _perm_fs(algo, X, y_perm):
    auc, ap = [], []
    for tr, te in RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                          random_state=SEED).split(X, y_perm):
        p = make_model_fs(algo).fit(X[tr], y_perm[tr]).predict_proba(X[te])[:, 1]
        auc.append(roc_auc_score(y_perm[te], p))
        ap.append(average_precision_score(y_perm[te], p))
    return np.mean(auc), np.mean(ap)


def plot_sets(ext, nt, prev):
    sets = ext.set.unique().tolist()
    rows = [(s, a) for s in sets for a in ALGOS]
    e = ext.set_index(["set", "algo"])
    ypos = {r: i + sets.index(r[0]) * 0.6 for i, r in enumerate(rows)}
    labels = [f"{s} | {a}" for s, a in rows]
    last = code(sets[-1])

    fig, axes = plt.subplots(1, 4, figsize=(15, 5.6), sharey=True)
    panels = (("f1", "F1-Score"), ("precision", "Precision"), ("recall", "Recall"), ("prauc", "PR-AUC"))
    for ax, (m, ttl) in zip(axes, panels):
        for (s, a) in rows:
            k = ypos[(s, a)]
            if m == "prauc":
                v, sd = e.loc[(s, a), "prauc_mean"], e.loc[(s, a), "prauc_std"]
                ax.errorbar(v, k, xerr=sd, fmt="o", color=ALGO_COLOR[a], ms=6, lw=1.3, mec="white")
            else:
                for j, (rule, filled) in enumerate((("F1max", True), ("Recall80", False))):
                    v, sd = e.loc[(s, a), f"{m}_{rule}_mean"], e.loc[(s, a), f"{m}_{rule}_std"]
                    ax.errorbar(v, k + (j - 0.5) * 0.3, xerr=sd, fmt="o", color=ALGO_COLOR[a], ms=5.5, lw=1.1,
                                mfc=ALGO_COLOR[a] if filled else "white", mew=1.2)
        if m in ("precision", "prauc"):
            ax.axvline(prev, color=INK2, ls="--", lw=1)
            ax.text(prev, -1.3, " 불량률", fontsize=7.5, color=INK2, va="center")
        if m == "prauc":
            for a in ALGOS:
                ax.plot(nt.set_index("algo").loc[a, "null_prauc_mean"], ypos[(sets[-1], a)], marker="|",
                        color=INK2, ms=14, mew=2, zorder=4)
        ax.set_title(ttl if m != "prauc" else f"PR-AUC (세로 막대 = {last} 순열 기준선)", loc="left")
        ax.grid(axis="y", visible=False)
        if m != "prauc":
            ax.set_xlim(-0.03, 1.03)
    axes[0].set_yticks([ypos[r] for r in rows], labels, fontsize=8)
    axes[0].set_ylim(max(ypos.values()) + 0.6, -1.8)
    axes[1].plot([], [], "o", color=INK2, label="F1 최대 임계값")
    axes[1].plot([], [], "o", color=INK2, mfc="white", label="Recall ≥ 0.8 임계값")
    axes[1].legend(loc="lower right")
    fig.suptitle("변수 세트별 성능 (병합 591행, 5-fold×10, 평균 ± 표준편차, class_weight)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "p2d_featureset_metrics.png"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    for ax, (m, ref, ttl) in zip(axes, (("roc_auc", 0.5, "ROC-AUC (참고)"), ("recall_at_20", 0.2, "Recall@상위 20%"))):
        for (s, a) in rows:
            ax.errorbar(e.loc[(s, a), f"{m}_mean"], ypos[(s, a)], xerr=e.loc[(s, a), f"{m}_std"], fmt="o",
                        color=ALGO_COLOR[a], ms=6, lw=1.3, mec="white")
        ax.axvline(ref, color=INK2, ls="--", lw=1)
        ax.text(ref, -1.3, " 무작위", fontsize=7.5, color=INK2, va="center")
        ax.set_title(ttl, loc="left"); ax.grid(axis="y", visible=False)
    axes[0].set_yticks([ypos[r] for r in rows], labels, fontsize=8)
    axes[0].set_ylim(max(ypos.values()) + 0.6, -1.8)
    fig.suptitle("변수 세트별 순위 성능 (병합 591행, 5-fold×10)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "p2d_featureset_ranking.png"); plt.close(fig)


def main():
    raw, _, feats = load_raw()
    d, sets = build_feature_sets(merge_pairs(raw, feats), feats)
    y = d[LABEL].to_numpy()

    desc = pd.DataFrame([{"set": s, "n_features": X.shape[1], "features": ", ".join(X.columns)}
                         for s, X in sets.items()])
    save(desc, "p2d_feature_sets.csv")

    pairs, lc, rcrit = correlation(d, sets, feats)

    splits = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                          random_state=SEED).split(d[feats], y))
    jobs = [(s, a, f, tr, te) for s in sets for a in ALGOS for f, (tr, te) in enumerate(splits)]
    recs = Parallel(n_jobs=-1)(delayed(run_fold_fs)(s, a, sets[s].to_numpy(), y, tr, te, f)
                               for s, a, f, tr, te in jobs)
    folds = pd.DataFrame(recs)
    save(folds, "p2d_cv_fold_metrics.csv")
    ext = extended_table(folds.drop(columns=["model_name", "imbalance"]).rename(columns={"set": "imbalance"}))
    ext = ext.rename(columns={"imbalance": "set"})
    save(ext, "p2d_model_comparison_extended.csv")
    team = team_table(folds.drop(columns=["set", "model_name", "n_features_used"]), PRODUCT)[TEAM_COLS]
    save(team, "p2d_model_comparison_featuresets.csv")

    last = list(sets)[-1]
    Xl = sets[last].to_numpy()
    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(y) for _ in range(N_PERM)]
    res = Parallel(n_jobs=-1)(delayed(_perm_fs)(a, Xl, perms[i]) for a in ALGOS for i in range(N_PERM))
    null = pd.DataFrame([{"algo": a, "perm": i, "roc_auc": r[0], "prauc": r[1]}
                         for (a, i), r in zip([(a, i) for a in ALGOS for i in range(N_PERM)], res)])
    save(null, "p2d_null_distribution.csv")
    e = ext[ext.set == last].set_index("algo")
    nt = pd.DataFrame([{
        "set": last, "algo": a, "observed_roc_auc": e.loc[a, "roc_auc_mean"], "observed_prauc": e.loc[a, "prauc_mean"],
        "null_roc_auc_mean": g.roc_auc.mean(), "null_roc_auc_std": g.roc_auc.std(),
        "p_roc_auc_greater": (1 + (g.roc_auc >= e.loc[a, "roc_auc_mean"]).sum()) / (N_PERM + 1),
        "null_prauc_mean": g.prauc.mean(), "null_prauc_std": g.prauc.std(),
        "p_prauc_greater": (1 + (g.prauc >= e.loc[a, "prauc_mean"]).sum()) / (N_PERM + 1), "n_perm": N_PERM}
        for a, g in null.groupby("algo", sort=False)])
    save(nt, "p2d_null_test.csv")

    plot_sets(ext, nt, y.mean())

    pd.set_option("display.width", 250)
    print("[변수 세트]\n", desc[["set", "n_features"]].to_string(index=False))
    print("\n[고상관 쌍 |ρ|≥0.7]\n", pairs[pairs.spearman.abs() >= 0.7].to_string(index=False))
    print(f"\n[불량 상관] 다중검정 유의선 |r|={rcrit:.3f}\n", lc.head(12).to_string(index=False))
    cols = ["set", "algo", "n_features_used_mean", "f1_F1max_mean", "precision_F1max_mean", "recall_F1max_mean",
            "f1_Recall80_mean", "precision_Recall80_mean", "recall_Recall80_mean", "prauc_mean", "prauc_std",
            "roc_auc_mean", "recall_at_20_mean"]
    print("\n[변수 세트별 성능]\n", ext[cols].round(3).to_string(index=False))
    print("\n[순열 검정]\n", nt.to_string(index=False))


if __name__ == "__main__":
    main()
