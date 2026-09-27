"""Phase 2. 모델 비교 (RG3, 중복 쌍 병합 데이터 기준).

실행: python src/phase2_models.py
산출물: outputs/tables/model_comparison_RG3.csv (팀 공통 형식), feature_importance_RG3.csv,
        outputs/tables/p2_*.csv, outputs/figures/p2_*.png
"""
from pathlib import Path
import random

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedGroupKFold, StratifiedKFold,
                                     TimeSeriesSplit)

from preprocess import IDX, LABEL, PRODUCT, RankTransformer, load_raw, merge_pairs
from train import ALGOS, IMBALANCE, SEED, fit_with_thresholds, make_model
from evaluate import RULES, extended_table, run_fold, score, team_table
from viz import ALGO_COLOR, C_FAIL, C_L, C_PASS, C_U, IMB_MARKER, INK2, plt

random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
N_SPLITS, N_REPEATS = 5, 10


def save(df, name):
    df.to_csv(TAB / name, index=False, encoding="utf-8-sig")


# ================================================================ ① 병합 전후 누수 확인
def leak_check(raw, feats):
    raw = raw.copy()
    raw["pair"] = raw.groupby(feats, sort=False).ngroup()
    merged = merge_pairs(raw, feats)
    Xr, yr, gr = raw[feats].to_numpy(), raw[LABEL].to_numpy(), raw["pair"].to_numpy()
    Xm, ym = merged[feats].to_numpy(), merged[LABEL].to_numpy()

    def grouped_splits():
        for r in range(N_REPEATS):
            yield from StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED + r).split(Xr, yr, gr)

    settings = [
        ("A. 원본 1182행 | 5-fold 1회 (사전관찰 재현)", Xr, yr,
         lambda: StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(Xr, yr)),
        ("B. 원본 1182행 | 5-fold×10 (쌍둥이가 train/test로 갈림)", Xr, yr,
         lambda: RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(Xr, yr)),
        ("C. 원본 1182행 | 쌍 단위 묶음 5-fold×10", Xr, yr, grouped_splits),
        ("D. 병합 591행 | 5-fold×10 (채택)", Xm, ym,
         lambda: RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(Xm, ym)),
    ]
    rows, mech = [], []
    for algo in ("LogReg", "RF"):
        model = make_model(algo, "class_weight")
        for name, X, y, splits in settings:
            auc, ap = [], []
            for tr, te in splits():
                p = clone(model).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                auc.append(roc_auc_score(y[te], p))
                ap.append(average_precision_score(y[te], p))
                if name.startswith("B"):
                    # 누수 메커니즘: 테스트 불량행의 쌍둥이(양품)가 train에 있을 때 순위가 떨어지는가
                    tr_set, neg = set(tr), p[y[te] == 0]
                    for i in np.flatnonzero(y[te] == 1):
                        twin = np.flatnonzero((gr == gr[te[i]]) & (np.arange(len(gr)) != te[i]))[0]
                        mech.append({"algo": algo, "twin_in_train": twin in tr_set,
                                     "defect_percentile": (neg < p[i]).mean() + 0.5 * (neg == p[i]).mean()})
            rows.append({"algo": algo, "setting": name, "n": len(y), "n_fail": int(y.sum()),
                         "roc_auc_mean": np.mean(auc), "roc_auc_std": np.std(auc, ddof=1) if len(auc) > 1 else np.nan,
                         "prauc_mean": np.mean(ap), "prauc_std": np.std(ap, ddof=1) if len(ap) > 1 else np.nan,
                         "prauc_baseline": y.mean(), "n_folds": len(auc)})
    out = pd.DataFrame(rows)
    mech = (pd.DataFrame(mech).groupby(["algo", "twin_in_train"])
              .defect_percentile.agg(["mean", "std", "size"]).reset_index())
    save(out, "p2_leak_check.csv")
    save(mech, "p2_leak_mechanism.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    labels = [s[0] for s in settings]
    for ax, metric, ttl in ((axes[0], "roc_auc", "ROC-AUC (참고, 0.5 = 무작위)"),
                            (axes[1], "prauc", "PR-AUC ÷ 불량률 (1 = 무작위)")):
        for j, algo in enumerate(("LogReg", "RF")):
            d = out[out.algo == algo].set_index("setting").loc[labels]
            base = 1 if metric == "roc_auc" else d.prauc_baseline
            m, s = d[f"{metric}_mean"] / base, d[f"{metric}_std"] / base
            ypos = np.arange(len(labels)) + (j - 0.5) * 0.25
            ax.errorbar(m, ypos, xerr=s, fmt="o", color=ALGO_COLOR[algo], ms=6, lw=1.5, capsize=0, label=algo)
        ax.axvline(0.5 if metric == "roc_auc" else 1, color=INK2, lw=1, ls="--")
        ax.set_title(ttl, loc="left"); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(labels)), labels); axes[0].invert_yaxis()
    axes[1].legend(loc="lower right")
    fig.suptitle("중복 처리 방식별 CV 성능 (평균 ± 표준편차, class_weight='balanced')", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "p2_leak_check.png"); plt.close(fig)
    return out, mech


# ================================================================ ① 라벨 순열 귀무분포
def _perm_cv(algo, X, y_perm):
    auc, ap = [], []
    for tr, te in RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                          random_state=SEED).split(X, y_perm):
        p = make_model(algo, "class_weight").fit(X[tr], y_perm[tr]).predict_proba(X[te])[:, 1]
        auc.append(roc_auc_score(y_perm[te], p))
        ap.append(average_precision_score(y_perm[te], p))
    return np.mean(auc), np.mean(ap)


def null_test(data, feats, ext, n_perm=30):
    """라벨을 섞어 '신호가 전혀 없을 때' 같은 CV 절차가 내는 AUC/PR-AUC 분포를 만든다."""
    X, y = data[feats].to_numpy(), data[LABEL].to_numpy()
    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(y) for _ in range(n_perm)]
    jobs = [(a, i) for a in ALGOS for i in range(n_perm)]
    res = Parallel(n_jobs=-1)(delayed(_perm_cv)(a, X, perms[i]) for a, i in jobs)
    null = pd.DataFrame([{"algo": a, "perm": i, "roc_auc": r[0], "prauc": r[1]} for (a, i), r in zip(jobs, res)])
    save(null, "p2_null_distribution.csv")
    e = ext[ext.imbalance == "class_weight"].set_index("algo")
    rows = []
    for a in ALGOS:
        nd = null[null.algo == a]
        oa, op = e.loc[a, "roc_auc_mean"], e.loc[a, "prauc_mean"]
        rows.append({"algo": a, "imbalance": "class_weight",
                     "observed_roc_auc": oa, "null_roc_auc_mean": nd.roc_auc.mean(), "null_roc_auc_std": nd.roc_auc.std(),
                     "p_roc_auc_greater": (1 + (nd.roc_auc >= oa).sum()) / (n_perm + 1),
                     "observed_prauc": op, "null_prauc_mean": nd.prauc.mean(), "null_prauc_std": nd.prauc.std(),
                     "p_prauc_greater": (1 + (nd.prauc >= op).sum()) / (n_perm + 1), "n_perm": n_perm})
    out = pd.DataFrame(rows)
    save(out, "p2_null_test.csv")

    fig, axes = plt.subplots(2, 3, figsize=(12, 5.2))
    for j, a in enumerate(ALGOS):
        nd, r = null[null.algo == a], out.iloc[j]
        for i, (m, obs, ttl) in enumerate((("roc_auc", r.observed_roc_auc, "ROC-AUC"),
                                           ("prauc", r.observed_prauc, "PR-AUC"))):
            ax = axes[i, j]
            ax.hist(nd[m], bins=12, color=C_PASS, alpha=0.8, label="라벨 순열(신호 없음)")
            ax.axvline(obs, color=ALGO_COLOR[a], lw=2.2, label="실제 라벨")
            p = r.p_roc_auc_greater if m == "roc_auc" else r.p_prauc_greater
            ax.set_title(f"{a} — {ttl}  (순열 p={p:.2f})", loc="left"); ax.set_yticks([])
    axes[0, 0].legend(loc="upper left")
    fig.suptitle(f"실제 성능 vs 라벨 순열 귀무분포 ({n_perm}회, 각 회 5-fold×10 평균, 병합 591행, class_weight)",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "p2_null_test.png"); plt.close(fig)
    return out


# ================================================================ 메인: 반복 층화 K-Fold
def main_cv(data, feats):
    X, y = data[feats].to_numpy(), data[LABEL].to_numpy()
    splits = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                          random_state=SEED).split(X, y))
    jobs = [(algo, imb, f, tr, te) for algo in ALGOS for imb in IMBALANCE for f, (tr, te) in enumerate(splits)]
    res = Parallel(n_jobs=-1)(
        delayed(run_fold)(a, i, X, y, tr, te, f, feats if i == "class_weight" else None)
        for a, i, f, tr, te in jobs)
    folds = pd.DataFrame([r[0] for r in res])
    folds["repeat"] = folds.fold // N_SPLITS
    save(folds, "p2_cv_fold_metrics.csv")

    team = team_table(folds, PRODUCT)
    team.to_csv(TAB / f"model_comparison_{PRODUCT}.csv", index=False, encoding="utf-8-sig")
    ext = extended_table(folds.drop(columns="repeat"))
    save(ext, "p2_model_comparison_extended.csv")

    imp = pd.concat([r[2] for r in res if r[2] is not None])
    fi = (imp.groupby(["algo", "imbalance", "feature"]).importance.agg(["mean", "std"]).reset_index()
             .rename(columns={"mean": "importance_mean", "std": "importance_std"}))
    fi.insert(0, "model", fi.algo + "|" + fi.imbalance)
    fi.insert(1, "product", PRODUCT)
    fi["method"] = "permutation_importance(PR-AUC 감소, test fold, 5-fold×10 평균)"
    fi = fi.drop(columns=["algo", "imbalance"]).sort_values(["model", "importance_mean"], ascending=[True, False])
    fi.to_csv(TAB / f"feature_importance_{PRODUCT}.csv", index=False, encoding="utf-8-sig")

    # 1회차 반복의 out-of-fold 확률 (PR 곡선용)
    oof = {}
    for (a, i, f, _, _), r in zip(jobs, res):
        if f < N_SPLITS:
            te, p = r[1]
            oof.setdefault((a, i), np.zeros(len(y)))[te] = p
    return folds, team, ext, fi, oof, y


def plot_main(folds, ext, fi, oof, y):
    prev = y.mean()
    cfgs = [(a, i) for a in ALGOS for i in IMBALANCE]
    labels = [f"{a} | {i}" for a, i in cfgs]
    e = ext.set_index(["algo", "imbalance"])

    # (1) 순위 지표: PR-AUC, ROC-AUC
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, m, ref, ttl in ((axes[0], "prauc", prev, f"PR-AUC (점선 = 불량률 {prev:.3f})"),
                            (axes[1], "roc_auc", 0.5, "ROC-AUC (참고, 점선 = 0.5)")):
        for k, (a, i) in enumerate(cfgs):
            ax.errorbar(e.loc[(a, i), f"{m}_mean"], k, xerr=e.loc[(a, i), f"{m}_std"], fmt=IMB_MARKER[i],
                        color=ALGO_COLOR[a], ms=7, lw=1.5, mec="white", mew=1)
        ax.axvline(ref, color=INK2, lw=1, ls="--")
        ax.set_title(ttl, loc="left"); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(cfgs)), labels); axes[0].invert_yaxis()
    fig.suptitle("모델 × 불균형 대응별 순위 성능 (병합 591행, 5-fold×10, 평균 ± 표준편차)", x=0.01, ha="left")
    fig.savefig(FIG / "p2_model_ranking_metrics.png"); plt.close(fig)

    # (2) 임계값 규칙별 F1 / Recall / Precision
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, m in zip(axes, ("f1", "recall", "precision")):
        for k, (a, i) in enumerate(cfgs):
            for j, (rule, fc) in enumerate((("F1max", True), ("Recall80", False))):
                v, s = e.loc[(a, i), f"{m}_{rule}_mean"], e.loc[(a, i), f"{m}_{rule}_std"]
                ax.errorbar(v, k + (j - 0.5) * 0.3, xerr=s, fmt=IMB_MARKER[i], ms=6, lw=1.2,
                            color=ALGO_COLOR[a], mfc=ALGO_COLOR[a] if fc else "white", mew=1.3)
        ax.set_title(m.capitalize(), loc="left"); ax.set_xlim(-0.02, 1.02); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(cfgs)), labels); axes[0].invert_yaxis()
    axes[2].plot([], [], "o", color=INK2, label="F1 최대 임계값 (채움)")
    axes[2].plot([], [], "o", color=INK2, mfc="white", label="Recall ≥ 0.8 임계값 (빈 원)")
    axes[2].legend(loc="lower right")
    fig.suptitle("임계값 규칙별 분류 성능 (임계값은 각 학습 fold 내부 3-fold OOF로 결정)", x=0.01, ha="left")
    fig.savefig(FIG / "p2_threshold_metrics.png"); plt.close(fig)

    # (3) Recall@상위 k%
    fig, ax = plt.subplots(figsize=(8, 4))
    ks = [5, 10, 20]
    for a, i in cfgs:
        ax.plot(ks, [e.loc[(a, i), f"recall_at_{k}_mean"] for k in ks], marker=IMB_MARKER[i],
                color=ALGO_COLOR[a], lw=1.5, ms=6, mec="white", label=f"{a} | {i}")
    ax.plot(ks, [k / 100 for k in ks], color=INK2, ls="--", lw=1, label="무작위 검사")
    ax.set_xticks(ks, [f"상위 {k}%" for k in ks]); ax.set_ylabel("불량 포착률 (Recall)")
    ax.legend(ncol=2, fontsize=7, loc="upper left")
    ax.set_title("검사 비율별 불량 포착률 (test fold 기준, 5-fold×10 평균)", loc="left")
    fig.savefig(FIG / "p2_recall_at_k.png"); plt.close(fig)

    # (4) PR 곡선 (1회차 OOF, class_weight 설정)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for a in ALGOS:
        p = oof[(a, "class_weight")]
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, color=ALGO_COLOR[a], lw=2, label=f"{a} (AP={average_precision_score(y, p):.3f})")
    ax.axhline(prev, color=INK2, ls="--", lw=1, label=f"무작위 ({prev:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_ylim(0, 1.02)
    ax.legend(); ax.set_title("PR 곡선 — 1회차 반복의 out-of-fold 확률, class_weight", loc="left")
    fig.savefig(FIG / "p2_pr_curves_oof.png"); plt.close(fig)

    # (5) 특징 중요도
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5), sharey=True)
    order = (fi.groupby("feature").importance_mean.mean().sort_values(ascending=False).index.tolist())
    for ax, a in zip(axes, ALGOS):
        d = fi[fi.model == f"{a}|class_weight"].set_index("feature").loc[order]
        ax.barh(range(len(order)), d.importance_mean, xerr=d.importance_std / np.sqrt(N_SPLITS * N_REPEATS),
                color=ALGO_COLOR[a], height=0.7, error_kw={"lw": 0.8, "ecolor": INK2})
        ax.axvline(0, color=INK2, lw=0.8); ax.set_title(f"{a} | class_weight", loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)), order); axes[0].invert_yaxis()
    fig.supxlabel("PR-AUC 감소량 (permutation, 오차막대 = 표준오차)", fontsize=9)
    fig.suptitle("특징 중요도 — test fold permutation importance, 5-fold×10 평균", x=0.01, ha="left")
    fig.savefig(FIG / "p2_feature_importance.png"); plt.close(fig)


# ================================================================ ③ 1890 전후 조건 변화
def changepoint(data, feats):
    """평균 이동이 가장 큰 단일 변화점 (전체 특징의 집단 간 제곱합 합계 최대)."""
    Z = data[feats].to_numpy()
    n = len(Z)
    cs = np.cumsum(Z, axis=0)
    tot = cs[-1]
    ks = np.arange(20, n - 20)
    m1 = cs[ks - 1] / ks[:, None]
    m2 = (tot - cs[ks - 1]) / (n - ks)[:, None]
    gain = (ks * (n - ks) / n)[:, None] * (m1 - m2) ** 2
    return ks, gain.sum(1)


def regime_analysis(data, feats):
    ks, gain = changepoint(data, feats)
    k = ks[np.argmax(gain)]
    cp_idx = int(data[IDX].iloc[k])
    before, after = data.iloc[:k], data.iloc[k:]
    tab = [[before[LABEL].sum(), len(before) - before[LABEL].sum()],
           [after[LABEL].sum(), len(after) - after[LABEL].sum()]]
    fisher = stats.fisher_exact(tab)
    summ = pd.DataFrame([
        {"regime": "before", "idx_from": before[IDX].min(), "idx_to": before[IDX].max(), "n": len(before),
         "n_fail": int(before[LABEL].sum()), "fail_rate": before[LABEL].mean()},
        {"regime": "after", "idx_from": after[IDX].min(), "idx_to": after[IDX].max(), "n": len(after),
         "n_fail": int(after[LABEL].sum()), "fail_rate": after[LABEL].mean()},
    ])
    summ["detected_changepoint_idx"] = cp_idx
    summ["fisher_odds_ratio"] = fisher.statistic
    summ["fisher_p"] = fisher.pvalue
    save(summ, "p2_regime_summary.csv")

    def smd(a, b):
        s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        return (b.mean() - a.mean()) / s if s > 0 else np.nan

    rows = []
    for c in feats:
        r = {"feature": c, "mean_before": before[c].mean(), "mean_after": after[c].mean(),
             "smd_after_vs_before": smd(before[c], after[c]),
             "ks_p": stats.ks_2samp(before[c], after[c]).pvalue if data[c].std() > 0 else np.nan}
        for name, d in (("before", before), ("after", after), ("all", data)):
            r[f"smd_fail_vs_pass_{name}"] = smd(d.loc[d[LABEL] == 0, c], d.loc[d[LABEL] == 1, c])
        rows.append(r)
    feat = pd.DataFrame(rows)
    save(feat, "p2_regime_feature_shift.csv")
    f = feat.dropna(subset=["smd_fail_vs_pass_before", "smd_fail_vs_pass_after"])
    sign_agree = int((np.sign(f.smd_fail_vs_pass_before) == np.sign(f.smd_fail_vs_pass_after)).sum())
    rho = stats.spearmanr(f.smd_fail_vs_pass_before, f.smd_fail_vs_pass_after)

    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.6])
    ax = fig.add_subplot(gs[0, :])
    ax.plot(data[IDX].iloc[ks], gain, color=C_L, lw=1.8)
    ax.axvline(cp_idx, color=C_FAIL, lw=1.2)
    ax.text(cp_idx + 8, gain.min(), f"최대 변화점 idx={cp_idx}", color=INK2, va="bottom", fontsize=8.5)
    ax.set_ylabel("평균 이동 점수"); ax.set_xlabel("원본 인덱스 (병합 후 쌍의 앞 인덱스)")
    ax.set_title("단일 변화점 탐색 — 전체 특징의 전후 평균 차이 합계 (값이 클수록 조건 변화가 큼)", loc="left")

    ax = fig.add_subplot(gs[1, 0])
    o = feat.dropna(subset=["smd_after_vs_before"]).sort_values("smd_after_vs_before")
    ax.barh(o.feature, o.smd_after_vs_before, color=[C_U if v > 0 else C_L for v in o.smd_after_vs_before], height=0.7)
    ax.axvline(0, color=INK2, lw=0.8); ax.grid(axis="y", visible=False); ax.tick_params(axis="y", labelsize=7.5)
    ax.set_xlabel("표준화 평균차 (이후 - 이전)")
    ax.set_title(f"변화점 전후 특징 이동 (이전 {len(before)}행 / 이후 {len(after)}행)", loc="left")

    ax = fig.add_subplot(gs[1, 1])
    ax.scatter(f.smd_fail_vs_pass_before, f.smd_fail_vs_pass_after, s=36, color=C_L, edgecolor="white", zorder=3)
    lim = np.nanmax(np.abs(f[["smd_fail_vs_pass_before", "smd_fail_vs_pass_after"]].to_numpy())) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color=C_PASS, lw=0.8, ls="--")
    ax.axhline(0, color=INK2, lw=0.6); ax.axvline(0, color=INK2, lw=0.6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    for _, r in f.reindex(f[["smd_fail_vs_pass_before", "smd_fail_vs_pass_after"]].abs().max(1)
                          .sort_values(ascending=False).index).head(6).iterrows():
        ax.annotate(r.feature, (r.smd_fail_vs_pass_before, r.smd_fail_vs_pass_after), fontsize=7,
                    color=INK2, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel(f"불량-양품 표준화 평균차: 이전 (불량 {int(before[LABEL].sum())}건)")
    ax.set_ylabel(f"이후 (불량 {int(after[LABEL].sum())}건)")
    ax.set_title(f"불량 신호가 두 구간에서 일관한가 — 부호 일치 {sign_agree}/{len(f)}, "
                 f"Spearman ρ={rho.statistic:.2f}", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "p2_regime_shift.png"); plt.close(fig)
    return summ, feat, k, {"sign_agree": sign_agree, "n": len(f), "rho": rho.statistic, "rho_p": rho.pvalue}


# ================================================================ ③ 시간 기반 분할 (보조)
def time_splits(data, feats, k_cp):
    X, y = data[feats].to_numpy(), data[LABEL].to_numpy()
    n = len(y)
    cut = int(n * 0.7)
    schemes = [("holdout 70/30", [(np.arange(cut), np.arange(cut, n))]),
               ("expanding window (4분할)", list(TimeSeriesSplit(n_splits=4).split(X))),
               ("변화점 이전→이후", [(np.arange(k_cp), np.arange(k_cp, n))]),
               ("변화점 이후→이전", [(np.arange(k_cp, n), np.arange(k_cp))])]
    rows = []
    for scheme, splits in schemes:
        for algo in ALGOS:
            for imb in IMBALANCE:
                ps, ys, thr_rows = [], [], []
                for tr, te in splits:
                    model, thr = fit_with_thresholds(make_model(algo, imb), X[tr], y[tr])
                    p = model.predict_proba(X[te])[:, 1]
                    thr_rows.append(score(y[te], p, thr))
                    ps.append(p); ys.append(y[te])
                if len(splits) == 1:
                    rec = thr_rows[0]
                else:  # 청크별 불량이 적어 예측을 합쳐(pooled) 계산, 임계값 지표는 청크별로 적용 후 합산
                    rec = score(np.concatenate(ys), np.concatenate(ps))
                    pred = {r: np.concatenate([(p >= t[f"thr_{r}"]).astype(int) for p, t in zip(ps, thr_rows)])
                            for r in RULES}
                    yy = np.concatenate(ys)
                    for r in RULES:
                        tp = ((pred[r] == 1) & (yy == 1)).sum()
                        rec[f"recall_{r}"] = tp / yy.sum()
                        rec[f"precision_{r}"] = tp / max(pred[r].sum(), 1)
                        pr_, rc_ = rec[f"precision_{r}"], rec[f"recall_{r}"]
                        rec[f"f1_{r}"] = 2 * pr_ * rc_ / (pr_ + rc_) if pr_ + rc_ > 0 else 0.0
                        rec[f"thr_{r}"] = np.mean([t[f"thr_{r}"] for t in thr_rows])
                rows.append({"scheme": scheme, "algo": algo, "imbalance": imb,
                             "n_train_first": len(splits[0][0]),
                             "n_pos_train_first": int(y[splits[0][0]].sum()), **rec})
    out = pd.DataFrame(rows)
    save(out, "p2_time_split.csv")
    return out


def plot_time(ext, ts, prev_all):
    cfgs = [(a, "class_weight") for a in ALGOS]
    e = ext.set_index(["algo", "imbalance"])
    schemes = ["랜덤 CV (메인)"] + ts.scheme.unique().tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, m, ttl in ((axes[0], "prauc", "PR-AUC ÷ 평가셋 불량률 (1 = 무작위)"),
                       (axes[1], "recall_at_20", "Recall@상위 20% (0.2 = 무작위)")):
        for j, (a, i) in enumerate(cfgs):
            vals = [e.loc[(a, i), f"{m}_mean"] / (prev_all if m == "prauc" else 1)]
            for s in schemes[1:]:
                r = ts[(ts.scheme == s) & (ts.algo == a) & (ts.imbalance == i)].iloc[0]
                vals.append(r[m] / (r.prevalence if m == "prauc" else 1))
            ax.scatter(vals, np.arange(len(schemes)) + (j - 1) * 0.22, color=ALGO_COLOR[a], s=40,
                       edgecolor="white", zorder=3, label=a)
        ax.axvline(1 if m == "prauc" else 0.2, color=INK2, ls="--", lw=1)
        ax.set_title(ttl, loc="left"); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(schemes)), schemes); axes[0].invert_yaxis()
    axes[1].legend(loc="lower right")
    fig.suptitle("랜덤 CV vs 시간 기반 분할 (class_weight 설정)", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "p2_time_split.png"); plt.close(fig)


# ================================================================ ② 비라벨 순위 변환의 한계 근거
def unlabeled_evidence(lab_raw, data, unl, feats):
    step = lambda s: (lambda d: d[d > 1e-9].min() if (d > 1e-9).any() else np.nan)(np.diff(np.sort(s.unique())))
    runs = np.r_[0, np.cumsum(np.diff(unl[IDX].to_numpy()) > 1000)]
    unl = unl.assign(run=runs)
    rt = RankTransformer().fit(data, feats)
    mapped = rt.transform(unl)
    half = unl.iloc[: len(unl) // 2]
    mapped_half = rt.transform(half)

    per = []
    for c in feats:
        mode = data[c].mode().iloc[0]
        ratio = step(data[c]) / step(unl[c]) if data[c].nunique() > 1 else np.nan
        eta2 = unl.groupby("run")[c].apply(lambda s: len(s) * (s.mean() - unl[c].mean()) ** 2).sum() / \
            ((unl[c] - unl[c].mean()) ** 2).sum()
        per.append({"feature": c, "nunique_labeled": data[c].nunique(), "nunique_unlabeled": unl[c].nunique(),
                    "raw_std_ratio_U_over_L(est)": ratio,
                    "unlabeled_between_run_var_share": eta2,
                    "share_U_mapped_to_L_mode": float(np.isclose(mapped[c], mode).mean()),
                    "U_rows_|z|>5": int((unl[c].abs() > 5).sum()),
                    "batch_dependence_mean_abs_diff": float(np.abs(mapped_half[c].to_numpy()
                                                                   - mapped[c].iloc[: len(half)].to_numpy()).mean())})
    per = pd.DataFrame(per)
    save(per, "p2_unlabeled_rank_by_feature.csv")

    live = [c for c in feats if data[c].nunique() > 1]
    sl = data[live].corr("spearman").to_numpy()
    su = unl[live].corr("spearman").to_numpy()
    iu = np.triu_indices(len(live), 1)
    dcorr = np.abs(sl - su)[iu]
    worst = np.argmax(dcorr)
    run_len = unl.groupby("run").size()

    ev = pd.DataFrame([
        {"no": 1, "item": "두 파일이 각각 따로 z-표준화됨",
         "evidence": "라벨·비라벨 모두 변수별 평균≈0, 표준편차≈1 (p1_basic_stats.csv)",
         "implication": "같은 z값이 같은 원 단위 값을 뜻하지 않음 → 원 스케일 복원 없이 직접 적용 불가"},
        {"no": 2, "item": "원 단위 산포가 크게 다름",
         "evidence": f"값 최소간격 비율로 추정한 원단위 std 비(비라벨/라벨) 중앙값 "
                     f"{per['raw_std_ratio_U_over_L(est)'].median():.0f}배, 범위 "
                     f"{per['raw_std_ratio_U_over_L(est)'].min():.0f}~{per['raw_std_ratio_U_over_L(est)'].max():.0f}배",
         "implication": "라벨 데이터는 비라벨 공정범위의 아주 좁은 구간 → 순위 변환은 넓은 범위를 좁은 구간에 강제로 압축"},
        {"no": 3, "item": "비라벨은 여러 생산 구간이 섞여 있음",
         "evidence": f"비라벨 idx 간격>1000 기준 {len(run_len)}개 구간 (행수 중앙값 {int(run_len.median())}); "
                     f"금형온도3 구간간 분산 비중 {per.set_index('feature').loc['Mold_Temperature_3', 'unlabeled_between_run_var_share']:.2f}",
         "implication": "라벨(연속 1182행, 단일 기간)과 주변분포가 같다는 순위 변환의 핵심 가정이 성립하지 않음"},
        {"no": 4, "item": "라벨 쪽 고유값이 적어 정보가 뭉개짐",
         "evidence": f"라벨 고유값 ≤4개 변수 {int((per.nunique_labeled <= 4).sum())}개; "
                     f"Clamp_Open_Position 라벨 상수 → 비라벨 {per.set_index('feature').loc['Clamp_Open_Position', 'nunique_unlabeled']}개 값이 한 값으로; "
                     f"Injection_Time 비라벨의 {per.set_index('feature').loc['Injection_Time', 'share_U_mapped_to_L_mode']:.0%}가 라벨 최빈값으로 매핑",
         "implication": "비라벨의 공정 변동이 모델 입력에서 사라짐"},
        {"no": 5, "item": "변수 간 관계(결합분포)는 보정되지 않음",
         "evidence": f"Spearman 상관 차이(라벨 vs 비라벨) 평균 |Δρ|={dcorr.mean():.2f}, 최대 {dcorr.max():.2f} "
                     f"({live[iu[0][worst]]}–{live[iu[1][worst]]})",
         "implication": "변수별 순위 변환 후에도 조합 패턴이 학습 데이터와 다름 → 트리 모델의 분기 조합이 외삽됨"},
        {"no": 6, "item": "극단값이 라벨 범위로 잘림",
         "evidence": f"비라벨에서 |z|>5인 값 {int(per['U_rows_|z|>5'].sum())}개 (예: Max_Switch_Over_Pressure 최대 "
                     f"{unl.Max_Switch_Over_Pressure.max():.0f}σ) → 변환 후 라벨 최대값으로 매핑",
         "implication": "실제 이상 공정(불량 가능성 높은 샷)의 신호가 약해짐"},
        {"no": 7, "item": "배치 의존성",
         "evidence": f"비라벨 앞 절반만 변환했을 때와 전체로 변환했을 때 같은 행의 값 차이 평균 "
                     f"{per.batch_dependence_mean_abs_diff.mean():.2f} (라벨 z 단위)",
         "implication": "같은 샷도 함께 예측하는 데이터 묶음에 따라 입력값·확률이 달라짐 → 운영 시 재현성 한계"},
        {"no": 8, "item": "검증 불가",
         "evidence": "비라벨 데이터에 정답이 없고 라벨과 특징값 일치 행 0건 (p1_duplicate_summary.csv)",
         "implication": "변환 후 예측의 정확도를 직접 확인할 수 없음 → 예측 확률은 상대 순위(검사 우선순위)로만 해석"},
    ])
    save(ev, "p2_unlabeled_rank_evidence.csv")

    show = ["Mold_Temperature_3", "Hopper_Temperature", "Barrel_Temperature_4", "Max_Injection_Speed"]
    fig, axes = plt.subplots(len(show), 1, figsize=(11, 1.6 * len(show)), sharex=True)
    bounds = np.flatnonzero(np.diff(runs)) + 0.5
    for ax, c in zip(axes, show):
        ax.plot(np.arange(len(unl)), unl[c], color=C_U, lw=0.5)
        for b in bounds:
            ax.axvline(b, color=C_PASS, lw=0.5, alpha=0.7)
        ax.set_ylabel(c, rotation=0, ha="right", va="center", fontsize=8); ax.grid(axis="x", visible=False)
    axes[0].set_title(f"비라벨 데이터의 특징 추이 (행 순서, 회색 선 = idx 간격 >1000 인 생산 구간 경계, {len(run_len)}개 구간)",
                      loc="left")
    axes[-1].set_xlabel("비라벨 행 순서")
    fig.savefig(FIG / "p2_unlabeled_drift.png"); plt.close(fig)
    return ev, per


def main():
    raw, unl, feats = load_raw()
    data = merge_pairs(raw, feats)
    print(f"병합: {len(raw)}행 → {len(data)}행, 불량 {int(data[LABEL].sum())}건 ({data[LABEL].mean():.3%})")

    lk, mech = leak_check(raw, feats)
    print("\n[①] 누수 확인\n", lk.to_string(index=False), "\n", mech.to_string(index=False))

    folds, team, ext, fi, oof, y = main_cv(data, feats)
    plot_main(folds, ext, fi, oof, y)
    print("\n[메인] 팀 형식 결과\n", team.to_string(index=False))
    cols = ["algo", "imbalance", "prauc_mean", "roc_auc_mean", "recall_at_5_mean", "recall_at_10_mean", "recall_at_20_mean"]
    print(ext[cols].to_string(index=False))
    print("\n특징 중요도 상위\n", fi.groupby("model").head(5).to_string(index=False))

    nt = null_test(data, feats, ext)
    print("\n[①] 라벨 순열 귀무분포\n", nt.to_string(index=False))

    summ, feat, k_cp, cons = regime_analysis(data, feats)
    print("\n[③] 변화점\n", summ.to_string(index=False), "\n", cons)
    print(feat.sort_values("smd_after_vs_before", key=abs, ascending=False).head(10).to_string(index=False))

    ts = time_splits(data, feats, k_cp)
    plot_time(ext, ts, y.mean())
    print("\n[③] 시간 분할\n", ts[["scheme", "algo", "imbalance", "n_train_first", "n_pos_train_first", "n_test",
                                 "n_pos_test", "prevalence", "prauc", "roc_auc", "recall_at_20",
                                 "f1_F1max", "recall_Recall80", "precision_Recall80"]].to_string(index=False))

    ev, per = unlabeled_evidence(raw, data, unl, feats)
    print("\n[②] 순위 변환 한계 근거\n", ev[["no", "item", "evidence"]].to_string(index=False))


if __name__ == "__main__":
    main()
