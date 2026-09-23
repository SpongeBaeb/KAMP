"""Phase 3. 영향요인 및 오류분석 (RG3, 최종 모델 = HGB class_weight, 쌍 병합 데이터).

실행: python src/phase3_analysis.py   (Phase 2 스크립트 다음에 실행 — 이전 결과표를 근거표에 인용)
산출물: outputs/tables/p3_*.csv, feature_importance_RG3.csv 에 SHAP 행 추가, outputs/figures/p3_*.png

1. SHAP: 전체 데이터로 학습한 모델의 SHAP(주요 변수·상호작용). 신호가 없는 데이터이므로
   라벨 순열 모델의 SHAP 중요도(귀무분포)와 50개 fold 모델 간 순위 일관성으로 진짜 여부를 검증한다.
2. RG3 실패 원인: Phase 1~3 근거를 원인별로 정리 + 불량 25건으로 검출 가능한 최소 효과크기(검정력).
3. FN/FP 구간화: 팀 공통 5-fold×10 평가 fold 예측(임계값은 학습 fold 내부)으로 샷별 FN/FP 빈도를
   구하고, 변수 구간별로 집계해 오류가 몰리는 공정 조건이 있는지 검정한다.
"""
from pathlib import Path
import random

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold

from preprocess import IDX, LABEL, PRODUCT, REGIME_START_IDX, load_raw, merge_pairs
from train import SEED, fit_with_thresholds, make_model
from viz import C_FAIL, C_L, C_PASS, C_U, INK2, plt

random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
N_SPLITS, N_REPEATS = 5, 10
N_PERM = 30
ALGO, IMB = "HGB", "class_weight"
MODEL_NAME = f"{ALGO}|{IMB}"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f3ef", "#9cc3ee", "#2a78d6", "#123f7a"])


def save(df, name):
    df.to_csv(TAB / name, index=False, encoding="utf-8-sig")


def fit_full(X, y):
    m = make_model(ALGO, IMB).fit(X, y)
    keep = m.named_steps["drop_const"].get_support()
    return m, keep


def shap_of(m, X):
    Z = m.named_steps["drop_const"].transform(X)
    return shap.TreeExplainer(m.named_steps["clf"]).shap_values(Z), Z


# ================================================================ 평가 fold 예측 + fold별 SHAP
def run_fold(X, y, tr, te, fold):
    model, thr = fit_with_thresholds(make_model(ALGO, IMB), X[tr], y[tr])
    p = model.predict_proba(X[te])[:, 1]
    sv, _ = shap_of(model, X[te])
    return fold, te, p, thr, sv


def _null_shap(X, y_perm):
    m, _ = fit_full(X, y_perm)
    sv, _ = shap_of(m, X)
    return np.abs(sv).mean(0)


# ================================================================ 1. SHAP
def shap_analysis(d, X, y, feats, res):
    m, keep = fit_full(X, y)
    names = [f for f, k in zip(feats, keep) if k]
    sv, Z = shap_of(m, X)
    real = np.abs(sv).mean(0)

    rng = np.random.default_rng(SEED)
    null = np.array(Parallel(n_jobs=-1)(delayed(_null_shap)(X, rng.permutation(y)) for _ in range(N_PERM)))

    fold_imp = np.array([np.abs(r[4]).mean(0) for r in res])            # 50 × 특징
    ranks = np.array([stats.rankdata(-v) for v in fold_imp])
    rho = stats.spearmanr(fold_imp.T).statistic
    stability = rho[np.triu_indices(len(res), 1)].mean()

    oof = np.zeros_like(sv)
    for _, te, _, _, s in res:
        oof[te] += s / N_REPEATS

    imp = pd.DataFrame({
        "feature": names, "shap_mean_abs": real, "null_mean": null.mean(0), "null_p95": np.percentile(null, 95, 0),
        "ratio_to_null": real / null.mean(0), "perm_p": (1 + (null >= real).sum(0)) / (N_PERM + 1),
        "fold_mean_abs_mean": fold_imp.mean(0), "fold_mean_abs_std": fold_imp.std(0, ddof=1),
        "fold_rank_mean": ranks.mean(0), "fold_rank_std": ranks.std(0, ddof=1),
        "oof_shap_defect_mean": oof[y == 1].mean(0), "oof_shap_normal_mean": oof[y == 0].mean(0),
    }).sort_values("shap_mean_abs", ascending=False)
    imp["perm_p_bonferroni"] = np.minimum(imp.perm_p * len(imp), 1.0)
    imp["fold_rank_stability_spearman"] = stability
    save(imp, "p3_shap_importance.csv")

    # 팀 공통 특징 중요도 파일에 최종 모델 SHAP 행 추가 (재실행 시 교체)
    fi_path = TAB / f"feature_importance_{PRODUCT}.csv"
    method = "SHAP mean|value| (전체 데이터 학습, log-odds; std = 50개 fold 모델 간)"
    fi = pd.read_csv(fi_path)
    fi = fi[~((fi.model == MODEL_NAME) & (fi.method == method))]
    add = pd.DataFrame({"model": MODEL_NAME, "product": PRODUCT, "feature": imp.feature,
                        "importance_mean": imp.shap_mean_abs, "importance_std": imp.fold_mean_abs_std, "method": method})
    pd.concat([fi, add], ignore_index=True).to_csv(fi_path, index=False, encoding="utf-8-sig")

    # 상호작용
    iv = shap.TreeExplainer(m.named_steps["clf"]).shap_interaction_values(Z)
    mi = np.abs(iv).mean(0)
    iu = np.triu_indices(len(names), 1)
    inter = pd.DataFrame({"feature_a": [names[i] for i in iu[0]], "feature_b": [names[j] for j in iu[1]],
                          "mean_abs_interaction": 2 * mi[iu]}).sort_values("mean_abs_interaction", ascending=False)
    inter["share_of_total_abs_shap"] = inter.mean_abs_interaction / np.abs(sv).sum(1).mean()
    save(inter, "p3_shap_interactions.csv")

    # --- 그림: beeswarm
    plt.figure()
    shap.summary_plot(sv, Z, feature_names=names, max_display=15, show=False, plot_size=(9, 6.5))
    fig = plt.gcf()
    fig.axes[0].set_title("SHAP 요약 (HGB class_weight, 전체 591행 학습, 상위 15개 변수)", loc="left", fontsize=10)
    fig.axes[0].set_xlabel("SHAP 값 (log-odds, 양수 = 불량 쪽으로 밀어올림)")
    fig.savefig(FIG / "p3_shap_beeswarm.png"); plt.close(fig)

    # --- 그림: 실제 중요도 vs 순열 귀무분포
    o = imp.sort_values("shap_mean_abs")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    yk = np.arange(len(o))
    lo = [np.percentile(null[:, names.index(f)], 5) for f in o.feature]
    ax.hlines(yk, lo, o.null_p95, color=C_PASS, lw=5, alpha=0.6, label="라벨 순열 모델 5~95% 범위")
    ax.plot(o.null_mean, yk, "|", color=INK2, ms=12, mew=2, label="라벨 순열 모델 평균")
    ax.plot(o.shap_mean_abs, yk, "o", color=C_L, ms=7, mec="white", label="실제 라벨 모델")
    ax.set_yticks(yk, o.feature, fontsize=8); ax.grid(axis="y", visible=False)
    ax.set_xlabel("평균 |SHAP| (log-odds)")
    n_sig = int((imp.perm_p_bonferroni < 0.05).sum())
    ax.set_title(f"SHAP 중요도는 진짜인가 — 실제 vs 라벨 순열 {N_PERM}회 (보정 후 유의 {n_sig}/{len(imp)}개, "
                 f"fold 간 순위 일관성 ρ={stability:.2f})", loc="left", fontsize=9.5)
    ax.legend(loc="lower right", fontsize=7.5)
    fig.savefig(FIG / "p3_shap_importance_vs_null.png"); plt.close(fig)

    # --- 그림: 상위 4개 변수 의존도 (불량 강조) — 학습에 쓴 모델 vs 학습에 안 쓴 fold 모델
    top = imp.feature.head(4).tolist()
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.8), sharey="row")
    for row, (S, lab) in enumerate(((sv, "학습에 쓴 모델 (전체 591행)"), (oof, "학습에 안 쓴 fold 모델 (10회 평균)"))):
        for ax, f in zip(axes[row], top):
            j = names.index(f)
            jit = np.random.default_rng(SEED).normal(0, 0.02 * (Z[:, j].std() or 1), len(Z))
            ax.scatter(Z[y == 0, j] + jit[y == 0], S[y == 0, j], s=9, color=C_PASS, alpha=0.5, label="정상")
            ax.scatter(Z[y == 1, j] + jit[y == 1], S[y == 1, j], s=30, color=C_FAIL, edgecolor="white", lw=0.6,
                       label="불량", zorder=3)
            ax.axhline(0, color=INK2, lw=0.6)
            ax.set_title(f"{f} — {lab}", loc="left", fontsize=8.5)
            if row == 1:
                ax.set_xlabel("변수 값 (z)")
        axes[row, 0].set_ylabel("SHAP 값")
    axes[0, 0].legend(loc="lower left", fontsize=7.5)
    fig.suptitle("상위 4개 변수의 SHAP 의존도 — 위: 학습한 불량은 외워서 양수로 몰림 / 아래: 처음 보는 불량은 정상과 섞임",
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "p3_shap_dependence.png"); plt.close(fig)

    # --- 그림: 상호작용 히트맵 (중요도 상위 12개)
    t12 = imp.feature.head(12).tolist()
    idx = [names.index(f) for f in t12]
    M = 2 * mi[np.ix_(idx, idx)]
    np.fill_diagonal(M, np.nan)
    fig, ax = plt.subplots(figsize=(8, 6.8))
    im = ax.imshow(M, cmap=SEQ)
    ax.set_xticks(range(len(t12)), t12, rotation=90, fontsize=7.5); ax.set_yticks(range(len(t12)), t12, fontsize=7.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.75).set_label("평균 |SHAP 상호작용| (log-odds)", fontsize=8)
    top_pair = inter.iloc[0]
    ax.set_title(f"SHAP 상호작용 (중요도 상위 12개) — 최대 {top_pair.feature_a} × {top_pair.feature_b}",
                 loc="left", fontsize=9.5)
    fig.savefig(FIG / "p3_shap_interactions.png"); plt.close(fig)
    return imp, inter, stability, n_sig


# ================================================================ 2. RG3 실패 원인 + 검정력
def mde(n1, n2, alpha, power=0.8):
    return (stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * np.sqrt(1 / n1 + 1 / n2)


def failure_causes(d, feats, y, imp, n_sig, stability):
    live = [c for c in feats if d[c].nunique() > 1]
    smd = []
    for c in live:
        a, b = d.loc[y == 0, c], d.loc[y == 1, c]
        s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        smd.append({"feature": c, "smd_defect_vs_normal": (b.mean() - a.mean()) / s})
    smd = pd.DataFrame(smd)
    dmax = smd.smd_defect_vs_normal.abs().max()
    n1, n0 = int(y.sum()), int((y == 0).sum())
    ratio = n0 / n1
    alpha_b = 0.05 / len(live)
    ns = np.arange(5, 401)
    curve = pd.DataFrame({"n_defects": ns, "mde_alpha05": mde(ns, ns * ratio, 0.05),
                          "mde_bonferroni": mde(ns, ns * ratio, alpha_b)})
    need = int(ns[np.argmax(curve.mde_bonferroni.to_numpy() <= dmax)]) if (curve.mde_bonferroni <= dmax).any() else None
    save(curve, "p3_mde_curve.csv")
    save(smd.sort_values("smd_defect_vs_normal", key=abs, ascending=False), "p3_defect_effect_sizes.csv")

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(ns, curve.mde_bonferroni, color=C_L, lw=2, label=f"검출 가능 최소 효과 (다중검정 보정 α=0.05/{len(live)})")
    ax.plot(ns, curve.mde_alpha05, color=C_L, lw=1.2, ls="--", label="보정 없이 α=0.05")
    ax.axhline(dmax, color=C_FAIL, lw=1.2)
    ax.text(ns.max(), dmax, f"RG3 관측 최대 효과 |d|={dmax:.2f} ", color=C_FAIL, fontsize=8, ha="right", va="bottom")
    ax.axvline(n1, color=INK2, lw=0.8, ls=":")
    ax.plot(n1, mde(n1, n0, alpha_b), "o", color=C_L, ms=7, mec="white")
    ax.text(n1, mde(n1, n0, alpha_b), f"  현재 불량 {n1}건 → 최소 |d|={mde(n1, n0, alpha_b):.2f}", fontsize=8,
            color=INK2, va="bottom")
    if need:
        ax.axvline(need, color=C_FAIL, lw=0.8, ls=":")
        ax.text(need, ax.get_ylim()[1] * 0.92, f" 관측 효과를 검출하려면 불량 약 {need}건", fontsize=8, color=C_FAIL)
    ax.set_xlabel("불량 샷 수 (정상 비율은 현재와 동일)"); ax.set_ylabel("표준화 평균차 |d|")
    ax.set_title("단일 변수 기준 검정력 (80%) — 불량 25건으로 잡을 수 있는 차이의 크기", loc="left")
    ax.legend(loc="upper right", fontsize=7.5)
    fig.savefig(FIG / "p3_power_mde.png"); plt.close(fig)

    # Phase 1~3 근거 인용
    lc = pd.read_csv(TAB / "p2d_label_correlation.csv")
    fam = pd.read_csv(TAB / "p2_summary_by_family.csv").dropna(subset=["perm_p_prauc"])
    nest = pd.read_csv(TAB / "p2e_nested_null_test.csv").iloc[0]
    pos = pd.read_csv(TAB / "p2e_pair_position_test.csv").set_index("item").value
    nn = pd.read_csv(TAB / "p2c_nearest_normal_summary.csv").set_index("metric")
    rg = pd.read_csv(TAB / "p2_regime_feature_shift.csv").dropna(subset=["smd_fail_vs_pass_before", "smd_fail_vs_pass_after"])
    agree = int((np.sign(rg.smd_fail_vs_pass_before) == np.sign(rg.smd_fail_vs_pass_after)).sum())
    rho_rg = stats.spearmanr(rg.smd_fail_vs_pass_before, rg.smd_fail_vs_pass_after).statistic
    bs = pd.read_csv(TAB / "p1_basic_stats.csv")
    nun = bs[(bs.dataset == "labeled") & (bs["nunique"] > 1)]["nunique"]
    top = lc.iloc[0]
    causes = pd.DataFrame([
        {"no": 1, "cause": "공정 변수에 불량 신호가 없음",
         "evidence": f"불량과의 최대 상관 |r|={abs(top.r):.3f}({top.feature}, 보정 p={top.p_bonferroni:.2f}); "
                     f"모델 11종 순열 p {fam.perm_p_prauc.min():.2f}~{fam.perm_p_prauc.max():.2f}; "
                     f"22개 후보 중첩 CV PR-AUC {nest.observed_prauc:.3f} (순열 {nest.null_prauc_mean:.3f})",
         "source": "p2d_label_correlation.csv, p2_summary_by_family.csv, p2e_nested_null_test.csv"},
        {"no": 2, "cause": "같은 공정 조건에서 양품·불량이 함께 나옴 (쌍 구조)",
         "evidence": f"불량 25건 모두 특징값이 같은 양품 쌍둥이를 가짐; 불량은 쌍의 앞쪽 기록에 "
                     f"{pos['불량이 쌍의 앞쪽 기록인 수']}/25 (p={float(pos['이항검정 p (앞/뒤 반반 가정)']):.1e})",
         "source": "p1_duplicate_summary.csv, p2e_pair_position_test.csv"},
        {"no": 3, "cause": "불량이 공정 공간의 이상치가 아님",
         "evidence": f"최근접 정상 거리 중앙값 정상 {nn.loc['nn1_dist', 'normal_median']:.2f} vs 불량 "
                     f"{nn.loc['nn1_dist', 'defect_median']:.2f} (p={nn.loc['nn1_dist', 'mannwhitney_p']:.2f}); "
                     f"이상탐지 4종 ROC-AUC 0.43~0.47",
         "source": "p2c_nearest_normal_summary.csv, p2c_model_comparison_extended.csv"},
        {"no": 4, "cause": "공정 조건 변화 전후로 불량 패턴이 뒤바뀜",
         "evidence": f"idx 1891 전후 불량-양품 차이의 부호 일치 {agree}/{len(rg)}, Spearman ρ={rho_rg:.2f}",
         "source": "p2_regime_feature_shift.csv"},
        {"no": 5, "cause": "불량 표본이 너무 적음",
         "evidence": f"불량 {n1}건으로 검출 가능한 최소 |d|={mde(n1, n0, alpha_b):.2f} (보정) vs 관측 최대 |d|={dmax:.2f}"
                     + (f"; 관측 효과 검출에 불량 약 {need}건 필요" if need else ""),
         "source": "p3_mde_curve.csv, p3_defect_effect_sizes.csv"},
        {"no": 6, "cause": "측정 해상도가 낮음",
         "evidence": f"라벨 데이터 변수별 고유값 {int(nun.min())}~{int(nun.max())}개 (중앙값 {int(nun.median())}); "
                     f"591샷 중 대부분이 같은 값 조합을 공유",
         "source": "p1_basic_stats.csv"},
        {"no": 7, "cause": "모델이 쓰는 변수도 우연 수준",
         "evidence": f"SHAP 중요도가 라벨 순열 모델보다 유의하게 큰 변수 {n_sig}/{len(imp)}개; "
                     f"50개 fold 모델 간 중요도 순위 일관성 ρ={stability:.2f}",
         "source": "p3_shap_importance.csv"},
    ])
    causes["cn7_comparison"] = "CN7 데이터/결과 수령 후 같은 지표 산출 예정"
    save(causes, "p3_rg3_failure_causes.csv")
    return causes, dmax, need


# ================================================================ 3. FN / FP 구간화
def error_bins(d, feats, y, res):
    n = len(y)
    reps = {r: {"flag_F1max": np.zeros(n), "flag_Recall80": np.zeros(n), "p": np.zeros(n)} for r in range(N_REPEATS)}
    for fold, te, p, thr, _ in res:
        r = fold // N_SPLITS
        reps[r]["p"][te] = p
        for rule in ("F1max", "Recall80"):
            reps[r][f"flag_{rule}"][te] = p >= thr[rule]
    pred = pd.DataFrame({IDX: d[IDX], LABEL: y,
                         "p_mean": np.mean([reps[r]["p"] for r in reps], 0)})
    for rule in ("F1max", "Recall80"):
        f = np.mean([reps[r][f"flag_{rule}"] for r in reps], 0)
        pred[f"flag_rate_{rule}"] = f
        pred[f"FP_{rule}"] = (y == 0) & (f >= 0.5)
        pred[f"FN_{rule}"] = (y == 1) & (f < 0.5)
    save(pred, "p3_oof_predictions.csv")

    live = [c for c in feats if d[c].nunique() > 1]
    cond = d[live].copy()
    cond["Regime(1891 이후)"] = (d[IDX] >= REGIME_START_IDX).astype(int)
    rows, tests = [], []
    for c in cond.columns:
        v = cond[c]
        bins = v.astype(str) if v.nunique() <= 4 else pd.qcut(v.rank(method="first"), 4, labels=False).astype(str)
        if v.nunique() > 4:
            edges = v.groupby(bins).agg(["min", "max"])
            lab = {b: f"Q{int(b) + 1} [{edges.loc[b, 'min']:.2f}, {edges.loc[b, 'max']:.2f}]" for b in edges.index}
        else:
            lab = {b: f"= {float(b):.2f}" for b in bins.unique()}
        for rule in ("F1max", "Recall80"):
            fp_tab, fn_tab = [], []
            for b in sorted(bins.unique(), key=lambda s: float(s)):
                m = (bins == b).to_numpy()
                nn0, nn1 = int((m & (y == 0)).sum()), int((m & (y == 1)).sum())
                fp = int(pred[f"FP_{rule}"][m].sum()); fn = int(pred[f"FN_{rule}"][m].sum())
                rows.append({"feature": c, "bin": lab[b], "rule": rule, "n_normal": nn0,
                             "fp": fp, "fp_rate": fp / nn0 if nn0 else np.nan,
                             "n_defect": nn1, "fn": fn, "fn_rate": fn / nn1 if nn1 else np.nan})
                fp_tab.append([fp, nn0 - fp]); fn_tab.append([fn, nn1 - fn])
            fp_tab, fn_tab = np.array(fp_tab), np.array(fn_tab)
            p_fp = stats.chi2_contingency(fp_tab)[1] if (fp_tab.sum(0) > 0).all() and len(fp_tab) > 1 else np.nan
            fn_ok = fn_tab[fn_tab.sum(1) > 0]
            p_fn = stats.chi2_contingency(fn_ok)[1] if len(fn_ok) > 1 and (fn_ok.sum(0) > 0).all() else np.nan
            tests.append({"feature": c, "rule": rule, "p_fp_heterogeneity": p_fp, "p_fn_heterogeneity": p_fn})
    eb = pd.DataFrame(rows)
    tt = pd.DataFrame(tests)
    k = cond.shape[1]
    tt["p_fp_bonferroni"] = np.minimum(tt.p_fp_heterogeneity * k, 1.0)
    tt["p_fn_bonferroni"] = np.minimum(tt.p_fn_heterogeneity * k, 1.0)
    save(eb, "p3_error_by_bin.csv")
    save(tt, "p3_error_bin_tests.csv")

    # 그림: F1 최대 임계값 기준 FP율 / FN율 (변수 × 구간)
    rule = "F1max"
    e = eb[eb.rule == rule].copy()
    e["col"] = e.groupby("feature").cumcount()
    feats_order = cond.columns.tolist()
    FP = np.full((len(feats_order), 4), np.nan); FN = FP.copy(); ND = FP.copy()
    for r in e.itertuples():
        i = feats_order.index(r.feature)
        FP[i, r.col], FN[i, r.col], ND[i, r.col] = r.fp_rate, r.fn_rate, r.n_defect
    t = tt[tt.rule == rule].set_index("feature")
    fig, axes = plt.subplots(1, 2, figsize=(12, 9), sharey=True)
    for ax, M, ttl, key in ((axes[0], FP, f"FP율 (정상 중 경보) — 전체 {pred[f'FP_{rule}'].sum() / (y == 0).sum():.0%}", "p_fp_bonferroni"),
                            (axes[1], FN, f"FN율 (불량 중 놓침) — 전체 {pred[f'FN_{rule}'].sum() / y.sum():.0%}", "p_fn_bonferroni")):
        im = ax.imshow(M, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    txt = f"{M[i, j]:.2f}" if ax is axes[0] else f"{M[i, j]:.2f}\n(n={int(ND[i, j])})"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6 if ax is axes[1] else 6.5,
                            color="white" if M[i, j] > 0.6 else "#0b0b0b")
        ax.set_xticks(range(4), ["구간1 (낮음)", "구간2", "구간3", "구간4 (높음)"], fontsize=8)
        ax.set_title(ttl, loc="left", fontsize=9.5); ax.grid(False)
        sig = [f for f in feats_order if t.loc[f, key] < 0.05]
        ax.set_xlabel("보정 후 구간 간 차이 유의: " + (", ".join(sig) if sig else "없음"), fontsize=8)
    axes[0].set_yticks(range(len(feats_order)), feats_order, fontsize=7.5)
    fig.colorbar(im, ax=axes, shrink=0.6).set_label("비율", fontsize=8)
    fig.suptitle("공정 조건 구간별 오류 집중도 (HGB, F1 최대 임계값, 평가 fold 예측 10회 반복 중 과반 기준; "
                 "구간 = 사분위 또는 고유값)", x=0.01, ha="left", fontsize=10)
    fig.savefig(FIG / "p3_error_by_bin.png"); plt.close(fig)
    return pred, eb, tt


def main():
    raw, _, feats = load_raw()
    d = merge_pairs(raw, feats)
    X, y = d[feats].to_numpy(), d[LABEL].to_numpy()
    splits = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED).split(X, y))
    res = Parallel(n_jobs=-1)(delayed(run_fold)(X, y, tr, te, f) for f, (tr, te) in enumerate(splits))

    imp, inter, stability, n_sig = shap_analysis(d, X, y, feats, res)
    causes, dmax, need = failure_causes(d, feats, y, imp, n_sig, stability)
    pred, eb, tt = error_bins(d, feats, y, res)

    pd.set_option("display.width", 250)
    print("[SHAP 중요도]\n", imp[["feature", "shap_mean_abs", "null_mean", "null_p95", "ratio_to_null", "perm_p",
                                "perm_p_bonferroni", "fold_rank_mean", "fold_rank_std"]].round(4).to_string(index=False))
    print(f"\nfold 간 순위 일관성 ρ={stability:.3f}, 보정 후 유의 {n_sig}개")
    print("\n[상호작용 상위]\n", inter.head(8).round(4).to_string(index=False))
    print("\n[RG3 실패 원인]\n", causes[["no", "cause", "evidence"]].to_string(index=False))
    print(f"\n관측 최대 |d|={dmax:.3f}, 필요 불량 수={need}")
    for rule in ("F1max", "Recall80"):
        print(f"\n[{rule}] FP {int(pred[f'FP_{rule}'].sum())}/{int((y == 0).sum())}, FN {int(pred[f'FN_{rule}'].sum())}/{int(y.sum())}")
    print("\n[구간 이질성 검정 — 보정 p<0.1]\n",
          tt[(tt.p_fp_bonferroni < 0.1) | (tt.p_fn_bonferroni < 0.1)].round(4).to_string(index=False))
    e = eb[eb.rule == "F1max"]
    print("\n[FP율 상위 구간]\n", e.sort_values("fp_rate", ascending=False).head(6).round(3).to_string(index=False))
    print("\n[FN율 상위 구간 (불량 3건 이상)]\n",
          e[e.n_defect >= 3].sort_values("fn_rate", ascending=False).head(6).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
