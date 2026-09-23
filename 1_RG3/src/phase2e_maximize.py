"""Phase 2 추가 실험: PR-AUC 개선을 위한 전방위 탐색 (중첩 CV로 정직하게 평가).

실행: python src/phase2e_maximize.py
산출물: outputs/tables/model_comparison_RG3.csv 에 NestedSearch 행 추가 (팀 공통 형식),
        outputs/tables/p2e_*.csv, outputs/figures/p2e_*.png

A. 샷 단위(쌍 병합, 팀 규칙): 후보 약 20개(모델·하이퍼파라미터·변수 세트)를 매 outer fold의
   학습 데이터 안에서 inner 3-fold PR-AUC로 골라, 고른 후보를 outer 평가 fold로 채점한다.
   = "여러 방법 중 최선을 고르는 절차"의 편향 없는 성능. 후보별 사후 최고값(낙관적)도 함께 기록한다.
B. 기록 단위 실험(결정 ① 예외): 원본 1182행에서 '쌍 안의 앞쪽 기록' 여부를 변수로 쓸 때의 성능.
   쌍 단위로 묶은 분할(StratifiedGroupKFold)로 쌍둥이 누수를 막는다. 메인 결과표에는 넣지 않는다.
"""
from pathlib import Path
import random

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocess import IDX, LABEL, PRODUCT, build_feature_sets, load_raw, merge_pairs
from train import SEED, RedundancyFilter, select_thresholds
from evaluate import RULES, TEAM_COLS, recall_at_k, score, team_table
from viz import ALGO_COLOR, C_FAIL, C_L, C_PASS, C_U, INK2, plt

random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
N_SPLITS, N_REPEATS = 5, 10
N_PERM, PERM_REPEATS = 20, 2
NESTED = "NestedSearch(best-of-candidates)"


def save(df, name):
    df.to_csv(TAB / name, index=False, encoding="utf-8-sig")


def pipe(clf, scale=False):
    steps = [("filter", RedundancyFilter())]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps + [("clf", clf)])


def candidates():
    """(이름, 변수 세트, 모델). 모두 fold 내부에서 적합된다."""
    lr = lambda C, pen="l2": LogisticRegression(C=C, penalty=pen, class_weight="balanced", solver="liblinear",
                                                max_iter=5000, random_state=SEED)
    rf = lambda leaf, **kw: RandomForestClassifier(n_estimators=200, min_samples_leaf=leaf, class_weight="balanced",
                                                   random_state=SEED, n_jobs=1, **kw)
    hgb = lambda depth, lr_, it, leaf=20, l2=0.0: HistGradientBoostingClassifier(
        max_depth=depth, learning_rate=lr_, max_iter=it, min_samples_leaf=leaf, l2_regularization=l2,
        class_weight="balanced", early_stopping=False, random_state=SEED)
    c = [
        ("LogReg L2 C=0.01", "S0", pipe(lr(0.01), True)),
        ("LogReg L2 C=0.1", "S0", pipe(lr(0.1), True)),
        ("LogReg L2 C=1", "S0", pipe(lr(1.0), True)),
        ("LogReg L1 C=0.05", "S0", pipe(lr(0.05, "l1"), True)),
        ("LogReg L1 C=0.3", "S0", pipe(lr(0.3, "l1"), True)),
        ("RF leaf=1", "S0", pipe(rf(1))),
        ("RF leaf=5", "S0", pipe(rf(5))),
        ("RF leaf=20", "S0", pipe(rf(20))),
        ("RF 균형배깅 (subsample 30%)", "S0", pipe(RandomForestClassifier(
            n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", max_samples=0.3,
            random_state=SEED, n_jobs=1))),
        ("ExtraTrees leaf=5", "S0", pipe(ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5,
                                                              class_weight="balanced", random_state=SEED, n_jobs=1))),
        ("HGB d=2 lr=.05 it=100", "S0", pipe(hgb(2, 0.05, 100))),
        ("HGB d=3 lr=.05 it=200", "S0", pipe(hgb(3, 0.05, 200))),
        ("HGB d=2 강한 규제", "S0", pipe(hgb(2, 0.03, 150, leaf=40, l2=1.0))),
        ("SVM RBF", "S0", pipe(SVC(C=1.0, class_weight="balanced", probability=True, random_state=SEED), True)),
        ("kNN k=15", "S0", pipe(KNeighborsClassifier(n_neighbors=15, weights="distance"), True)),
        ("Naive Bayes", "S0", pipe(GaussianNB(), True)),
        ("LogReg L2 C=0.1", "S2", pipe(lr(0.1), True)),
        ("RF leaf=5", "S2", pipe(rf(5))),
        ("HGB d=2 강한 규제", "S2", pipe(hgb(2, 0.03, 150, leaf=40, l2=1.0))),
        ("LogReg L2 C=0.1", "S3", pipe(lr(0.1), True)),
        ("RF leaf=5", "S3", pipe(rf(5))),
        ("HGB d=2 강한 규제", "S3", pipe(hgb(2, 0.03, 150, leaf=40, l2=1.0))),
    ]
    return [(f"{n} [{s}]", s, m) for n, s, m in c]


def inner_oof(model, X, y):
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=SEED).split(X, y):
        oof[va] = clone(model).fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return oof


def run_outer(Xs, y, tr, te, fold, cands, with_candidates=True):
    """한 outer fold: 후보 전부를 학습 fold 안에서 inner CV로 평가 → 최선 선택 → 평가 fold 채점."""
    inner, oofs, cand_rows = [], {}, []
    for name, s, model in cands:
        X = Xs[s]
        oof = inner_oof(model, X[tr], y[tr])
        oofs[name] = oof
        inner.append(average_precision_score(y[tr], oof))
        if with_candidates:  # 사후 비교용 (선택에는 쓰지 않음)
            p = clone(model).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            cand_rows.append({"candidate": name, "fold": fold, "inner_prauc": inner[-1],
                              "prauc": average_precision_score(y[te], p), "roc_auc": roc_auc_score(y[te], p),
                              "recall_at_20": recall_at_k(y[te], p, 0.2)})
    b = int(np.argmax(inner))
    name, s, model = cands[b]
    thr = select_thresholds(y[tr], oofs[name])
    p = clone(model).fit(Xs[s][tr], y[tr]).predict_proba(Xs[s][te])[:, 1]
    nested = {"algo": NESTED, "imbalance": "class_weight", "fold": fold, "selected": name, **score(y[te], p, thr)}
    return nested, cand_rows


def _perm_nested(Xs, y_perm, cands):
    out = []
    for tr, te in RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=PERM_REPEATS,
                                          random_state=SEED).split(Xs["S0"], y_perm):
        out.append(run_outer(Xs, y_perm, tr, te, 0, cands, with_candidates=False)[0])
    return np.mean([o["prauc"] for o in out]), np.mean([o["roc_auc"] for o in out])


# ================================================================ B. 기록 단위 (쌍 안 순서)
def record_level(raw, feats):
    r = raw.copy()
    r["pair"] = r.groupby(feats, sort=False).ngroup()
    r["pair_first"] = (r.groupby("pair")[IDX].transform("min") == r[IDX]).astype(float)
    y, g = r[LABEL].to_numpy(), r["pair"].to_numpy()
    f = r[r[LABEL] == 1]
    n_first = int(f.pair_first.sum())
    binom = stats.binomtest(n_first, len(f), 0.5)
    after = r[IDX] >= 1891
    pos_tab = pd.DataFrame([
        {"item": "불량 기록 수", "value": len(f)},
        {"item": "불량이 쌍의 앞쪽 기록인 수", "value": n_first},
        {"item": "이항검정 p (앞/뒤 반반 가정)", "value": binom.pvalue},
        {"item": "1891 이전: 앞쪽/전체 불량", "value": f"{int(f[~after[f.index]].pair_first.sum())}/{int((~after[f.index]).sum())}"},
        {"item": "1891 이후: 앞쪽/전체 불량", "value": f"{int(f[after[f.index]].pair_first.sum())}/{int(after[f.index].sum())}"},
        {"item": "인접하지 않은 쌍(간격 2~3) 속 불량", "value": int((r[r[LABEL] == 1].groupby('pair')[IDX].first().index.isin(
            r.groupby('pair')[IDX].agg(lambda s: s.max() - s.min()).loc[lambda s: s > 1].index)).sum())},
    ])
    save(pos_tab, "p2e_pair_position_test.csv")

    live = [c for c in feats if r[c].nunique() > 1]
    Xp = r[live].to_numpy()
    Xpp = np.c_[Xp, r.pair_first.to_numpy()]
    lr = lambda: pipe(LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000, random_state=SEED), True)
    hgb = lambda: pipe(HistGradientBoostingClassifier(max_depth=2, learning_rate=0.03, max_iter=150,
                                                      min_samples_leaf=40, l2_regularization=1.0,
                                                      class_weight="balanced", random_state=SEED))
    configs = [("규칙: 앞쪽 기록이면 1", None, None),
               ("LogReg 공정변수만", Xp, lr), ("LogReg 공정변수+앞쪽여부", Xpp, lr),
               ("HGB 공정변수만", Xp, hgb), ("HGB 공정변수+앞쪽여부", Xpp, hgb)]
    rows = []
    for rep in range(N_REPEATS):
        for fold, (tr, te) in enumerate(StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=SEED + rep)
                                        .split(Xp, y, g)):
            assert not set(g[tr]) & set(g[te])
            for name, X, mk in configs:
                p = r.pair_first.to_numpy()[te] if X is None else mk().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                rows.append({"config": name, "fold": rep * N_SPLITS + fold, "prauc": average_precision_score(y[te], p),
                             "roc_auc": roc_auc_score(y[te], p), "recall_at_50": recall_at_k(y[te], p, 0.5),
                             "recall_at_20": recall_at_k(y[te], p, 0.2), "prevalence": y[te].mean()})
    rl = pd.DataFrame(rows)
    summ = rl.drop(columns="fold").groupby("config", sort=False).agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    save(summ, "p2e_record_level.csv")
    return pos_tab, summ, binom


# ================================================================ 그래프
def plot(cand_sum, nested_ext, null, rec, prev_shot, prev_rec):
    # (1) 후보별 PR-AUC(사후) + 중첩 CV 결과
    o = cand_sum.sort_values("prauc_mean")
    fig, ax = plt.subplots(figsize=(9.5, 8))
    y0 = np.arange(len(o))
    ax.errorbar(o.prauc_mean, y0, xerr=o.prauc_std / np.sqrt(N_SPLITS * N_REPEATS), fmt="o", color=C_PASS,
                ms=6, lw=1.2, mec="white", label="후보별 평가 fold 성능 (사후 비교, 오차 = 표준오차)")
    k = len(o) + 0.8
    ax.errorbar(nested_ext.prauc_mean, k, xerr=nested_ext.prauc_std / np.sqrt(N_SPLITS * N_REPEATS), fmt="D",
                color=C_L, ms=8, lw=1.5, mec="white", label="중첩 CV (선택 절차까지 포함한 정직한 성능)")
    ax.axvline(prev_shot, color=INK2, ls="--", lw=1)
    ax.text(prev_shot, -1.4, " 불량률", fontsize=7.5, color=INK2, va="center")
    ax.axvline(null.prauc.mean(), color=C_FAIL, ls=":", lw=1.4)
    ax.text(null.prauc.mean(), k + 0.5, "라벨 순열 기준선(중첩 CV) ", fontsize=7.5, color=C_FAIL, va="center", ha="right")
    ax.set_yticks(list(y0) + [k], list(o.candidate) + [NESTED], fontsize=7.5)
    ax.set_ylim(-2, k + 0.8); ax.grid(axis="y", visible=False)
    ax.set_xlabel("PR-AUC (병합 591행, 5-fold×10 평균)")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.set_title(f"후보 {len(o)}개 전방위 탐색 — 사후 최고값 vs 중첩 CV", loc="left")
    fig.savefig(FIG / "p2e_search_prauc.png"); plt.close(fig)

    # (2) 기록 단위 실험
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    lab = rec.config.tolist()
    for ax, (m, ref, ttl) in zip(axes, (("prauc", prev_rec, "PR-AUC (점선 = 불량률)"),
                                        ("recall_at_50", 0.5, "Recall@상위 50% (점선 = 무작위)"))):
        ax.errorbar(rec[f"{m}_mean"], range(len(lab)), xerr=rec[f"{m}_std"], fmt="o",
                    color=C_U, ms=7, lw=1.5, mec="white")
        ax.axvline(ref, color=INK2, ls="--", lw=1)
        ax.set_title(ttl, loc="left"); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(lab)), lab); axes[0].invert_yaxis()
    fig.suptitle("기록 단위 실험 (원본 1182행, 쌍 단위 묶음 5-fold×10) — 결정 ① 예외, 메인 결과 아님", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(FIG / "p2e_record_level.png"); plt.close(fig)


def main():
    raw, _, feats = load_raw()
    d, sets = build_feature_sets(merge_pairs(raw, feats), feats)
    y = d[LABEL].to_numpy()
    Xs = {k.split()[0]: v.to_numpy() for k, v in sets.items()}
    cands = candidates()
    splits = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                          random_state=SEED).split(Xs["S0"], y))

    res = Parallel(n_jobs=-1)(delayed(run_outer)(Xs, y, tr, te, f, cands) for f, (tr, te) in enumerate(splits))
    nested = pd.DataFrame([r[0] for r in res])
    cand = pd.DataFrame([row for r in res for row in r[1]])
    save(nested, "p2e_nested_fold_metrics.csv")
    save(cand, "p2e_candidate_fold_metrics.csv")
    cand_sum = cand.groupby("candidate", sort=False).agg(
        prauc_mean=("prauc", "mean"), prauc_std=("prauc", "std"), roc_auc_mean=("roc_auc", "mean"),
        recall_at_20_mean=("recall_at_20", "mean"), inner_prauc_mean=("inner_prauc", "mean")).reset_index()
    sel = nested.selected.value_counts().rename("n_selected")
    cand_sum = cand_sum.merge(sel, left_on="candidate", right_index=True, how="left").fillna({"n_selected": 0})
    save(cand_sum.sort_values("prauc_mean", ascending=False), "p2e_candidate_summary.csv")

    num = nested.select_dtypes("number").drop(columns="fold")
    nested_ext = pd.concat([num.mean().add_suffix("_mean"), num.std().add_suffix("_std")]).to_frame().T
    save(nested_ext, "p2e_nested_summary.csv")

    new = team_table(nested.drop(columns="selected"), PRODUCT)
    path = TAB / f"model_comparison_{PRODUCT}.csv"
    old = pd.read_csv(path)
    merged = pd.concat([old[~old.model.str.startswith(NESTED)], new], ignore_index=True)[TEAM_COLS]
    merged.to_csv(path, index=False, encoding="utf-8-sig")

    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(y) for _ in range(N_PERM)]
    null = pd.DataFrame(Parallel(n_jobs=-1)(delayed(_perm_nested)(Xs, yp, cands) for yp in perms),
                        columns=["prauc", "roc_auc"])
    save(null, "p2e_nested_null.csv")
    obs = nested_ext.prauc_mean.iloc[0]
    null_test = pd.DataFrame([{"observed_prauc": obs, "null_prauc_mean": null.prauc.mean(),
                               "null_prauc_std": null.prauc.std(),
                               "p_prauc_greater": (1 + (null.prauc >= obs).sum()) / (N_PERM + 1),
                               "observed_roc_auc": nested_ext.roc_auc_mean.iloc[0],
                               "null_roc_auc_mean": null.roc_auc.mean(),
                               "n_perm": N_PERM, "perm_cv": f"5-fold×{PERM_REPEATS}"}])
    save(null_test, "p2e_nested_null_test.csv")

    pos_tab, rec, binom = record_level(raw, feats)
    plot(cand_sum, nested_ext, null, rec, y.mean(), raw[LABEL].mean())

    pd.set_option("display.width", 250)
    print("[후보별 사후 성능 (선택에 쓰지 않음)]\n",
          cand_sum.sort_values("prauc_mean", ascending=False).round(3).to_string(index=False))
    print("\n[중첩 CV]\n", nested_ext[["prauc_mean", "prauc_std", "roc_auc_mean", "recall_at_20_mean",
                                       "f1_F1max_mean", "precision_F1max_mean", "recall_F1max_mean",
                                       "f1_Recall80_mean", "precision_Recall80_mean",
                                       "recall_Recall80_mean"]].round(3).to_string(index=False))
    print("\n[중첩 CV 순열 검정]\n", null_test.to_string(index=False))
    print("\n[쌍 안 순서]\n", pos_tab.to_string(index=False))
    print("\n[기록 단위 실험]\n", rec.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
