"""Phase 1. 데이터 진단 (RG3).

실행: python src/phase1_diagnosis.py
산출물: outputs/figures/p1_*.png, outputs/tables/p1_*.csv
"""
from pathlib import Path
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
from preprocess import DATA  # noqa: E402  (로컬 data/ 또는 ../①Molding/)
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

LABEL = "PassOrFail"
IDX = "idx"  # 원본 'Unnamed: 0'

# 차트 스타일: 라벨=blue, 비라벨=orange, 불량=red, 양품=gray
C_L, C_U, C_FAIL, C_PASS = "#2a78d6", "#eb6834", "#e34948", "#9a9892"
plt.rcParams.update({
    "font.family": "Malgun Gothic",
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6,
    "axes.edgecolor": "#8a8984",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})


def load():
    lab = pd.read_csv(DATA / "moldset_labeled_rg3.csv").rename(columns={"Unnamed: 0": IDX})
    unl = pd.read_csv(DATA / "moldset_unlabeled_rg3.csv").rename(columns={"Unnamed: 0": IDX})
    feats = [c for c in lab.columns if c not in (IDX, LABEL)]
    assert feats == [c for c in unl.columns if c != IDX], "라벨/비라벨 특징 컬럼 불일치"
    return lab.sort_values(IDX).reset_index(drop=True), unl.sort_values(IDX).reset_index(drop=True), feats


def min_step(s):
    """고유값 사이 최소 간격 = 원 단위 측정 해상도 / 표준편차."""
    d = np.diff(np.sort(s.unique()))
    d = d[d > 1e-9]
    return d.min() if len(d) else np.nan


# ---------------------------------------------------------------- 1. 기초 통계
def basic_stats(lab, unl, feats):
    rows = []
    for name, df in (("labeled", lab), ("unlabeled", unl)):
        for c in feats:
            s = df[c]
            top_ratio = s.value_counts(normalize=True).iloc[0]
            rows.append({
                "dataset": name, "feature": c, "n": len(s), "missing": int(s.isna().sum()),
                "mean": s.mean(), "std": s.std(), "min": s.min(), "q25": s.quantile(.25),
                "median": s.median(), "q75": s.quantile(.75), "max": s.max(),
                "nunique": s.nunique(), "top_value_ratio": top_ratio,
                "constant": s.nunique() <= 1,
                "quasi_constant(top>=0.95)": top_ratio >= 0.95,
            })
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "p1_basic_stats.csv", index=False, encoding="utf-8-sig")

    n_fail = int(lab[LABEL].sum())
    bal = pd.DataFrame([{
        "product": "RG3", "n_rows": len(lab), "n_fail": n_fail, "n_pass": len(lab) - n_fail,
        "fail_rate": n_fail / len(lab),
        "n_unique_feature_rows": len(lab[feats].drop_duplicates()),
        "n_unlabeled_rows": len(unl),
    }])
    bal.to_csv(TAB / "p1_class_balance.csv", index=False, encoding="utf-8-sig")
    return out, bal


# ---------------------------------------------------------------- 2. 중복 구조
def duplicate_structure(lab, unl, feats):
    lab = lab.copy()
    lab["gid"] = lab.groupby(feats, sort=False).ngroup()
    grp = lab.groupby("gid").agg(
        size=(IDX, "size"), idx_list=(IDX, lambda s: ",".join(map(str, s))),
        idx_span=(IDX, lambda s: s.max() - s.min()),
        n_fail=(LABEL, "sum"), label_min=(LABEL, "min"), label_max=(LABEL, "max"))
    grp["label_conflict"] = grp.label_min != grp.label_max
    grp.drop(columns=["label_min", "label_max"]).to_csv(TAB / "p1_duplicate_groups.csv", encoding="utf-8-sig")

    twins = []
    for _, r in lab[lab[LABEL] == 1].iterrows():
        other = lab[(lab.gid == r.gid) & (lab[IDX] != r[IDX])]
        for _, o in other.iterrows():
            twins.append({"fail_idx": r[IDX], "twin_idx": o[IDX], "twin_label": o[LABEL],
                          "idx_distance": o[IDX] - r[IDX],
                          "fail_position_in_pair": "first" if r[IDX] < o[IDX] else "second"})
    twins = pd.DataFrame(twins)
    twins.to_csv(TAB / "p1_defect_twins.csv", index=False, encoding="utf-8-sig")

    u_dup = unl.duplicated(feats, keep=False)
    summ = pd.DataFrame([
        {"item": "라벨 행 수", "value": len(lab)},
        {"item": "라벨 고유 특징행 수", "value": grp.shape[0]},
        {"item": "라벨 중복에 속한 행 수", "value": int((grp["size"] > 1).mul(grp["size"]).sum())},
        {"item": "라벨 그룹 크기 분포", "value": str(grp["size"].value_counts().to_dict())},
        {"item": "라벨 그룹 내 idx 간격 분포", "value": str(grp.idx_span.value_counts().sort_index().to_dict())},
        {"item": "라벨 충돌 그룹 수 (같은 특징, 다른 라벨)", "value": int(grp.label_conflict.sum())},
        {"item": "불량행 중 쌍둥이가 양품인 비율", "value": f"{(twins.twin_label == 0).mean():.3f}" if len(twins) else "NA"},
        {"item": "불량행이 쌍의 앞쪽인 건수", "value": int((twins.fail_position_in_pair == 'first').sum())},
        {"item": "비라벨 중복에 속한 행 수", "value": int(u_dup.sum())},
        {"item": "비라벨 고유 특징행 수", "value": len(unl[feats].drop_duplicates())},
        {"item": "라벨-비라벨 특징값 완전일치 행 수",
         "value": len(lab[feats].drop_duplicates().merge(unl[feats].drop_duplicates(), on=feats))},
    ])
    summ.to_csv(TAB / "p1_duplicate_summary.csv", index=False, encoding="utf-8-sig")
    return lab, grp, twins, summ


def leak_check(lab, feats):
    """중복 처리 방식에 따른 성능 차이(참고용 진단, 모델 선정 아님)."""
    X, y = lab[feats].values, lab[LABEL].values
    dedup = lab.groupby("gid").agg({**{c: "first" for c in feats}, LABEL: "max"})
    Xd, yd = dedup[feats].values, dedup[LABEL].values
    rows = []
    for name, XX, yy in (("원본 행 그대로 (쌍둥이 train/test 분리 가능)", X, y),
                         ("쌍 병합, 라벨=max (591행)", Xd, yd)):
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
        auc, ap = [], []
        for tr, te in cv.split(XX, yy):
            m = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                       random_state=SEED, n_jobs=-1).fit(XX[tr], yy[tr])
            p = m.predict_proba(XX[te])[:, 1]
            auc.append(roc_auc_score(yy[te], p))
            ap.append(average_precision_score(yy[te], p))
        rows.append({"setting": name, "n": len(yy), "n_fail": int(yy.sum()),
                     "roc_auc_mean": np.mean(auc), "roc_auc_std": np.std(auc),
                     "prauc_mean": np.mean(ap), "prauc_std": np.std(ap),
                     "prauc_baseline(=fail_rate)": yy.mean()})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "p1_duplicate_leak_check.csv", index=False, encoding="utf-8-sig")
    return out


# ---------------------------------------------------------------- 3. 시간 순서
def time_order(lab, unl, feats):
    rng = np.random.default_rng(SEED)
    rows = []
    # (a) 'Unnamed: 0'이 시간 순서인지: 고유행 기준 특징의 lag-1 자기상관 vs 순열
    uniq = lab.drop_duplicates("gid").sort_values(IDX)
    for c in feats:
        s = uniq[c].values
        if np.std(s) == 0:
            continue
        ac = np.corrcoef(s[:-1], s[1:])[0, 1]
        perm = [np.corrcoef(p[:-1], p[1:])[0, 1] for p in (rng.permutation(s) for _ in range(500))]
        rows.append({"test": "lag1_autocorr", "feature": c, "stat": ac,
                     "perm_p": (np.sum(np.abs(perm) >= abs(ac)) + 1) / 501})
    ac_tab = pd.DataFrame(rows)

    # (b) 불량이 특정 구간에 몰리는지: 윈도우 최대 불량수(scan) & 5구간 카이제곱
    y = lab[LABEL].values
    w = 100
    scan = lambda v: np.convolve(v, np.ones(w, int), "valid").max()
    obs = scan(y)
    perm = np.array([scan(rng.permutation(y)) for _ in range(2000)])
    seg = pd.qcut(np.arange(len(y)), 5, labels=False)
    seg_counts = pd.Series(y).groupby(seg).sum().values
    chi = stats.chisquare(seg_counts)
    gaps = np.diff(lab.loc[y == 1, IDX].values)
    cluster = pd.DataFrame([
        {"test": f"scan_max_fail_in_{w}rows", "feature": "PassOrFail", "stat": obs,
         "perm_p": (np.sum(perm >= obs) + 1) / 2001},
        {"test": "chisq_5_segments", "feature": "PassOrFail", "stat": chi.statistic, "perm_p": chi.pvalue},
        {"test": "segment_fail_counts", "feature": "PassOrFail", "stat": str(seg_counts.tolist()), "perm_p": np.nan},
        {"test": "fail_gap_min/median/max", "feature": "PassOrFail",
         "stat": f"{gaps.min()}/{np.median(gaps):.0f}/{gaps.max()}", "perm_p": np.nan},
    ])
    out = pd.concat([ac_tab, cluster], ignore_index=True)
    out.to_csv(TAB / "p1_time_order_tests.csv", index=False, encoding="utf-8-sig")

    # 비라벨 idx 간격
    ug = np.diff(unl[IDX].values)
    ugap = pd.DataFrame([{"n": len(unl), "idx_min": unl[IDX].min(), "idx_max": unl[IDX].max(),
                          "gap_median": np.median(ug), "gap_p99": np.percentile(ug, 99), "gap_max": ug.max(),
                          "n_gaps_over_1000": int((ug > 1000).sum())}])
    ugap.to_csv(TAB / "p1_unlabeled_index_gaps.csv", index=False, encoding="utf-8-sig")

    # 그림 1: 불량 타임라인
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    ax = axes[0]
    ax.vlines(lab.loc[y == 1, IDX], 0, 1, color=C_FAIL, lw=1.5)
    ax.set_yticks([]); ax.set_title("불량 발생 위치 ('Unnamed: 0' 순서, 빨간 선 = 불량 1건)", loc="left")
    ax.grid(False)
    ax = axes[1]
    roll = pd.Series(y).rolling(w, center=True, min_periods=w).mean() * 100
    ax.plot(lab[IDX], roll, color=C_L, lw=2)
    ax.axhline(y.mean() * 100, color=C_PASS, lw=1, ls="--")
    ax.text(lab[IDX].max(), y.mean() * 100, f" 전체 {y.mean()*100:.1f}%", va="bottom", ha="right",
            fontsize=8, color="#52514e")
    ax.set_ylabel(f"이동 불량률 (%, 창 {w}행)")
    ax.set_xlabel("원본 인덱스 (Unnamed: 0)")
    ax.set_title(f"구간별 불량률 — scan 검정 p={cluster.perm_p.iloc[0]:.3f}, 5구간 χ² p={chi.pvalue:.3f}", loc="left")
    fig.savefig(FIG / "p1_defect_timeline.png"); plt.close(fig)

    # 그림 2: 주요 특징의 시간 추이 + 불량 위치
    top = ac_tab.sort_values("stat", ascending=False).feature.head(8).tolist()
    fig, axes = plt.subplots(len(top), 1, figsize=(11, 1.35 * len(top)), sharex=True)
    for ax, c in zip(axes, top):
        ax.plot(lab[IDX], lab[c], color=C_L, lw=1)
        f = lab[y == 1]
        ax.scatter(f[IDX], f[c], color=C_FAIL, s=14, zorder=3, edgecolor="white", linewidth=0.8)
        ax.set_ylabel(c, rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_title("자기상관 상위 8개 특징의 인덱스 순 추이 (빨간 점 = 불량)", loc="left")
    axes[-1].set_xlabel("원본 인덱스 (Unnamed: 0)")
    fig.savefig(FIG / "p1_features_over_index.png"); plt.close(fig)
    return out, ugap


# ---------------------------------------------------------------- 4. 라벨 vs 비라벨 스케일
def scale_compare(lab, unl, feats):
    rows = []
    for c in feats:
        sl, su = min_step(lab[c]), min_step(unl[c])
        ks = stats.ks_2samp(lab[c], unl[c])
        rows.append({"feature": c,
                     "mean_L": lab[c].mean(), "std_L": lab[c].std(), "mean_U": unl[c].mean(), "std_U": unl[c].std(),
                     "min_L": lab[c].min(), "max_L": lab[c].max(), "min_U": unl[c].min(), "max_U": unl[c].max(),
                     "nunique_L": lab[c].nunique(), "nunique_U": unl[c].nunique(),
                     "min_step_L": sl, "min_step_U": su,
                     "step_ratio_L/U(=원단위 std_U/std_L 추정)": sl / su if su else np.nan,
                     "ks_stat": ks.statistic, "ks_p": ks.pvalue})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "p1_scale_comparison.csv", index=False, encoding="utf-8-sig")

    # 그림 3: 변수별 분포 겹쳐보기
    n = len(feats); ncol = 4; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 2.1 * nrow))
    for ax, c in zip(axes.flat, feats):
        lo = np.percentile(np.r_[lab[c], unl[c]], 0.5); hi = np.percentile(np.r_[lab[c], unl[c]], 99.5)
        if hi <= lo:
            lo, hi = lo - 1, hi + 1
        bins = np.linspace(lo, hi, 40)
        ax.hist(unl[c].clip(lo, hi), bins=bins, density=True, color=C_U, alpha=0.55, label="비라벨")
        ax.hist(lab[c].clip(lo, hi), bins=bins, density=True, color=C_L, alpha=0.55, label="라벨")
        ax.set_title(c, fontsize=8.5, loc="left"); ax.set_yticks([]); ax.tick_params(labelsize=7)
    for ax in axes.flat[n:]:
        ax.axis("off")
    axes.flat[0].legend(loc="upper right")
    fig.suptitle("라벨 vs 비라벨 분포 (각 파일에 저장된 z값 그대로, 0.5–99.5% 구간)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "p1_dist_labeled_vs_unlabeled.png"); plt.close(fig)

    # 그림 4: 최소 간격 비율 (log)
    o = out.dropna(subset=["step_ratio_L/U(=원단위 std_U/std_L 추정)"]).sort_values(
        "step_ratio_L/U(=원단위 std_U/std_L 추정)")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(o.feature, o["step_ratio_L/U(=원단위 std_U/std_L 추정)"], color=C_L, height=0.7)
    ax.axvline(1, color="#52514e", lw=1)
    ax.set_xscale("log"); ax.grid(axis="y", visible=False)
    for i, v in enumerate(o["step_ratio_L/U(=원단위 std_U/std_L 추정)"]):
        ax.text(v * 1.08, i, f"{v:,.0f}×", va="center", fontsize=7.5, color="#52514e")
    ax.set_xlabel("라벨 z값 최소간격 ÷ 비라벨 z값 최소간격 (log)\n= 비라벨 원단위 std ÷ 라벨 원단위 std (추정)")
    ax.set_title("같은 스케일이면 1이어야 한다 — 모든 변수에서 약 10배~24,000배 차이", loc="left")
    fig.savefig(FIG / "p1_scale_step_ratio.png"); plt.close(fig)
    return out


def main():
    lab, unl, feats = load()
    print(f"labeled {lab.shape}, unlabeled {unl.shape}, features {len(feats)}")

    st, bal = basic_stats(lab, unl, feats)
    print("\n[1] 불균형\n", bal.to_string(index=False))
    cq = st[st["constant"] | st["quasi_constant(top>=0.95)"]][["dataset", "feature", "nunique", "top_value_ratio"]]
    print("상수/준상수\n", cq.to_string(index=False))
    print("라벨 nunique\n", st[st.dataset == "labeled"].set_index("feature")["nunique"].to_dict())

    lab, grp, twins, summ = duplicate_structure(lab, unl, feats)
    print("\n[2] 중복\n", summ.to_string(index=False))
    print(leak_check(lab, feats).to_string(index=False))

    to, ugap = time_order(lab, unl, feats)
    print("\n[3] 시간 순서\n", to.to_string(index=False))
    print(ugap.to_string(index=False))

    sc = scale_compare(lab, unl, feats)
    print("\n[4] 스케일\n", sc[["feature", "nunique_L", "nunique_U", "min_L", "max_L", "min_U", "max_U",
                               "step_ratio_L/U(=원단위 std_U/std_L 추정)", "ks_stat"]].to_string(index=False))

    print("\n[5] CN7 vs RG3: CN7 데이터가 data/에 없어 건너뜀")


if __name__ == "__main__":
    main()
