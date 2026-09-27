"""전처리: 데이터 로드, 중복 쌍 병합, 비라벨 데이터 순위 변환."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _data_dir():
    """로컬 data/ 우선, 없으면 팀 저장소(KAMP)의 ①Molding/ 폴더를 쓴다."""
    for p in (ROOT / "data", ROOT.parent / "①Molding"):
        if (p / "moldset_labeled_rg3.csv").exists():
            return p
    raise FileNotFoundError("moldset_labeled_rg3.csv 를 data/ 또는 ../①Molding/ 에 두세요.")


DATA = _data_dir()
LABEL = "PassOrFail"
IDX = "idx"  # 원본 'Unnamed: 0' (시간 순서, 특징으로 쓰지 않음)
PRODUCT = "RG3"


def load_raw():
    lab = pd.read_csv(DATA / "moldset_labeled_rg3.csv").rename(columns={"Unnamed: 0": IDX})
    unl = pd.read_csv(DATA / "moldset_unlabeled_rg3.csv").rename(columns={"Unnamed: 0": IDX})
    feats = [c for c in lab.columns if c not in (IDX, LABEL)]
    assert feats == [c for c in unl.columns if c != IDX], "라벨/비라벨 특징 컬럼 불일치"
    return lab.sort_values(IDX).reset_index(drop=True), unl.sort_values(IDX).reset_index(drop=True), feats


def merge_pairs(lab, feats):
    """특징값이 같은 행(동일 샷의 중복 기록)을 한 행으로 병합한다.

    라벨은 max: 두 기록 중 하나라도 불량이면 불량. idx는 쌍의 앞쪽 인덱스.
    """
    out = (lab.groupby(feats, sort=False)
              .agg(**{IDX: (IDX, "min"), "n_records": (IDX, "size"), LABEL: (LABEL, "max")})
              .reset_index())
    return out.sort_values(IDX).reset_index(drop=True)[[IDX, "n_records", LABEL, *feats]]


def load_merged():
    lab, unl, feats = load_raw()
    return merge_pairs(lab, feats), unl, feats


# ---------------------------------------------------------------- 변수 정리 + 파생변수
# 원 단위를 알 수 없고 모든 값이 변수별 z값이므로, 파생변수는 같은 물리량 묶음 안의 선형 결합(평균·차이·
# 산포)으로만 만든다. 비율은 0 근처 z값으로 나누게 되어 의미가 없으므로 만들지 않는다.
DROP_COLS = ["Clamp_Open_Position", "Injection_Time"]  # 상수 / 준상수(98.6%가 한 값)
PAIR_COLS = ["Mold_Temperature_3", "Mold_Temperature_4", "Max_Back_Pressure", "Average_Back_Pressure"]
BARREL = [f"Barrel_Temperature_{i}" for i in range(1, 7)]
REGIME_START_IDX = 1891  # Phase 2 변화점 탐지 결과


def build_feature_sets(data, feats):
    """라벨을 쓰지 않는 샷 단위 변환. data는 시간(idx) 순으로 정렬된 병합 데이터."""
    d = data.sort_values(IDX).reset_index(drop=True)
    X = d[feats]

    clean = X.drop(columns=DROP_COLS + PAIR_COLS).copy()
    clean["Mold_Temp_mean"] = (X.Mold_Temperature_3 + X.Mold_Temperature_4) / 2
    clean["Mold_Temp_diff"] = X.Mold_Temperature_3 - X.Mold_Temperature_4
    clean["Back_Pressure_mean"] = (X.Max_Back_Pressure + X.Average_Back_Pressure) / 2
    clean["Back_Pressure_spread"] = X.Max_Back_Pressure - X.Average_Back_Pressure

    static = clean.copy()
    static["Screw_RPM_spread"] = X.Max_Screw_RPM - X.Average_Screw_RPM
    static["Inj_minus_Switch_Pressure"] = X.Max_Injection_Pressure - X.Max_Switch_Over_Pressure
    static["Barrel_Temp_mean"] = X[BARREL].mean(axis=1)
    static["Barrel_Temp_zone_std"] = X[BARREL].std(axis=1)
    static["Barrel_Temp_front_rear"] = X[BARREL[:3]].mean(axis=1) - X[BARREL[3:]].mean(axis=1)
    static["Barrel_Temp_max_abs"] = X[BARREL].abs().max(axis=1)
    proc = [c for c in feats if c not in DROP_COLS]
    static["Process_Deviation_Index"] = X[proc].abs().mean(axis=1)

    full = static.copy()
    P = X[proc]
    prev_mean = P.shift(1).rolling(5, min_periods=1).mean()
    full["Change_from_prev"] = P.diff().abs().mean(axis=1)
    full["Volatility_5"] = P.shift(1).rolling(5, min_periods=2).std().mean(axis=1)
    full["Deviation_from_recent5"] = (P - prev_mean).abs().mean(axis=1)
    full["Mold_Temp_mean_d1"] = clean.Mold_Temp_mean.diff()
    full["Barrel_Temp_mean_d1"] = static.Barrel_Temp_mean.diff()
    full["Hopper_Temperature_d1"] = X.Hopper_Temperature.diff()
    full["Cycle_Time_d1"] = X.Cycle_Time.diff()
    full["Regime_after_1891"] = (d[IDX] >= REGIME_START_IDX).astype(float)
    full = full.fillna(0.0)  # 첫 샷(이전 샷 없음)

    sets = {"S0 원본": X.copy(), "S1 정리": clean, "S2 정리+공정 파생": static, "S3 S2+시간 파생": full}
    return d, sets


class RankTransformer:
    """비라벨 배치의 변수별 순위(분위)를 라벨 데이터의 같은 분위 값으로 옮긴다 (quantile mapping).

    가정: 두 데이터의 변수별 주변분포가 같다. Phase 1/2 진단에서 이 가정은 성립하지 않으므로
    결과 해석 시 outputs/tables/p2_unlabeled_rank_evidence.csv의 한계를 함께 명시한다.
    """

    def fit(self, ref, feats):
        self.feats = list(feats)
        self.ref_ = {c: np.sort(ref[c].to_numpy()) for c in self.feats}
        return self

    def transform(self, df):
        out = df.copy()
        for c in self.feats:
            u = (df[c].rank(method="average").to_numpy() - 0.5) / len(df)
            out[c] = np.quantile(self.ref_[c], u)
        return out
