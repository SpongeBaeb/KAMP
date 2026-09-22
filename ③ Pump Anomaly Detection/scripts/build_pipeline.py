"""
프레스 유압펌프 진동/전류 시계열 기반 이상 조기탐지 및 오경보 분석
--------------------------------------------------------------
개발/검증용 스크립트. 이 파일의 로직을 그대로 pump.ipynb 셀로 옮긴다.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("D:/Presspump/data")

# ------------------------------------------------------------------
# 1. 데이터 로드
# ------------------------------------------------------------------
normal = pd.read_csv(DATA_DIR / "press_data_normal.csv", index_col=0, parse_dates=["TimeStamp"])
outlier = pd.read_csv(DATA_DIR / "outlier_data.csv", index_col=0, parse_dates=["TimeStamp"])

print(normal.shape, outlier.shape)
print(normal.isna().sum())
print(outlier.isna().sum())
print("dup normal:", normal.duplicated().sum(), "dup outlier:", outlier.duplicated().sum())

# 완전 중복행 제거 (정상데이터 1건)
normal = normal.drop_duplicates().reset_index(drop=True)

dt_normal = normal["TimeStamp"].diff().dt.total_seconds().dropna()
dt_outlier = outlier["TimeStamp"].diff().dt.total_seconds().dropna()
print("normal sampling interval (median/std):", dt_normal.median(), dt_normal.std())
print("outlier sampling interval (median/std):", dt_outlier.median(), dt_outlier.std())
# 중요 진단: 샘플링 간격이 대부분 0.1s이지만 일부 구간에서 최대 16초 이상 gap 발생.
# -> 프레스가 스트로크(사이클) 단위로 동작하며, 사이클 사이 유휴시간에는 로깅이 끊기는
#    "버스트(burst) 수집" 구조로 판단됨. 따라서 윈도우 특징추출 시 gap을 넘어서는
#    윈도우가 만들어지지 않도록 세그먼트 단위로 분리해야 함 (아래 GAP_SEC 기준).
GAP_SEC = 0.5
normal["segment_id"] = (dt_normal.reindex(normal.index).fillna(0) > GAP_SEC).cumsum()
outlier["segment_id"] = (dt_outlier.reindex(outlier.index).fillna(0) > GAP_SEC).cumsum()
print("normal segments:", normal["segment_id"].nunique(), "outlier segments:", outlier["segment_id"].nunique())

# ------------------------------------------------------------------
# 2. 데이터 분할 전략
#    normal 20,000행을 시간순으로 3분할
#    - train (60%) : 정상 패턴 학습
#    - calib (20%) : 임계값 보정 (validation)
#    - fa_test(20%) : 오경보율(false alarm rate) 측정용 held-out 정상 구간
#    outlier 600행 전체는 탐지율(recall) 측정용 held-out 이상 구간
# ------------------------------------------------------------------
n = len(normal)
i_train = int(n * 0.6)
i_calib = int(n * 0.8)

train_df = normal.iloc[:i_train].reset_index(drop=True)
calib_df = normal.iloc[i_train:i_calib].reset_index(drop=True)
fa_test_df = normal.iloc[i_calib:].reset_index(drop=True)
outlier_df = outlier.reset_index(drop=True)

print({k: len(v) for k, v in dict(train=train_df, calib=calib_df, fa_test=fa_test_df, outlier=outlier_df).items()})

# ------------------------------------------------------------------
# 3. 윈도우 기반 특징추출
# ------------------------------------------------------------------
SIGNAL_COLS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
WIN = 20       # 20 샘플 = 약 2초 (10Hz 가정)
STRIDE = 10    # 50% overlap


def extract_window_features(df: pd.DataFrame, win: int = WIN, stride: int = STRIDE) -> pd.DataFrame:
    """세그먼트(gap-free 연속구간) 단위로만 윈도우를 생성한다.
    프레스 사이클 사이의 유휴 gap을 넘어 윈도우가 만들어지면
    실제로는 연속되지 않은 신호를 하나의 윈도우로 취급하게 되어
    RMS/FFT 등 시계열 특징이 왜곡되므로 segment_id 경계를 넘지 않도록 한다.
    """
    rows = []
    for _, seg_df in df.groupby("segment_id"):
        rows.extend(_extract_from_segment(seg_df, win, stride))
    return pd.DataFrame(rows)


def _extract_from_segment(df: pd.DataFrame, win: int, stride: int) -> list:
    rows = []
    values = df[SIGNAL_COLS].to_numpy()
    ts = df["TimeStamp"].to_numpy()
    n_rows = len(df)
    for start in range(0, n_rows - win + 1, stride):
        end = start + win
        seg = values[start:end]
        feat = {"t_start": ts[start], "t_end": ts[end - 1]}
        for j, col in enumerate(SIGNAL_COLS):
            x = seg[:, j].astype(float)
            rms = np.sqrt(np.mean(x ** 2))
            peak = np.max(np.abs(x))
            std = np.std(x)
            mean = np.mean(x)
            crest = peak / (rms + 1e-9)
            # 4th/3rd moment 기반 첨도/왜도 (스케일 안정화를 위해 std로 정규화)
            if std > 1e-9:
                skew = np.mean(((x - mean) / std) ** 3)
                kurt = np.mean(((x - mean) / std) ** 4) - 3
            else:
                skew, kurt = 0.0, 0.0
            # 주파수 영역: FFT 지배 성분 에너지 비중
            spec = np.abs(np.fft.rfft(x - mean))
            dom_ratio = (spec.max() / (spec.sum() + 1e-9)) if spec.sum() > 0 else 0.0
            feat[f"{col}_rms"] = rms
            feat[f"{col}_peak"] = peak
            feat[f"{col}_std"] = std
            feat[f"{col}_crest"] = crest
            feat[f"{col}_skew"] = skew
            feat[f"{col}_kurt"] = kurt
            feat[f"{col}_domratio"] = dom_ratio
        # 채널 간 관계
        v0, v1, cur = seg[:, 0], seg[:, 1], seg[:, 2]
        feat["corr_v0_v1"] = np.corrcoef(v0, v1)[0, 1] if np.std(v0) > 1e-9 and np.std(v1) > 1e-9 else 0.0
        rows.append(feat)
    return rows


feat_train = extract_window_features(train_df)
feat_calib = extract_window_features(calib_df)
feat_fa_test = extract_window_features(fa_test_df)
feat_outlier = extract_window_features(outlier_df)

FEATURE_COLS = [c for c in feat_train.columns if c not in ("t_start", "t_end")]
print("feature windows:", len(feat_train), len(feat_calib), len(feat_fa_test), len(feat_outlier))

# ------------------------------------------------------------------
# 4. 베이스라인: 3-sigma 관리도 (RMS 기반)
# ------------------------------------------------------------------
mu = feat_train[FEATURE_COLS].mean()
sigma = feat_train[FEATURE_COLS].std()


def baseline_score(feat_df: pd.DataFrame) -> np.ndarray:
    z = (feat_df[FEATURE_COLS] - mu) / (sigma + 1e-9)
    return z.abs().max(axis=1).to_numpy()  # 각 윈도우에서 가장 크게 벗어난 z-score


base_calib_score = baseline_score(feat_calib)
base_threshold = np.percentile(base_calib_score, 99)  # calib 정상구간 기준 상위 1%를 임계값으로

base_fa_score = baseline_score(feat_fa_test)
base_out_score = baseline_score(feat_outlier)

print("baseline threshold(z):", base_threshold)
print("baseline FA rate:", (base_fa_score > base_threshold).mean())
print("baseline detection rate:", (base_out_score > base_threshold).mean())

# ------------------------------------------------------------------
# 5. 모델 A: Isolation Forest
# ------------------------------------------------------------------
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(feat_train[FEATURE_COLS])
X_train = scaler.transform(feat_train[FEATURE_COLS])
X_calib = scaler.transform(feat_calib[FEATURE_COLS])
X_fa = scaler.transform(feat_fa_test[FEATURE_COLS])
X_out = scaler.transform(feat_outlier[FEATURE_COLS])

iso = IsolationForest(n_estimators=300, contamination="auto", random_state=42)
iso.fit(X_train)

iso_calib_score = -iso.score_samples(X_calib)  # 클수록 이상
iso_threshold = np.percentile(iso_calib_score, 99)

iso_fa_score = -iso.score_samples(X_fa)
iso_out_score = -iso.score_samples(X_out)

print("IF FA rate:", (iso_fa_score > iso_threshold).mean())
print("IF detection rate:", (iso_out_score > iso_threshold).mean())

# ------------------------------------------------------------------
# 6. 모델 B: 1D-Conv Autoencoder (raw multivariate window 재구성 오차)
# ------------------------------------------------------------------
import torch
import torch.nn as nn

torch.manual_seed(42)


def make_raw_windows(df: pd.DataFrame, win=WIN, stride=STRIDE):
    """segment_id 경계를 넘지 않는 raw window만 생성 (gap 구간 오염 방지)."""
    windows = []
    for _, seg_df in df.groupby("segment_id"):
        values = seg_df[SIGNAL_COLS].to_numpy().astype(np.float32)
        for start in range(0, len(values) - win + 1, stride):
            windows.append(values[start:start + win])
    return np.stack(windows)  # (N, win, C)


raw_train = make_raw_windows(train_df)
raw_calib = make_raw_windows(calib_df)
raw_fa = make_raw_windows(fa_test_df)
raw_out = make_raw_windows(outlier_df)

raw_mean = raw_train.reshape(-1, 3).mean(axis=0)
raw_std = raw_train.reshape(-1, 3).std(axis=0) + 1e-6


def norm(x):
    return (x - raw_mean) / raw_std


class ConvAE(nn.Module):
    def __init__(self, channels=3, win=WIN):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 8, 3, padding=1), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Conv1d(8, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, channels, 3, padding=1),
        )

    def forward(self, x):
        z = self.enc(x)
        out = self.dec(z)
        return out


def to_tensor(raw):
    x = norm(raw)
    return torch.tensor(x.transpose(0, 2, 1))  # (N, C, T)


ae = ConvAE()
opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X_tr = to_tensor(raw_train)
n_epochs = 30
batch = 128
for epoch in range(n_epochs):
    perm = torch.randperm(len(X_tr))
    epoch_loss = 0.0
    for i in range(0, len(X_tr), batch):
        idx = perm[i:i + batch]
        xb = X_tr[idx]
        opt.zero_grad()
        out = ae(xb)
        loss = loss_fn(out, xb)
        loss.backward()
        opt.step()
        epoch_loss += loss.item() * len(idx)
    if epoch % 5 == 0 or epoch == n_epochs - 1:
        print(f"epoch {epoch} loss {epoch_loss / len(X_tr):.4f}")


def recon_error(raw):
    ae.eval()
    with torch.no_grad():
        x = to_tensor(raw)
        out = ae(x)
        err = ((out - x) ** 2).mean(dim=(1, 2)).numpy()
    return err


ae_calib_score = recon_error(raw_calib)
ae_threshold = np.percentile(ae_calib_score, 99)
ae_fa_score = recon_error(raw_fa)
ae_out_score = recon_error(raw_out)

print("AE FA rate:", (ae_fa_score > ae_threshold).mean())
print("AE detection rate:", (ae_out_score > ae_threshold).mean())

# ------------------------------------------------------------------
# 7. 모델 비교 (F1 기준)
# ------------------------------------------------------------------
from sklearn.metrics import f1_score, precision_score, recall_score


def eval_model(name, fa_score, out_score, threshold):
    y_true = np.concatenate([np.zeros(len(fa_score)), np.ones(len(out_score))])
    y_score = np.concatenate([fa_score, out_score])
    y_pred = (y_score > threshold).astype(int)
    return {
        "model": name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "fa_rate": (fa_score > threshold).mean(),
        "detection_rate": (out_score > threshold).mean(),
    }


results = [
    eval_model("baseline_3sigma", base_fa_score, base_out_score, base_threshold),
    eval_model("isolation_forest", iso_fa_score, iso_out_score, iso_threshold),
    eval_model("conv_autoencoder", ae_fa_score, ae_out_score, ae_threshold),
]
res_df = pd.DataFrame(results)
print(res_df)
