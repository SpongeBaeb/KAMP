"""확장판 프로토타이핑 스크립트: 특징/모델/평가 고도화 검증용."""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import hilbert

DATA_DIR = Path("D:/Presspump/data")
RANDOM_STATE = 42

# ------------------------------------------------------------------
# 1. 로드 + 세그먼트
# ------------------------------------------------------------------
normal = pd.read_csv(DATA_DIR / "press_data_normal.csv", index_col=0, parse_dates=["TimeStamp"])
outlier = pd.read_csv(DATA_DIR / "outlier_data.csv", index_col=0, parse_dates=["TimeStamp"])
normal = normal.drop_duplicates().reset_index(drop=True)
outlier = outlier.reset_index(drop=True)

GAP_SEC = 0.5


def add_segment_id(df):
    dt = df["TimeStamp"].diff().dt.total_seconds()
    df = df.copy()
    df["segment_id"] = (dt.fillna(0) > GAP_SEC).cumsum()
    return df


normal = add_segment_id(normal)
outlier = add_segment_id(outlier)

n = len(normal)
i_train = int(n * 0.6)
i_calib = int(n * 0.8)
train_df = normal.iloc[:i_train].reset_index(drop=True)
calib_df = normal.iloc[i_train:i_calib].reset_index(drop=True)
fa_test_df = normal.iloc[i_calib:].reset_index(drop=True)
outlier_df = outlier.reset_index(drop=True)

# ------------------------------------------------------------------
# 2. 확장 특징 추출
#    - 기존: rms/peak/std/crest/skew/kurt/dom_ratio, corr_v0_v1
#    - 추가: shape/impulse/clearance factor, zero-crossing rate, 1차차분 통계
#            대역 에너지비(저/중/고), spectral entropy, envelope(Hilbert) RMS
#            전류-진동 교차상관
# ------------------------------------------------------------------
SIGNAL_COLS = ["AI0_Vibration", "AI1_Vibration", "AI2_Current"]
WIN = 20
STRIDE = 10


def band_energy_ratios(spec, n_bands=3):
    edges = np.linspace(0, len(spec), n_bands + 1).astype(int)
    total = spec.sum() + 1e-9
    return [spec[edges[i]:edges[i + 1]].sum() / total for i in range(n_bands)]


def spectral_entropy(spec):
    p = spec / (spec.sum() + 1e-9)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _extract_from_segment(df, win, stride):
    rows = []
    values = df[SIGNAL_COLS].to_numpy()
    ts = df["TimeStamp"].to_numpy()
    for start in range(0, len(df) - win + 1, stride):
        end = start + win
        seg = values[start:end]
        feat = {"t_start": ts[start], "t_end": ts[end - 1]}
        chan_x = {}
        for j, col in enumerate(SIGNAL_COLS):
            x = seg[:, j].astype(float)
            chan_x[col] = x
            rms = np.sqrt(np.mean(x ** 2))
            peak = np.max(np.abs(x))
            std = np.std(x)
            mean = np.mean(x)
            mean_abs = np.mean(np.abs(x))
            crest = peak / (rms + 1e-9)
            shape = rms / (mean_abs + 1e-9)
            impulse = peak / (mean_abs + 1e-9)
            clearance = peak / (np.mean(np.sqrt(np.abs(x))) ** 2 + 1e-9)
            zcr = np.mean(np.diff(np.sign(x - mean)) != 0)
            dx = np.diff(x)
            dx_mean_abs = np.mean(np.abs(dx)) if len(dx) else 0.0
            dx_max_abs = np.max(np.abs(dx)) if len(dx) else 0.0
            if std > 1e-9:
                skew = np.mean(((x - mean) / std) ** 3)
                kurt = np.mean(((x - mean) / std) ** 4) - 3
            else:
                skew, kurt = 0.0, 0.0
            spec = np.abs(np.fft.rfft(x - mean))
            dom_ratio = (spec.max() / (spec.sum() + 1e-9)) if spec.sum() > 0 else 0.0
            b_low, b_mid, b_high = band_energy_ratios(spec)
            sent = spectral_entropy(spec)
            env = np.abs(hilbert(x))
            env_rms = np.sqrt(np.mean(env ** 2))

            feat[f"{col}_rms"] = rms
            feat[f"{col}_peak"] = peak
            feat[f"{col}_std"] = std
            feat[f"{col}_crest"] = crest
            feat[f"{col}_shape"] = shape
            feat[f"{col}_impulse"] = impulse
            feat[f"{col}_clearance"] = clearance
            feat[f"{col}_zcr"] = zcr
            feat[f"{col}_dxmeanabs"] = dx_mean_abs
            feat[f"{col}_dxmaxabs"] = dx_max_abs
            feat[f"{col}_skew"] = skew
            feat[f"{col}_kurt"] = kurt
            feat[f"{col}_domratio"] = dom_ratio
            feat[f"{col}_bandlow"] = b_low
            feat[f"{col}_bandmid"] = b_mid
            feat[f"{col}_bandhigh"] = b_high
            feat[f"{col}_specentropy"] = sent
            feat[f"{col}_envrms"] = env_rms

        v0, v1, cur = chan_x["AI0_Vibration"], chan_x["AI1_Vibration"], chan_x["AI2_Current"]
        feat["corr_v0_v1"] = np.corrcoef(v0, v1)[0, 1] if np.std(v0) > 1e-9 and np.std(v1) > 1e-9 else 0.0
        feat["corr_v0_cur"] = np.corrcoef(v0, cur)[0, 1] if np.std(v0) > 1e-9 and np.std(cur) > 1e-9 else 0.0
        feat["corr_v1_cur"] = np.corrcoef(v1, cur)[0, 1] if np.std(v1) > 1e-9 and np.std(cur) > 1e-9 else 0.0
        rows.append(feat)
    return rows


def extract_window_features(df, win=WIN, stride=STRIDE):
    rows = []
    for _, seg_df in df.groupby("segment_id"):
        rows.extend(_extract_from_segment(seg_df, win, stride))
    return pd.DataFrame(rows)


feat_train = extract_window_features(train_df)
feat_calib = extract_window_features(calib_df)
feat_fa_test = extract_window_features(fa_test_df)
feat_outlier = extract_window_features(outlier_df)
FEATURE_COLS = [c for c in feat_train.columns if c not in ("t_start", "t_end")]
print("feature dim:", len(FEATURE_COLS), "windows:", len(feat_train), len(feat_calib), len(feat_fa_test), len(feat_outlier))

# ------------------------------------------------------------------
# 3. 모델들
# ------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA

scaler = StandardScaler().fit(feat_train[FEATURE_COLS])
X_train = scaler.transform(feat_train[FEATURE_COLS])
X_calib = scaler.transform(feat_calib[FEATURE_COLS])
X_fa = scaler.transform(feat_fa_test[FEATURE_COLS])
X_out = scaler.transform(feat_outlier[FEATURE_COLS])

mu = feat_train[FEATURE_COLS].mean()
sigma = feat_train[FEATURE_COLS].std()


def baseline_score(feat_df):
    z = (feat_df[FEATURE_COLS] - mu) / (sigma + 1e-9)
    return z.abs().max(axis=1).to_numpy()


scores = {}  # name -> dict(calib, fa, out)

scores["baseline_3sigma"] = dict(calib=baseline_score(feat_calib), fa=baseline_score(feat_fa_test), out=baseline_score(feat_outlier))

iso = IsolationForest(n_estimators=300, contamination="auto", random_state=RANDOM_STATE).fit(X_train)
scores["isolation_forest"] = dict(calib=-iso.score_samples(X_calib), fa=-iso.score_samples(X_fa), out=-iso.score_samples(X_out))

ocsvm = OneClassSVM(kernel="rbf", nu=0.02, gamma="scale").fit(X_train)
scores["one_class_svm"] = dict(calib=-ocsvm.decision_function(X_calib), fa=-ocsvm.decision_function(X_fa), out=-ocsvm.decision_function(X_out))

lof = LocalOutlierFactor(n_neighbors=35, novelty=True).fit(X_train)
scores["lof"] = dict(calib=-lof.decision_function(X_calib), fa=-lof.decision_function(X_fa), out=-lof.decision_function(X_out))

pca = PCA(n_components=0.95, random_state=RANDOM_STATE).fit(X_train)
var = pca.explained_variance_


def t2_spe(X):
    sc = pca.transform(X)
    t2 = (sc ** 2 / var).sum(axis=1)
    recon = pca.inverse_transform(sc)
    spe = ((X - recon) ** 2).sum(axis=1)
    return t2, spe


t2_c, spe_c = t2_spe(X_calib)
t2_thr, spe_thr = np.percentile(t2_c, 99), np.percentile(spe_c, 99)


def pca_score(X):
    t2, spe = t2_spe(X)
    return np.maximum(t2 / t2_thr, spe / spe_thr)


scores["pca_t2_spe"] = dict(calib=pca_score(X_calib), fa=pca_score(X_fa), out=pca_score(X_out))

# ------------------------------------------------------------------
# 4. GRU-Autoencoder (시퀀스 모델)
# ------------------------------------------------------------------
import torch
import torch.nn as nn

torch.manual_seed(RANDOM_STATE)


def make_raw_windows(df, win=WIN, stride=STRIDE):
    windows = []
    for _, seg_df in df.groupby("segment_id"):
        values = seg_df[SIGNAL_COLS].to_numpy().astype(np.float32)
        for start in range(0, len(values) - win + 1, stride):
            windows.append(values[start:start + win])
    return np.stack(windows)


raw_train = make_raw_windows(train_df)
raw_calib = make_raw_windows(calib_df)
raw_fa = make_raw_windows(fa_test_df)
raw_out = make_raw_windows(outlier_df)

raw_mean = raw_train.reshape(-1, 3).mean(axis=0)
raw_std = raw_train.reshape(-1, 3).std(axis=0) + 1e-6


def norm(x):
    return (x - raw_mean) / raw_std


class ConvAE(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(channels, 16, 3, padding=1), nn.ReLU(), nn.Conv1d(16, 8, 3, padding=1), nn.ReLU())
        self.dec = nn.Sequential(nn.Conv1d(8, 16, 3, padding=1), nn.ReLU(), nn.Conv1d(16, channels, 3, padding=1))

    def forward(self, x):
        return self.dec(self.enc(x))


class GRUAE(nn.Module):
    def __init__(self, channels=3, hidden=16, latent=8):
        super().__init__()
        self.hidden = hidden
        self.enc = nn.GRU(channels, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.dec = nn.GRU(hidden, hidden, batch_first=True)
        self.out_proj = nn.Linear(hidden, channels)

    def forward(self, x):
        # x: (N, T, C)
        _, h = self.enc(x)
        z = self.to_latent(h[-1])
        h0 = self.from_latent(z).unsqueeze(0)
        dec_in = torch.zeros(x.size(0), x.size(1), self.hidden, device=x.device)
        out, _ = self.dec(dec_in, h0)
        return self.out_proj(out)


def to_tensor_conv(raw):
    return torch.tensor(norm(raw).transpose(0, 2, 1))


def to_tensor_seq(raw):
    return torch.tensor(norm(raw))


def train_model(model, X_tr, n_epochs=30, batch=128, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(n_epochs):
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), batch):
            idx = perm[i:i + batch]
            xb = X_tr[idx]
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, xb)
            loss.backward()
            opt.step()
    return model


ae_conv = train_model(ConvAE(), to_tensor_conv(raw_train))


def recon_error_conv(raw):
    ae_conv.eval()
    with torch.no_grad():
        x = to_tensor_conv(raw)
        out = ae_conv(x)
        return ((out - x) ** 2).mean(dim=(1, 2)).numpy()


scores["conv_autoencoder"] = dict(calib=recon_error_conv(raw_calib), fa=recon_error_conv(raw_fa), out=recon_error_conv(raw_out))

ae_gru = train_model(GRUAE(), to_tensor_seq(raw_train), n_epochs=20)


def recon_error_gru(raw):
    ae_gru.eval()
    with torch.no_grad():
        x = to_tensor_seq(raw)
        out = ae_gru(x)
        return ((out - x) ** 2).mean(dim=(1, 2)).numpy()


scores["gru_autoencoder"] = dict(calib=recon_error_gru(raw_calib), fa=recon_error_gru(raw_fa), out=recon_error_gru(raw_out))

# ------------------------------------------------------------------
# 5. 임계값(calib 99pct) + 평가
# ------------------------------------------------------------------
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score

results = []
for name, s in scores.items():
    thr = np.percentile(s["calib"], 99)
    y_true = np.concatenate([np.zeros(len(s["fa"])), np.ones(len(s["out"]))])
    y_score = np.concatenate([s["fa"], s["out"]])
    y_pred = (y_score > thr).astype(int)
    results.append({
        "model": name,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "fa_rate": (s["fa"] > thr).mean(),
        "detection_rate": (s["out"] > thr).mean(),
    })

res_df = pd.DataFrame(results).sort_values("f1", ascending=False)
print(res_df.to_string(index=False))

# ------------------------------------------------------------------
# 6. 앙상블 (rank-average) + 확률보정(Platt/logistic)
# ------------------------------------------------------------------
top_models = res_df.sort_values("f1", ascending=False)["model"].head(3).tolist()
print("ensemble members (top-3 by F1):", top_models)


def rank_normalize(train_arr, *arrs):
    # calib 분포 기준으로 percentile rank화 (0~1)
    sorted_train = np.sort(train_arr)
    return [np.searchsorted(sorted_train, a) / len(sorted_train) for a in arrs]


ens_calib = np.zeros(len(feat_calib))
ens_fa = np.zeros(len(feat_fa_test))
ens_out = np.zeros(len(feat_outlier))
for name in top_models:
    s = scores[name]
    r_calib, r_fa, r_out = rank_normalize(s["calib"], s["calib"], s["fa"], s["out"])
    ens_calib += r_calib
    ens_fa += r_fa
    ens_out += r_out
ens_calib /= len(top_models)
ens_fa /= len(top_models)
ens_out /= len(top_models)

ens_thr = np.percentile(ens_calib, 99)
y_true = np.concatenate([np.zeros(len(ens_fa)), np.ones(len(ens_out))])
y_score = np.concatenate([ens_fa, ens_out])
y_pred = (y_score > ens_thr).astype(int)
print("ENSEMBLE f1:", f1_score(y_true, y_pred), "fa_rate:", (ens_fa > ens_thr).mean(), "det_rate:", (ens_out > ens_thr).mean())

# Platt scaling: rank score -> 확률
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(y_score.reshape(-1, 1), y_true)
proba = lr.predict_proba(y_score.reshape(-1, 1))[:, 1]
print("calibrated proba range:", proba.min(), proba.max())

# ------------------------------------------------------------------
# 7. 변수 중요도 (permutation importance on isolation forest)
# ------------------------------------------------------------------
from sklearn.inspection import permutation_importance

y_true_all = np.concatenate([np.zeros(len(X_fa)), np.ones(len(X_out))])
X_all = np.vstack([X_fa, X_out])


class ScoreWrapper:
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self

    def score(self, X, y):
        s = -self.model.score_samples(X)
        thr = np.percentile(-self.model.score_samples(X_calib), 99)
        pred = (s > thr).astype(int)
        return f1_score(y, pred, zero_division=0)


wrapper = ScoreWrapper(iso)
pi = permutation_importance(wrapper, X_all, y_true_all, n_repeats=10, random_state=RANDOM_STATE)
imp_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": pi.importances_mean}).sort_values("importance", ascending=False)
print(imp_df.head(15))

# ------------------------------------------------------------------
# 8. Walk-forward CV로 오경보율 안정성 확인
# ------------------------------------------------------------------
n_folds = 4
fold_size = len(feat_train) // (n_folds + 1)
fa_rates = []
for k in range(n_folds):
    tr_end = fold_size * (k + 1)
    val_start = tr_end
    val_end = val_start + fold_size
    if val_end > len(feat_train):
        break
    tr_feat = feat_train.iloc[:tr_end]
    val_feat = feat_train.iloc[val_start:val_end]
    Xtr = scaler.transform(tr_feat[FEATURE_COLS])
    Xval = scaler.transform(val_feat[FEATURE_COLS])
    m = IsolationForest(n_estimators=200, random_state=RANDOM_STATE).fit(Xtr)
    sc_val = -m.score_samples(Xval)
    thr = np.percentile(sc_val, 99)
    fa_rates.append((sc_val > thr).mean())
print("walk-forward FA rates:", fa_rates, "mean:", np.mean(fa_rates), "std:", np.std(fa_rates))
