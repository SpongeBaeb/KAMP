import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from scipy.fft import rfft, rfftfreq
import warnings
warnings.filterwarnings('ignore')

print("=== Grandmaster Exp 1: Frequency Domain Features (FFT) ===")
df = pd.read_csv('preprocessed_data.csv')

lags_to_add = list(range(1, 25)) + [48, 72, 96, 168]
for i in lags_to_add: df[f'Peak_lag_{i}'] = df['Peak'].shift(i)
for w in [3, 6, 12, 24, 48, 168]:
    df[f'Peak_roll_mean_{w}'] = df['Peak'].shift(1).rolling(window=w).mean()
    df[f'Peak_roll_std_{w}'] = df['Peak'].shift(1).rolling(window=w).std()

df['Peak_diff_1'] = df['Peak_lag_1'] - df['Peak_lag_2']
df['Peak_diff_24'] = df['Peak_lag_1'] - df['Peak_lag_24']
df['Peak_diff_168'] = df['Peak_lag_1'] - df['Peak_lag_168']
df['THI'] = df['기온'] - 0.55 * (1 - df['습도']/100.0) * (df['기온'] - 14.5)
for span in [3, 6, 12, 24, 168]:
    df[f'Peak_ewm_{span}'] = df['Peak'].shift(1).ewm(span=span).mean()

# FFT 피처 추출 (전체 데이터에서 학습하지 않고 롤링 방식으로 주입하거나 고정 주파수 사용)
# 1시간 간격 데이터이므로 sampling rate = 1 (per hour)
# 전체 데이터의 Peak에 대해 FFT 수행 (가장 강한 고유 진동 주기 탐색)
y_signal = df['Peak'].fillna(method='bfill').values
N = len(y_signal)
yf = rfft(y_signal)
xf = rfftfreq(N, 1) # 주파수 (1/시간)

# 가장 강한 주파수 (DC 성분[0] 제외, 이미 알고 있는 24시간(0.0416), 12시간(0.0833), 168시간 주기 근처 제외)
amplitudes = np.abs(yf)
amplitudes[0] = 0 # DC
# 24h, 12h, 168h 주파수 대역 삭제 (이미 캘린더 피처로 존재)
for i, freq in enumerate(xf):
    period = 1/freq if freq > 0 else 0
    if (23 < period < 25) or (11.5 < period < 12.5) or (160 < period < 175):
        amplitudes[i] = 0

top3_indices = amplitudes.argsort()[-3:][::-1]
top3_freqs = xf[top3_indices]
top3_periods = [1/f for f in top3_freqs]
print(f"발견된 숨겨진 주기 (Top 3): {[f'{p:.1f}시간' for p in top3_periods]}")

# 도출된 고유 주기를 푸리에 항(Fourier Terms)으로 추가
time_idx = np.arange(len(df))
for i, freq in enumerate(top3_freqs):
    df[f'fft_sin_{i}'] = np.sin(2 * np.pi * freq * time_idx)
    df[f'fft_cos_{i}'] = np.cos(2 * np.pi * freq * time_idx)

df = df.dropna().reset_index(drop=True)
df = df[df['is_inrush'] == 0].reset_index(drop=True)

y_abs = df['Peak'].copy()
y_diff = df['Peak'] - df['Peak_lag_1']
X = df.drop(columns=['Peak','날짜','시간','15분','30분','45분','60분','평균',
                      'is_inrush','Peak_MA3','Peak_MA6','Peak_MA12','Peak_MA24'], errors='ignore')
for c in X.select_dtypes(['category']).columns: X[c] = X[c].cat.codes

selector = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
selector.fit(X, y_diff)
imp = pd.Series(selector.feature_importances_, index=X.columns)
# FFT 피처 무조건 포함
top30 = imp.nlargest(30).index.tolist()
fft_cols = [c for c in X.columns if 'fft' in c]
for c in fft_cols:
    if c not in top30: top30.append(c)
X = X[top30]

print(f"피처 개수: {len(X.columns)}")

tscv = TimeSeriesSplit(n_splits=5)
seeds_5 = [42, 123, 456, 789, 2024]
w_et = 0.90
rmse_scores = []
cluster_cols = [c for c in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', '기온'] if c in X.columns]

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_tr_full, y_tr_full = X.iloc[train_idx], y_diff.iloc[train_idx]
    X_te, actual = X.iloc[test_idx], y_abs.iloc[test_idx]
    
    scaler = StandardScaler()
    X_tr_clus = scaler.fit_transform(X_tr_full[cluster_cols])
    kmeans = KMeans(n_clusters=2, random_state=42)
    regimes_tr = kmeans.fit_predict(X_tr_clus)
    
    X_te_clus = scaler.transform(X_te[cluster_cols])
    regimes_te = kmeans.predict(X_te_clus)
    
    final_preds_te = np.zeros(len(X_te))
    
    for regime in [0, 1]:
        idx_tr_r = np.where(regimes_tr == regime)[0]
        idx_te_r = np.where(regimes_te == regime)[0]
        if len(idx_tr_r) < 10 or len(idx_te_r) == 0: continue
            
        X_tr_r, y_tr_r = X_tr_full.iloc[idx_tr_r], y_tr_full.iloc[idx_tr_r]
        X_te_r = X_te.iloc[idx_te_r]
        val_sz = max(int(len(X_tr_r)*0.1), 1)
        
        preds_et = []
        for s in seeds_5:
            m_et = ExtraTreesRegressor(n_estimators=500, random_state=s, n_jobs=-1, min_samples_leaf=3)
            m_et.fit(X_tr_r, y_tr_r)
            preds_et.append(m_et.predict(X_te_r))
        p_et = np.mean(preds_et, axis=0)

        m_lgb = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.036, num_leaves=64, max_depth=12, verbosity=-1, random_state=42)
        m_lgb.fit(X_tr_r.iloc[:-val_sz], y_tr_r.iloc[:-val_sz], eval_set=[(X_tr_r.iloc[-val_sz:], y_tr_r.iloc[-val_sz:])], eval_metric='rmse', callbacks=[lgb.early_stopping(50, verbose=False)])
        p_lgb = m_lgb.predict(X_te_r)
        
        final_preds_te[idx_te_r] = p_et * w_et + p_lgb * (1 - w_et)

    preds_abs = X_te['Peak_lag_1'].values + final_preds_te
    rmse = np.sqrt(mean_squared_error(actual, preds_abs))
    print(f'Fold {fold+1}: RMSE={rmse:.4f}')
    rmse_scores.append(rmse)

avg = np.mean(rmse_scores)
print(f'\nFinal Average RMSE (FFT + MoE): {avg:.4f}')
