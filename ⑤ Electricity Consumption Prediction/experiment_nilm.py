import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=== Experiment 7: NILM Hidden Schedule Extraction + MoE ===")
df = pd.read_csv('preprocessed_data.csv')

# 기초 랙 변수
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

# ---------------------------------------------------------
# [핵심] NILM 기반 가상 스케줄(Pseudo-Schedule) 피처 추출
# ---------------------------------------------------------
# Peak_diff_1 (전력 변화량)을 바탕으로 대형 설비의 ON/OFF를 역추적합니다.
# 과거 데이터를 통해 현재 시점(lag_1)까지 몇 대의 기계가 켜져있는지 누적 합산합니다.

# 누수를 피하기 위해 전체 데이터에 대해 단순 임계값(GMM 대신 간단한 K-Means 1D)을 구하거나,
# 시계열 순서대로 누적 합(Cumulative Sum)을 계산합니다.
diffs = df['Peak_diff_1'].fillna(0).values.reshape(-1, 1)

# GMM으로 전력 변화량을 3개의 상태(크게 감소, 유지, 크게 증가)로 군집화
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(diffs)
means = gmm.means_.flatten()
sorted_indices = np.argsort(means)

# 군집 할당
clusters = gmm.predict(diffs)

# 군집에 따라 의미 부여 (0: 감소(OFF), 1: 유지, 2: 증가(ON))
mapped_clusters = np.zeros_like(clusters)
mapped_clusters[clusters == sorted_indices[0]] = -1  # 가장 큰 감소 (기계 OFF)
mapped_clusters[clusters == sorted_indices[1]] = 0   # 유지
mapped_clusters[clusters == sorted_indices[2]] = 1   # 가장 큰 증가 (기계 ON)

# 누적 기계 가동 대수 (Pseudo Schedule) 생성
# 값이 무한히 커지거나 작아지는 것을 막기 위해 감쇠(decay) 효과 적용 또는 MinMax 범위 지정
pseudo_schedule = np.zeros(len(df))
current_machines = 0
for i in range(len(df)):
    current_machines += mapped_clusters[i]
    # 기계 대수는 보통 0 이하로 가지 않고, 지나치게 누적되지 않도록 하루(24시간)가 지나면 서서히 리셋되는 로직 가정
    current_machines = max(0, current_machines)
    if i % 24 == 0:  
        current_machines = current_machines * 0.5 # 자정마다 리셋 효과
    pseudo_schedule[i] = current_machines

df['Pseudo_Machines_Running'] = pseudo_schedule
# ---------------------------------------------------------

df = df.dropna().reset_index(drop=True)
df = df[df['is_inrush'] == 0].reset_index(drop=True)

y_abs = df['Peak'].copy()
y_diff = df['Peak'] - df['Peak_lag_1']

# Drop raw target features
X = df.drop(columns=['Peak','날짜','시간','15분','30분','45분','60분','평균',
                      'is_inrush','Peak_MA3','Peak_MA6','Peak_MA12','Peak_MA24'], errors='ignore')
for c in X.select_dtypes(['category']).columns: X[c] = X[c].cat.codes

# Top 30 Feature Selection + 1 Pseudo Schedule Feature
selector = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
selector.fit(X, y_diff)
imp = pd.Series(selector.feature_importances_, index=X.columns)
top30 = imp.nlargest(30).index.tolist()

if 'Pseudo_Machines_Running' not in top30:
    top30.append('Pseudo_Machines_Running')
X = X[top30]

print(f"사용된 피처: {X.columns.tolist()[:5]} ... 포함")

tscv = TimeSeriesSplit(n_splits=5)
seeds_5 = [42, 123, 456, 789, 2024]
w_et = 0.90
rmse_scores = []

cluster_cols = [c for c in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', '기온', 'Pseudo_Machines_Running'] if c in X.columns]

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_tr_full = X.iloc[train_idx]
    y_tr_full = y_diff.iloc[train_idx]
    X_te = X.iloc[test_idx]
    actual = y_abs.iloc[test_idx]
    
    # 1. MoE Gating Network
    scaler = StandardScaler()
    X_tr_clus = scaler.fit_transform(X_tr_full[cluster_cols])
    kmeans = KMeans(n_clusters=2, random_state=42)
    regimes_tr = kmeans.fit_predict(X_tr_clus)
    
    X_te_clus = scaler.transform(X_te[cluster_cols])
    regimes_te = kmeans.predict(X_te_clus)
    
    final_preds_te = np.zeros(len(X_te))
    
    # 2. Train Experts
    for regime in [0, 1]:
        idx_tr_r = np.where(regimes_tr == regime)[0]
        idx_te_r = np.where(regimes_te == regime)[0]
        
        if len(idx_tr_r) < 10 or len(idx_te_r) == 0:
            continue
            
        X_tr_r = X_tr_full.iloc[idx_tr_r]
        y_tr_r = y_tr_full.iloc[idx_tr_r]
        X_te_r = X_te.iloc[idx_te_r]
        
        val_sz = max(int(len(X_tr_r)*0.1), 1)
        
        # Expert 1: ExtraTrees
        preds_et = []
        for s in seeds_5:
            m_et = ExtraTreesRegressor(n_estimators=500, random_state=s, n_jobs=-1, min_samples_leaf=3)
            m_et.fit(X_tr_r, y_tr_r)
            preds_et.append(m_et.predict(X_te_r))
        p_et = np.mean(preds_et, axis=0)

        # Expert 2: LightGBM
        m_lgb = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.036, num_leaves=64,
            max_depth=12, feature_fraction=0.7, verbosity=-1, random_state=42)
        m_lgb.fit(X_tr_r.iloc[:-val_sz], y_tr_r.iloc[:-val_sz],
                  eval_set=[(X_tr_r.iloc[-val_sz:], y_tr_r.iloc[-val_sz:])],
                  eval_metric='rmse', callbacks=[lgb.early_stopping(50, verbose=False)])
        p_lgb = m_lgb.predict(X_te_r)
        
        # Blend
        regime_preds = p_et * w_et + p_lgb * (1 - w_et)
        final_preds_te[idx_te_r] = regime_preds

    preds_abs = X_te['Peak_lag_1'].values + final_preds_te
    rmse = np.sqrt(mean_squared_error(actual, preds_abs))
    print(f'Fold {fold+1}: RMSE={rmse:.4f}')
    rmse_scores.append(rmse)

avg = np.mean(rmse_scores)
print(f'\nFinal Average RMSE (MoE + NILM Pseudo-Schedule): {avg:.4f}')
