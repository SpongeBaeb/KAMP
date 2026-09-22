import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=== Experiment 6-2: Mixture of Experts (MoE) ===")
df = pd.read_csv('preprocessed_data.csv')

lags_to_add = list(range(1, 25)) + [48, 72, 96, 168]
for i in lags_to_add: df[f'Peak_lag_{i}'] = df['Peak'].shift(i)
for w in [3, 6, 12, 24, 48, 168]:
    df[f'Peak_roll_mean_{w}'] = df['Peak'].shift(1).rolling(window=w).mean()
    df[f'Peak_roll_std_{w}'] = df['Peak'].shift(1).rolling(window=w).std()

df['Peak_roll_max_24'] = df['Peak'].shift(1).rolling(window=24).max()
df['Peak_roll_min_24'] = df['Peak'].shift(1).rolling(window=24).min()
df['Peak_roll_max_168'] = df['Peak'].shift(1).rolling(window=168).max()
df['Peak_roll_min_168'] = df['Peak'].shift(1).rolling(window=168).min()
df['Peak_diff_1'] = df['Peak_lag_1'] - df['Peak_lag_2']
df['Peak_diff_24'] = df['Peak_lag_1'] - df['Peak_lag_24']
df['Peak_diff_168'] = df['Peak_lag_1'] - df['Peak_lag_168']
df['THI'] = df['기온'] - 0.55 * (1 - df['습도']/100.0) * (df['기온'] - 14.5)
for span in [3, 6, 12, 24, 168]:
    df[f'Peak_ewm_{span}'] = df['Peak'].shift(1).ewm(span=span).mean()

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
top30 = imp.nlargest(30).index.tolist()
X = X[top30]

# Clustering features (Context to determine regime)
cluster_cols = [c for c in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', '기온'] if c in X.columns]
if not cluster_cols: cluster_cols = X.columns[:5].tolist()

tscv = TimeSeriesSplit(n_splits=5)
seeds_5 = [42, 123, 456, 789, 2024]
w_et = 0.90
rmse_scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_tr_full = X.iloc[train_idx]
    y_tr_full = y_diff.iloc[train_idx]
    X_te = X.iloc[test_idx]
    actual = y_abs.iloc[test_idx]
    
    # 1. Gating Network (Clustering to find Regimes)
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
        
        # Blend experts for this regime
        regime_preds = p_et * w_et + p_lgb * (1 - w_et)
        final_preds_te[idx_te_r] = regime_preds

    preds_abs = X_te['Peak_lag_1'].values + final_preds_te
    rmse = np.sqrt(mean_squared_error(actual, preds_abs))
    print(f'Fold {fold+1}: RMSE={rmse:.4f}')
    rmse_scores.append(rmse)

avg = np.mean(rmse_scores)
print(f'\nFinal Average RMSE (MoE): {avg:.4f}')
