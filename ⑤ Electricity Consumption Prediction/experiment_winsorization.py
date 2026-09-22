import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=== Experiment 6-1: Target Winsorization ===")
df = pd.read_csv('preprocessed_data.csv')

lags_to_add = list(range(1, 25)) + [48, 72, 96, 168]
for i in lags_to_add:
    df[f'Peak_lag_{i}'] = df['Peak'].shift(i)

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
# IMPORTANT: No dropping of is_inrush == 1! We keep ALL data.
# df = df[df['is_inrush'] == 0].reset_index(drop=True)

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

tscv = TimeSeriesSplit(n_splits=5)
seeds_10 = [42, 123, 456, 789, 2024, 7, 31, 99, 555, 1337]

w_et = 0.90
rmse_scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_tr_full = X.iloc[train_idx]
    y_tr_full = y_diff.iloc[train_idx]
    X_te = X.iloc[test_idx]
    actual = y_abs.iloc[test_idx]
    
    # Target Winsorization (Clipping at 98th and 2nd percentiles) on TRAIN data only
    upper_bound = y_tr_full.quantile(0.98)
    lower_bound = y_tr_full.quantile(0.02)
    y_tr_full_clipped = np.clip(y_tr_full, lower_bound, upper_bound)
    
    val_sz = max(int(len(X_tr_full)*0.1), 1)
    
    preds_et = []
    for s in seeds_10:
        m_et = ExtraTreesRegressor(n_estimators=500, random_state=s, n_jobs=-1, min_samples_leaf=3)
        m_et.fit(X_tr_full, y_tr_full_clipped)
        preds_et.append(m_et.predict(X_te))
    p_et = np.mean(preds_et, axis=0)

    m_lgb = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.036, num_leaves=72,
        max_depth=14, min_data_in_leaf=11, feature_fraction=0.62,
        bagging_fraction=0.89, bagging_freq=3, verbosity=-1, random_state=42)
    m_lgb.fit(X_tr_full.iloc[:-val_sz], y_tr_full_clipped.iloc[:-val_sz],
              eval_set=[(X_tr_full.iloc[-val_sz:], y_tr_full_clipped.iloc[-val_sz:])],
              eval_metric='rmse', callbacks=[lgb.early_stopping(50, verbose=False)])
    p_lgb = m_lgb.predict(X_te)

    preds = X.iloc[test_idx]['Peak_lag_1'].values + w_et * p_et + (1-w_et) * p_lgb
    rmse = np.sqrt(mean_squared_error(actual, preds))
    print(f'Fold {fold+1}: RMSE={rmse:.4f}')
    rmse_scores.append(rmse)

avg = np.mean(rmse_scores)
print(f'\nFinal Average RMSE (Winsorization, All Data): {avg:.4f}')
