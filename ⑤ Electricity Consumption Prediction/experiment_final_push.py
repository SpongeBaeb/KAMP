import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('preprocessed_data.csv')
for i in range(1, 49): df[f'Peak_lag_{i}'] = df['Peak'].shift(i)
for w in [2, 3, 4, 5, 6, 12, 24, 48]:
    df[f'Peak_roll_mean_{w}'] = df['Peak'].shift(1).rolling(window=w).mean()
    df[f'Peak_roll_std_{w}'] = df['Peak'].shift(1).rolling(window=w).std()
df['Peak_roll_max_24'] = df['Peak'].shift(1).rolling(window=24).max()
df['Peak_roll_min_24'] = df['Peak'].shift(1).rolling(window=24).min()
df['Peak_roll_max_48'] = df['Peak'].shift(1).rolling(window=48).max()
df['Peak_roll_min_48'] = df['Peak'].shift(1).rolling(window=48).min()
df['Peak_diff_1'] = df['Peak_lag_1'] - df['Peak_lag_2']
df['Peak_diff_24'] = df['Peak_lag_1'] - df['Peak_lag_24']
df['Peak_diff_48'] = df['Peak_lag_1'] - df['Peak_lag_48']
df['THI'] = df['기온'] - 0.55 * (1 - df['습도']/100.0) * (df['기온'] - 14.5)
for span in [3, 6, 12, 24]:
    df[f'Peak_ewm_{span}'] = df['Peak'].shift(1).ewm(span=span).mean()
df = df.dropna().reset_index(drop=True)
df = df[df['is_inrush'] == 0].reset_index(drop=True)

y_abs = df['Peak'].copy()
y_diff = df['Peak'] - df['Peak_lag_1']
X = df.drop(columns=['Peak','날짜','시간','15분','30분','45분','60분','평균',
                      'is_inrush','Peak_MA3','Peak_MA6','Peak_MA12','Peak_MA24'], errors='ignore')
for c in X.select_dtypes(['category']).columns:
    X[c] = X[c].cat.codes

selector = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
selector.fit(X, y_diff)
imp = pd.Series(selector.feature_importances_, index=X.columns)
top25 = imp.nlargest(25).index.tolist()

tscv = TimeSeriesSplit(n_splits=5)
seeds = [42, 123, 456, 789, 2024, 7, 31, 99, 555, 1337]

# Narrow weight search around 0.80
best_w, best_rmse_val = 0, 999
for w_et in np.arange(0.75, 0.86, 0.01):
    rmse_scores = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr = X[top25].iloc[train_idx]
        y_tr = y_diff.iloc[train_idx]
        X_te = X[top25].iloc[test_idx]
        actual = y_abs.iloc[test_idx]
        val_sz = max(int(len(X_tr)*0.1), 1)

        # 10-seed ET
        preds_et = []
        for s in seeds:
            m = ExtraTreesRegressor(n_estimators=1000, random_state=s, n_jobs=-1, min_samples_leaf=3)
            m.fit(X_tr, y_tr)
            preds_et.append(m.predict(X_te))
        p_et = np.mean(preds_et, axis=0)

        # LightGBM
        m_lgb = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.036, num_leaves=72,
            max_depth=14, min_data_in_leaf=11, feature_fraction=0.62,
            bagging_fraction=0.89, bagging_freq=3, verbosity=-1, random_state=42)
        m_lgb.fit(X_tr.iloc[:-val_sz], y_tr.iloc[:-val_sz],
                  eval_set=[(X_tr.iloc[-val_sz:], y_tr.iloc[-val_sz:])],
                  eval_metric='rmse', callbacks=[lgb.early_stopping(50, verbose=False)])
        p_lgb = m_lgb.predict(X_te)

        preds = X.iloc[test_idx]['Peak_lag_1'].values + w_et * p_et + (1-w_et) * p_lgb
        rmse_scores.append(np.sqrt(mean_squared_error(actual, preds)))
    avg = np.mean(rmse_scores)
    print(f'w_ET={w_et:.2f} -> RMSE={avg:.4f}')
    if avg < best_rmse_val:
        best_w, best_rmse_val = w_et, avg

print(f'\nBest w_ET={best_w:.2f}, Best RMSE: {best_rmse_val:.4f}')

# Final detailed run
print(f'\n=== Final (w_ET={best_w:.2f}) ===')
rmse_scores, r2_scores = [], []
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_tr = X[top25].iloc[train_idx]
    y_tr = y_diff.iloc[train_idx]
    X_te = X[top25].iloc[test_idx]
    actual = y_abs.iloc[test_idx]
    val_sz = max(int(len(X_tr)*0.1), 1)

    preds_et = []
    for s in seeds:
        m = ExtraTreesRegressor(n_estimators=1000, random_state=s, n_jobs=-1, min_samples_leaf=3)
        m.fit(X_tr, y_tr)
        preds_et.append(m.predict(X_te))
    p_et = np.mean(preds_et, axis=0)

    m_lgb = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.036, num_leaves=72,
        max_depth=14, min_data_in_leaf=11, feature_fraction=0.62,
        bagging_fraction=0.89, bagging_freq=3, verbosity=-1, random_state=42)
    m_lgb.fit(X_tr.iloc[:-val_sz], y_tr.iloc[:-val_sz],
              eval_set=[(X_tr.iloc[-val_sz:], y_tr.iloc[-val_sz:])],
              eval_metric='rmse', callbacks=[lgb.early_stopping(50, verbose=False)])
    p_lgb = m_lgb.predict(X_te)

    preds = X.iloc[test_idx]['Peak_lag_1'].values + best_w * p_et + (1-best_w) * p_lgb
    rmse = np.sqrt(mean_squared_error(actual, preds))
    r2 = r2_score(actual, preds)
    print(f'Fold {fold+1}: RMSE={rmse:.4f}, R2={r2:.4f}')
    rmse_scores.append(rmse)
    r2_scores.append(r2)
print(f'\nFinal Avg RMSE: {np.mean(rmse_scores):.4f}')
print(f'Final Avg R2: {np.mean(r2_scores):.4f}')
