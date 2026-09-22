import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=== Grandmaster Exp 2: Gating Network (Soft Routing MoE) ===")
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

context_cols = [c for c in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', '기온'] if c in X.columns]

tscv = TimeSeriesSplit(n_splits=5)
rmse_scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_tr_full = X.iloc[train_idx].reset_index(drop=True)
    y_tr_full = y_diff.iloc[train_idx].reset_index(drop=True)
    X_te = X.iloc[test_idx].reset_index(drop=True)
    actual = y_abs.iloc[test_idx].values
    
    # Generate OOF predictions for Meta-Learner (Gating Network)
    # Using TimeSeriesSplit internally to prevent leakage in Meta-Learner training
    inner_cv = TimeSeriesSplit(n_splits=4)
    oof_et = np.zeros(len(X_tr_full))
    oof_lgb = np.zeros(len(X_tr_full))
    
    for inner_tr_idx, inner_va_idx in inner_cv.split(X_tr_full):
        X_in_tr, y_in_tr = X_tr_full.iloc[inner_tr_idx], y_tr_full.iloc[inner_tr_idx]
        X_in_va = X_tr_full.iloc[inner_va_idx]
        
        m_et_in = ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=3)
        m_et_in.fit(X_in_tr, y_in_tr)
        oof_et[inner_va_idx] = m_et_in.predict(X_in_va)
        
        m_lgb_in = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.036, num_leaves=64, verbosity=-1, random_state=42)
        m_lgb_in.fit(X_in_tr, y_in_tr)
        oof_lgb[inner_va_idx] = m_lgb_in.predict(X_in_va)
    
    # Train Base Models on Full Train Set for Test Set Predictions
    m_et = ExtraTreesRegressor(n_estimators=500, random_state=42, n_jobs=-1, min_samples_leaf=3)
    m_et.fit(X_tr_full, y_tr_full)
    te_et = m_et.predict(X_te)
    
    m_lgb = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.036, num_leaves=64, verbosity=-1, random_state=42)
    m_lgb.fit(X_tr_full, y_tr_full)
    te_lgb = m_lgb.predict(X_te)
    
    # Create Meta-Dataset (Context features + Base Predictions)
    # Only use valid OOF predictions (ignore the first fold where OOF is 0)
    valid_oof_idx = np.where(oof_et != 0)[0]
    
    meta_X_tr = pd.DataFrame({'p_et': oof_et[valid_oof_idx], 'p_lgb': oof_lgb[valid_oof_idx]})
    for c in context_cols: meta_X_tr[c] = X_tr_full.loc[valid_oof_idx, c].values
    meta_y_tr = y_tr_full.iloc[valid_oof_idx].values
    
    meta_X_te = pd.DataFrame({'p_et': te_et, 'p_lgb': te_lgb})
    for c in context_cols: meta_X_te[c] = X_te[c].values
    
    # Train Meta-Learner (Gating Network) -> Soft Routing
    meta_model = Ridge(alpha=10.0) # Ridge prevents extreme weights
    meta_model.fit(meta_X_tr, meta_y_tr)
    
    # Also try a Tree Meta-Learner (non-linear Soft Routing)
    meta_lgb = lgb.LGBMRegressor(n_estimators=100, num_leaves=8, learning_rate=0.05, verbosity=-1, random_state=42)
    meta_lgb.fit(meta_X_tr, meta_y_tr)
    
    final_preds_ridge = meta_model.predict(meta_X_te)
    final_preds_lgb = meta_lgb.predict(meta_X_te)
    
    # Blend the Meta-Learners
    final_preds_diff = (final_preds_ridge + final_preds_lgb) / 2
    
    preds_abs = X_te['Peak_lag_1'].values + final_preds_diff
    rmse = np.sqrt(mean_squared_error(actual, preds_abs))
    print(f'Fold {fold+1}: RMSE={rmse:.4f}')
    rmse_scores.append(rmse)

avg = np.mean(rmse_scores)
print(f'\nFinal Average RMSE (Soft MoE): {avg:.4f}')
