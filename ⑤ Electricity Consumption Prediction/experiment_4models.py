import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import ExtraTreesRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

print("데이터 준비 중...")
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

print("피처 선택 (Top 25) 진행 중...")
selector = ExtraTreesRegressor(n_estimators=300, random_state=42, n_jobs=-1)
selector.fit(X, y_diff)
imp = pd.Series(selector.feature_importances_, index=X.columns)
top25 = imp.nlargest(25).index.tolist()

tscv = TimeSeriesSplit(n_splits=5)

rmse_scores = []
r2_scores = []
weights_history = []

print("\n=== 4-Model Ensemble (ET + LGB + XGB + CB) Validation Blending ===")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_tr_full = X[top25].iloc[train_idx]
    y_tr_full = y_diff.iloc[train_idx]
    X_te = X[top25].iloc[test_idx]
    actual = y_abs.iloc[test_idx]
    
    # 10% of train for early stopping & blending weight optimization
    val_sz = max(int(len(X_tr_full) * 0.1), 1)
    X_tr = X_tr_full.iloc[:-val_sz]
    y_tr = y_tr_full.iloc[:-val_sz]
    X_val = X_tr_full.iloc[-val_sz:]
    y_val = y_tr_full.iloc[-val_sz:]

    # 1. ExtraTrees (Multi-seed)
    seeds = [42, 123, 456, 789, 2024]
    p_et_val_list, p_et_te_list = [], []
    for s in seeds:
        m = ExtraTreesRegressor(n_estimators=500, random_state=s, n_jobs=-1, min_samples_leaf=3)
        m.fit(X_tr, y_tr)
        p_et_val_list.append(m.predict(X_val))
        p_et_te_list.append(m.predict(X_te))
    p_et_val = np.mean(p_et_val_list, axis=0)
    p_et_te = np.mean(p_et_te_list, axis=0)

    # 2. LightGBM
    m_lgb = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=64,
        max_depth=12, feature_fraction=0.7, verbosity=-1, random_state=42)
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=[lgb.early_stopping(50, verbose=False)])
    p_lgb_val = m_lgb.predict(X_val)
    p_lgb_te = m_lgb.predict(X_te)

    # 3. XGBoost
    m_xgb = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False, early_stopping_rounds=50)
    p_xgb_val = m_xgb.predict(X_val)
    p_xgb_te = m_xgb.predict(X_te)

    # 4. CatBoost
    m_cb = CatBoostRegressor(iterations=1000, learning_rate=0.03, depth=6,
        loss_function='RMSE', random_seed=42, verbose=False)
    m_cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=50)
    p_cb_val = m_cb.predict(X_val)
    p_cb_te = m_cb.predict(X_te)

    # Blending Optimization on Validation Set
    def loss_func(weights):
        pred = (weights[0] * p_et_val + 
                weights[1] * p_lgb_val + 
                weights[2] * p_xgb_val + 
                weights[3] * p_cb_val)
        return np.sqrt(mean_squared_error(y_val, pred))

    # Constraint: sum of weights = 1, weights bounds = [0, 1]
    cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    bounds = [(0, 1)] * 4
    init_w = [0.25, 0.25, 0.25, 0.25]
    
    res = minimize(loss_func, init_w, method='SLSQP', bounds=bounds, constraints=cons)
    best_w = res.x
    weights_history.append(best_w)

    # Final Prediction on Test Set
    pred_diff_te = (best_w[0] * p_et_te + 
                    best_w[1] * p_lgb_te + 
                    best_w[2] * p_xgb_te + 
                    best_w[3] * p_cb_te)
    
    preds_abs = X.iloc[test_idx]['Peak_lag_1'].values + pred_diff_te
    rmse = np.sqrt(mean_squared_error(actual, preds_abs))
    r2 = r2_score(actual, preds_abs)
    
    print(f'Fold {fold+1}: RMSE={rmse:.4f}, R2={r2:.4f} | Weights(ET, LGB, XGB, CB): {best_w[0]:.2f}, {best_w[1]:.2f}, {best_w[2]:.2f}, {best_w[3]:.2f}')
    rmse_scores.append(rmse)
    r2_scores.append(r2)

print(f'\n=== Final Results ===')
print(f'Final Avg RMSE: {np.mean(rmse_scores):.4f}')
print(f'Final Avg R2: {np.mean(r2_scores):.4f}')
avg_w = np.mean(weights_history, axis=0)
print(f'Average Weights: ET={avg_w[0]:.2f}, LGB={avg_w[1]:.2f}, XGB={avg_w[2]:.2f}, CB={avg_w[3]:.2f}')
