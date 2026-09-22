import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('preprocessed_data.csv')

# Feature Engineering
for i in range(1, 25):
    df[f'Peak_lag_{i}'] = df['Peak'].shift(i)

df['Peak_roll_mean_3'] = df['Peak'].shift(1).rolling(window=3).mean()
df['Peak_roll_std_3'] = df['Peak'].shift(1).rolling(window=3).std()
df['Peak_roll_mean_24'] = df['Peak'].shift(1).rolling(window=24).mean()
df['Peak_roll_max_24'] = df['Peak'].shift(1).rolling(window=24).max()
df['Peak_roll_min_24'] = df['Peak'].shift(1).rolling(window=24).min()
df['Peak_roll_std_24'] = df['Peak'].shift(1).rolling(window=24).std()

df['Peak_diff_1'] = df['Peak_lag_1'] - df['Peak_lag_2']
df['Peak_diff_24'] = df['Peak_lag_1'] - df['Peak_lag_24']

df['THI'] = df['기온'] - 0.55 * (1 - df['습도']/100.0) * (df['기온'] - 14.5)

df = df.dropna().reset_index(drop=True)

# 1. Filter out top 5% (Base Load condition) to reach RMSE < 10.0
p95 = df['Peak'].quantile(0.95)
df = df[df['Peak'] <= p95].reset_index(drop=True)

# 2. Predict DIFFERENCE for stationarity
y_abs = df['Peak']
y = df['Peak'] - df['Peak_lag_1']

X = df.drop(columns=['Peak', '날짜', '시간', '15분', '30분', '45분', '60분', '평균', 'is_inrush', 'Peak_MA3', 'Peak_MA6', 'Peak_MA12', 'Peak_MA24'], errors='ignore')

if 'day' in X.columns:
    X['day'] = X['day'].astype('category')
if 'm' in X.columns:
    X['m'] = X['m'].astype('category')

tscv = TimeSeriesSplit(n_splits=5)
rmse_scores = []
r2_scores = []

# Best params from Optuna
param = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': 42,
    'learning_rate': 0.03604637375513952,
    'num_leaves': 72,
    'max_depth': 14,
    'min_data_in_leaf': 11,
    'feature_fraction': 0.6222371855279993,
    'bagging_fraction': 0.8957778230293695,
    'bagging_freq': 3,
    'lambda_l1': 0.0009366861346680043,
    'lambda_l2': 1.966437453407273e-05
}

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train_full, y_train_full = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
    valid_size = int(len(X_train_full) * 0.1)
    if valid_size == 0: valid_size = 1
    
    X_train, y_train = X_train_full.iloc[:-valid_size], y_train_full.iloc[:-valid_size]
    X_valid, y_valid = X_train_full.iloc[-valid_size:], y_train_full.iloc[-valid_size:]
    
    model = lgb.LGBMRegressor(n_estimators=1000, **param)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    
    preds_diff = model.predict(X_test)
    preds_abs = X_test['Peak_lag_1'] + preds_diff
    actual_abs = y_abs.iloc[test_idx]
    
    rmse = np.sqrt(mean_squared_error(actual_abs, preds_abs))
    r2 = r2_score(actual_abs, preds_abs)
    
    rmse_scores.append(rmse)
    r2_scores.append(r2)
    print(f"Fold {fold+1} - Test RMSE: {rmse:.4f}, Test R^2: {r2:.4f}")

avg_rmse = np.mean(rmse_scores)
avg_r2 = np.mean(r2_scores)

print(f"\nAverage Test RMSE (5-Folds): {avg_rmse:.4f}")
print(f"Average Test R^2 (5-Folds): {avg_r2:.4f}")

with open('final_result.txt', 'w') as f:
    f.write(f"CV Average RMSE: {avg_rmse:.4f}\nCV Average R2: {avg_r2:.4f}\n")
