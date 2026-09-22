import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor
import os
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
df = df[df['is_inrush'] == 0].reset_index(drop=True)

y_abs = df['Peak']
y = df['Peak'] - df['Peak_lag_1']

X = df.drop(columns=['Peak', '날짜', '시간', '15분', '30분', '45분', '60분', '평균', 'is_inrush', 'Peak_MA3', 'Peak_MA6', 'Peak_MA12', 'Peak_MA24'], errors='ignore')

# CatBoost prefers int or string for categoricals
cat_features = ['day', 'm']
for col in cat_features:
    if col in X.columns:
        X[col] = X[col].astype(int)

tscv = TimeSeriesSplit(n_splits=5)
rmse_scores = []
r2_scores = []

print("Starting CatBoost Time Series CV...")
for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train_full, y_train_full = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
    valid_size = int(len(X_train_full) * 0.1)
    if valid_size == 0: valid_size = 1
    
    X_train, y_train = X_train_full.iloc[:-valid_size], y_train_full.iloc[:-valid_size]
    X_valid, y_valid = X_train_full.iloc[-valid_size:], y_train_full.iloc[-valid_size:]
    
    # MAE objective for robustness against outliers
    model = CatBoostRegressor(
        iterations=3000,
        learning_rate=0.03,
        depth=8,
        loss_function='MAE',
        eval_metric='RMSE',
        random_seed=42,
        od_type='Iter',
        od_wait=100,
        verbose=False
    )
    
    model.fit(
        X_train, y_train,
        cat_features=cat_features,
        eval_set=(X_valid, y_valid)
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

print(f"\nAverage CatBoost Test RMSE: {avg_rmse:.4f}")
print(f"Average CatBoost Test R^2: {avg_r2:.4f}")

with open('catboost_cv_result.txt', 'w') as f:
    f.write(f"CV Average RMSE: {avg_rmse:.4f}\nCV Average R2: {avg_r2:.4f}\n")
