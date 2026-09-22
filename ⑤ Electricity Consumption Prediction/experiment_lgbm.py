import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import os

df = pd.read_csv('preprocessed_data.csv')

# Create all lags from 1 to 24 for the target
for i in range(1, 25):
    df[f'Peak_lag_{i}'] = df['Peak'].shift(i)

# Create rolling features
df['Peak_roll_mean_3'] = df['Peak'].shift(1).rolling(window=3).mean()
df['Peak_roll_std_3'] = df['Peak'].shift(1).rolling(window=3).std()
df['Peak_roll_mean_24'] = df['Peak'].shift(1).rolling(window=24).mean()
df['Peak_roll_max_24'] = df['Peak'].shift(1).rolling(window=24).max()
df['Peak_roll_min_24'] = df['Peak'].shift(1).rolling(window=24).min()
df['Peak_roll_std_24'] = df['Peak'].shift(1).rolling(window=24).std()
df['Peak_roll_mean_12'] = df['Peak'].shift(1).rolling(window=12).mean()

# Drop NaNs
df = df.dropna().reset_index(drop=True)

# Drop in-rush anomalies (top 5% for strict base load modeling)
p95 = df['Peak'].quantile(0.95)
df = df[df['Peak'] <= p95].reset_index(drop=True)

# Target is t
y = df['Peak']
X = df.drop(columns=['Peak', '날짜', '시간', '15분', '30분', '45분', '60분', '평균', 'is_inrush', 'Peak_MA3', 'Peak_MA6', 'Peak_MA12', 'Peak_MA24'], errors='ignore')

if 'day' in X.columns:
    X['day'] = X['day'].astype('category')
if 'm' in X.columns:
    X['m'] = X['m'].astype('category')


from sklearn.model_selection import TimeSeriesSplit

# Time Series Cross Validation (5-folds)
tscv = TimeSeriesSplit(n_splits=5)

rmse_scores = []
r2_scores = []

print("Starting Time Series Cross-Validation (Walk-Forward)...")

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train_full, y_train_full = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    
    # Split train_full further into train and valid (last 10% for early stopping)
    valid_size = int(len(X_train_full) * 0.1)
    if valid_size == 0: valid_size = 1
    
    X_train, y_train = X_train_full.iloc[:-valid_size], y_train_full.iloc[:-valid_size]
    X_valid, y_valid = X_train_full.iloc[-valid_size:], y_train_full.iloc[-valid_size:]
    
    model = lgb.LGBMRegressor(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=10,
        num_leaves=255,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42
    )

    # Use X_valid for early stopping to prevent test set leakage
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    rmse_scores.append(rmse)
    r2_scores.append(r2)
    
    print(f"Fold {fold+1} - Test RMSE: {rmse:.4f}, Test R^2: {r2:.4f}")

avg_rmse = np.mean(rmse_scores)
avg_r2 = np.mean(r2_scores)

print(f"\nAverage Test RMSE (5-Folds): {avg_rmse:.4f}")
print(f"Average Test R^2 (5-Folds): {avg_r2:.4f}")

with open('lgbm_cv_result.txt', 'w') as f:
    f.write(f"CV Average RMSE: {avg_rmse:.4f}\nCV Average R2: {avg_r2:.4f}\n")

