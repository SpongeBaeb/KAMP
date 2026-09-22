import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
from xgboost import XGBRegressor
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

# Exponential moving average
df['Peak_ewma_3'] = df['Peak'].shift(1).ewm(span=3).mean()
df['Peak_ewma_24'] = df['Peak'].shift(1).ewm(span=24).mean()

# Drop NaNs
df = df.dropna().reset_index(drop=True)

# Drop in-rush anomalies as agreed
df = df[df['is_inrush'] == 0].reset_index(drop=True)

# Target is t
y = df['Peak']
X = df.drop(columns=['Peak', '날짜', '시간', '15분', '30분', '45분', '60분', '평균', 'is_inrush', 'Peak_MA3', 'Peak_MA6', 'Peak_MA12', 'Peak_MA24'], errors='ignore')

if 'day' in X.columns:
    X['day'] = X['day'].astype('category')
if 'm' in X.columns:
    X['m'] = X['m'].astype('category')

# Train test split (temporal)
train_size = int(len(X) * 0.8)
X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]

print("Training LightGBM model...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=10,
    num_leaves=255,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=100)]
)

print("Training XGBoost model...")
xgb_model = XGBRegressor(
    n_estimators=3000,
    learning_rate=0.01,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    enable_categorical=True,
    early_stopping_rounds=100
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

lgb_preds = lgb_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)

final_preds = (lgb_preds + xgb_preds) / 2

rmse = np.sqrt(mean_squared_error(y_test, final_preds))
r2 = r2_score(y_test, final_preds)

print(f"\nEnsemble Test RMSE: {rmse:.4f}")
print(f"Ensemble Test R^2: {r2:.4f}")

with open('ensemble_result.txt', 'w') as f:
    f.write(f"RMSE: {rmse:.4f}\nR2: {r2:.4f}\n")
