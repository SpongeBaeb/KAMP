import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import optuna
import os
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('preprocessed_data.csv')

# Feature Engineering
for i in range(1, 25):
    df[f'Peak_lag_{i}'] = df['Peak'].shift(i)

# Rolling features
df['Peak_roll_mean_3'] = df['Peak'].shift(1).rolling(window=3).mean()
df['Peak_roll_std_3'] = df['Peak'].shift(1).rolling(window=3).std()
df['Peak_roll_mean_24'] = df['Peak'].shift(1).rolling(window=24).mean()
df['Peak_roll_max_24'] = df['Peak'].shift(1).rolling(window=24).max()
df['Peak_roll_min_24'] = df['Peak'].shift(1).rolling(window=24).min()
df['Peak_roll_std_24'] = df['Peak'].shift(1).rolling(window=24).std()

df['Peak_diff_1'] = df['Peak_lag_1'] - df['Peak_lag_2']
df['Peak_diff_24'] = df['Peak_lag_1'] - df['Peak_lag_24']

# Discomfort Index (THI)
# THI = T - 0.55 * (1 - H/100) * (T - 14.5)
df['THI'] = df['기온'] - 0.55 * (1 - df['습도']/100.0) * (df['기온'] - 14.5)

df = df.dropna().reset_index(drop=True)

# Drop in-rush anomalies as agreed
df = df[df['is_inrush'] == 0].reset_index(drop=True)

# Important: We predict the DIFFERENCE (Peak_t - Peak_lag_1) instead of absolute Peak.
# This makes the time-series stationary and much easier for LightGBM.
y_abs = df['Peak']
y = df['Peak'] - df['Peak_lag_1'] # Target is now the difference

X = df.drop(columns=['Peak', '날짜', '시간', '15분', '30분', '45분', '60분', '평균', 'is_inrush', 'Peak_MA3', 'Peak_MA6', 'Peak_MA12', 'Peak_MA24'], errors='ignore')

if 'day' in X.columns:
    X['day'] = X['day'].astype('category')
if 'm' in X.columns:
    X['m'] = X['m'].astype('category')

tscv = TimeSeriesSplit(n_splits=5)

def objective(trial):
    param = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 512),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    }

    rmse_scores = []
    
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
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        preds_diff = model.predict(X_test)
        # Reconstruct absolute prediction
        preds_abs = X_test['Peak_lag_1'] + preds_diff
        actual_abs = y_abs.iloc[test_idx]
        
        rmse = np.sqrt(mean_squared_error(actual_abs, preds_abs))
        rmse_scores.append(rmse)
        
    return np.mean(rmse_scores)

if __name__ == "__main__":
    print("Starting Optuna optimization for differenced target...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, n_jobs=1)
    
    print("\nBest parameters:", study.best_params)
    print("Best CV RMSE:", study.best_value)
    
    with open("optuna_best_result.txt", "w") as f:
        f.write(f"Best RMSE: {study.best_value}\n")
        f.write(f"Best Params: {study.best_params}\n")
