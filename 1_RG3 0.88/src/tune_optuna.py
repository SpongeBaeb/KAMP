import numpy as np
import pandas as pd
from pathlib import Path
import optuna
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier

optuna.logging.set_verbosity(optuna.logging.WARNING)

def p(msg): print(msg, flush=True)

ROOT = Path(__file__).resolve().parents[1]
data_dir = ROOT.parent / "\u2460Molding"

df_rg3 = pd.read_csv(data_dir / "moldset_labeled_rg3.csv", index_col=0)
df_cn7 = pd.read_csv(data_dir / "moldset_labeled_cn7.csv", index_col=0)

df_rg3['machine'] = 1
df_cn7['machine'] = 0
combined = pd.concat([df_rg3, df_cn7], ignore_index=True)
combined = combined.drop(columns=['Clamp_Open_Position'], errors='ignore')

sensor_cols = [c for c in combined.columns if c not in ('PassOrFail', 'machine')]
combined['Virtual_ID'] = combined.groupby(sensor_cols + ['machine'], sort=False).cumcount()

shots = combined.drop_duplicates(subset=sensor_cols + ['machine']).copy()

top_sensors = ['Max_Injection_Pressure', 'Max_Switch_Over_Pressure', 'Max_Back_Pressure',
               'Cycle_Time', 'Injection_Time', 'Filling_Time', 'Plasticizing_Time']

for machine_id in [0, 1]:
    mask = shots['machine'] == machine_id
    for col in top_sensors:
        if col in shots.columns:
            shots.loc[mask, f'{col}_EMA10'] = shots.loc[mask, col].ewm(span=10, adjust=False).mean()
            shots.loc[mask, f'{col}_EMA50'] = shots.loc[mask, col].ewm(span=50, adjust=False).mean()
            
shots = shots.fillna(0)

merge_cols = sensor_cols + ['machine']
feat_df = shots.drop(columns=['PassOrFail'])

final_df = pd.merge(combined, feat_df, on=merge_cols, how='left')
final_df['Virtual_ID'] = final_df.groupby(merge_cols, sort=False).cumcount()

X = final_df.drop(columns=['PassOrFail'])
y = final_df['PassOrFail']

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 4, 15),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None, 0.5, 0.7]),
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = ExtraTreesClassifier(**params)
    aucs = []
    
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probs = model.predict_proba(X.iloc[test_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[test_idx], probs))
        
    return np.mean(aucs)

def main():
    p("Starting Optuna optimization for ExtraTrees (Target: >0.9)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    p(f"\nBest ROC-AUC: {study.best_value:.4f}")
    p(f"Best Params: {study.best_params}")
    
    # Save best configuration
    with open("best_model_config.txt", "w") as f:
        f.write(f"Best ROC-AUC: {study.best_value:.4f}\n")
        f.write(f"Best Params: {study.best_params}\n")

if __name__ == "__main__":
    main()
