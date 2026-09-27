import numpy as np
import pandas as pd
from pathlib import Path
import time
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from catboost import CatBoostClassifier

def p(msg): print(msg, flush=True)

ROOT = Path(__file__).resolve().parents[1]
data_dir = ROOT.parent / "\u2460Molding"

def main():
    start = time.time()
    
    df_rg3 = pd.read_csv(data_dir / "moldset_labeled_rg3.csv", index_col=0)
    df_cn7 = pd.read_csv(data_dir / "moldset_labeled_cn7.csv", index_col=0)
    
    df_rg3['machine'] = 1
    df_cn7['machine'] = 0
    combined = pd.concat([df_rg3, df_cn7], ignore_index=True)
    combined = combined.drop(columns=['Clamp_Open_Position'], errors='ignore')
    
    sensor_cols = [c for c in combined.columns if c not in ('PassOrFail', 'machine')]
    
    # Isolate unique shots to calculate true process dynamics
    shots = combined.drop_duplicates(subset=sensor_cols + ['machine']).copy()
    
    # Selected critical physics sensors
    critical_sensors = [
        'Injection_Time', 'Filling_Time', 'Plasticizing_Time', 'Cycle_Time', 
        'Max_Injection_Pressure', 'Max_Switch_Over_Pressure', 'Max_Back_Pressure',
        'Barrel_Temperature_3', 'Barrel_Temperature_4'
    ]
    
    # Process Stability Features (SPC logic)
    for machine_id in [0, 1]:
        mask = shots['machine'] == machine_id
        
        for col in critical_sensors:
            if col in shots.columns:
                # 1. Moving Range (Instability) - SPC Chart logic
                roll_max = shots.loc[mask, col].rolling(5, min_periods=1).max()
                roll_min = shots.loc[mask, col].rolling(5, min_periods=1).min()
                shots.loc[mask, f'{col}_MR5'] = roll_max - roll_min
                
                # 2. Drift from baseline (first 20 shots moving average)
                roll_mean = shots.loc[mask, col].rolling(20, min_periods=1).mean()
                shots.loc[mask, f'{col}_Drift20'] = shots.loc[mask, col] - roll_mean
                
                # 3. Short-term derivative (Velocity of change)
                shots.loc[mask, f'{col}_Vel'] = shots.loc[mask, col].diff(1).fillna(0)
                
    shots = shots.fillna(0)
    
    # Merge back
    merge_cols = sensor_cols + ['machine']
    feat_df = shots.drop(columns=['PassOrFail'])
    final_df = pd.merge(combined, feat_df, on=merge_cols, how='left')
    
    # Virtual ID (Cavity ID) & Cumulative Fatigue
    final_df['Virtual_ID'] = final_df.groupby(merge_cols, sort=False).cumcount()
    final_df['Cumulative_Shots'] = final_df.groupby('machine').cumcount()
    
    # Evaluation
    X = final_df.drop(columns=['PassOrFail'])
    y = final_df['PassOrFail']
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    models = [
        ("ExtraTrees", ExtraTreesClassifier(n_estimators=300, max_features='sqrt', max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)),
        ("CatBoost", CatBoostClassifier(iterations=500, depth=6, learning_rate=0.03, auto_class_weights='Balanced', verbose=0, random_state=42))
    ]
    
    for name, model in models:
        aucs = []
        for train_idx, test_idx in cv.split(X, y):
            Xtr, ytr = X.iloc[train_idx], y.iloc[train_idx]
            Xte, yte = X.iloc[test_idx], y.iloc[test_idx]
            model.fit(Xtr, ytr)
            probs = model.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(yte, probs))
        p(f"{name} ROC-AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")

if __name__ == "__main__":
    main()
