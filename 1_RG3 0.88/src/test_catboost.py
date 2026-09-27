import numpy as np
import pandas as pd
from pathlib import Path
import time
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
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
                shots.loc[mask, f'{col}_Diff1'] = shots.loc[mask, col].diff(1)
                
    shots = shots.fillna(0)
    
    merge_cols = sensor_cols + ['machine']
    feat_df = shots.drop(columns=['PassOrFail'])
    
    final_df = pd.merge(combined, feat_df, on=merge_cols, how='left')
    final_df['Virtual_ID'] = final_df.groupby(merge_cols, sort=False).cumcount()
    
    X = final_df.drop(columns=['PassOrFail'])
    y = final_df['PassOrFail']
    
    # Specify Categorical Features
    cat_features = ['Virtual_ID', 'machine']
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    # CatBoost with tuned params and cat_features
    model = CatBoostClassifier(
        iterations=1000, 
        depth=6, 
        learning_rate=0.03, 
        auto_class_weights='Balanced', 
        cat_features=cat_features,
        l2_leaf_reg=3,
        verbose=0, 
        random_state=42
    )
    
    aucs = []
    for train_idx, test_idx in cv.split(X, y):
        Xtr, ytr = X.iloc[train_idx], y.iloc[train_idx]
        Xte, yte = X.iloc[test_idx], y.iloc[test_idx]
        model.fit(Xtr, ytr, cat_features=cat_features, verbose=False)
        probs = model.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(yte, probs))
        
    p(f"CatBoost (with Cat Features) ROC-AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")

if __name__ == "__main__":
    main()
