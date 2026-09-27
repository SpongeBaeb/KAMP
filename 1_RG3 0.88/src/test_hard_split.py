import numpy as np
import pandas as pd
from pathlib import Path
import time
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier

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
    
    # EMA for process drift
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
    
    aucs = []
    
    for train_idx, test_idx in cv.split(X, y):
        Xtr, ytr = X.iloc[train_idx].copy(), y.iloc[train_idx].copy()
        Xte, yte = X.iloc[test_idx].copy(), y.iloc[test_idx].copy()
        
        # Train two completely separate models based on Virtual_ID (Cavity)
        model_v0 = ExtraTreesClassifier(n_estimators=400, max_depth=13, min_samples_split=11, min_samples_leaf=3, max_features=0.5, class_weight='balanced', random_state=42, n_jobs=-1)
        model_v1 = ExtraTreesClassifier(n_estimators=400, max_depth=13, min_samples_split=11, min_samples_leaf=3, max_features=0.5, class_weight='balanced', random_state=42, n_jobs=-1)
        
        # Split train by Virtual_ID
        tr_mask_v0 = Xtr['Virtual_ID'] == 0
        tr_mask_v1 = Xtr['Virtual_ID'] == 1
        
        if tr_mask_v0.sum() > 0:
            model_v0.fit(Xtr[tr_mask_v0].drop(columns=['Virtual_ID']), ytr[tr_mask_v0])
        if tr_mask_v1.sum() > 0:
            # Note: v1 has very few defects, but we train anyway
            model_v1.fit(Xtr[tr_mask_v1].drop(columns=['Virtual_ID']), ytr[tr_mask_v1])
            
        # Predict test by Virtual_ID
        te_mask_v0 = Xte['Virtual_ID'] == 0
        te_mask_v1 = Xte['Virtual_ID'] == 1
        
        probs = np.zeros(len(Xte))
        if te_mask_v0.sum() > 0:
            probs[te_mask_v0] = model_v0.predict_proba(Xte[te_mask_v0].drop(columns=['Virtual_ID']))[:, 1]
        if te_mask_v1.sum() > 0:
            # If model_v1 only has 1 class in training, predict_proba might fail or output 1 column
            try:
                probs[te_mask_v1] = model_v1.predict_proba(Xte[te_mask_v1].drop(columns=['Virtual_ID']))[:, 1]
            except IndexError:
                probs[te_mask_v1] = 0.0 # If it only saw class 0 during training
            
        aucs.append(roc_auc_score(yte, probs))
        
    p(f"Hard-Split by Virtual_ID (Cavity) ROC-AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")

if __name__ == "__main__":
    main()
