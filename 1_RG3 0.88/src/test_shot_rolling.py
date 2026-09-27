import numpy as np
import pandas as pd
from pathlib import Path
import time
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
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
    
    # Identify cavities correctly.
    # Group by machine, then consecutive identical sensors are the same shot.
    # Actually, we can just deduplicate to get the shot sequence.
    # But wait, we need to merge the features back. 
    # Since the sensors are exactly identical for both cavities, merging on sensor_cols will duplicate properly!
    
    shots = combined.drop_duplicates(subset=sensor_cols + ['machine']).copy()
    p(f"Total rows: {len(combined)}, Unique shots: {len(shots)}")
    
    # Calculate rolling features on the shots dataframe
    top_sensors = ['Injection_Time', 'Filling_Time', 'Plasticizing_Time', 'Cycle_Time', 
                   'Max_Injection_Pressure', 'Max_Switch_Over_Pressure', 'Max_Back_Pressure']
    
    for machine_id in [0, 1]:
        mask = shots['machine'] == machine_id
        
        # We must preserve order! Sort by original index if necessary, but drop_duplicates keeps first appearance order.
        for col in sensor_cols: # Let's do it for all sensor cols to be comprehensive
            shots.loc[mask, f'{col}_diff1'] = shots.loc[mask, col].diff(1)
            shots.loc[mask, f'{col}_roll3_mean'] = shots.loc[mask, col].rolling(3, min_periods=1).mean()
            shots.loc[mask, f'{col}_roll3_std'] = shots.loc[mask, col].rolling(3, min_periods=1).std()
            shots.loc[mask, f'{col}_roll10_mean'] = shots.loc[mask, col].rolling(10, min_periods=1).mean()
            shots.loc[mask, f'{col}_roll10_std'] = shots.loc[mask, col].rolling(10, min_periods=1).std()
            shots.loc[mask, f'{col}_roll20_mean'] = shots.loc[mask, col].rolling(20, min_periods=1).mean()
            shots.loc[mask, f'{col}_roll20_std'] = shots.loc[mask, col].rolling(20, min_periods=1).std()
            
        shots.loc[mask, 'Shot_Index'] = range(mask.sum())
    
    shots = shots.fillna(0)
    
    # Now merge these features back to the combined dataframe
    # We can just join on the original index of the shots dataframe!
    # Wait, if we join on sensor_cols, it's safer.
    merge_cols = sensor_cols + ['machine']
    feat_df = shots.drop(columns=['PassOrFail'])
    
    final_df = pd.merge(combined, feat_df, on=merge_cols, how='left')
    
    # Add Virtual ID (Cavity ID)
    final_df['Virtual_ID'] = final_df.groupby(merge_cols, sort=False).cumcount()
    
    # Evaluation
    X = final_df.drop(columns=['PassOrFail'])
    y = final_df['PassOrFail']
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    model = ExtraTreesClassifier(n_estimators=200, max_depth=6, class_weight='balanced', random_state=42, n_jobs=-1)
    
    aucs = []
    for train_idx, test_idx in cv.split(X, y):
        Xtr, ytr = X.iloc[train_idx], y.iloc[train_idx]
        Xte, yte = X.iloc[test_idx], y.iloc[test_idx]
        model.fit(Xtr, ytr)
        probs = model.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(yte, probs))
        
    p(f"ExtraTrees ROC-AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")
    
if __name__ == "__main__":
    main()
