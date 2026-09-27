import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parents[1]
data_dir = ROOT.parent / "\u2460Molding"

def main():
    df_rg3 = pd.read_csv(data_dir / "moldset_labeled_rg3.csv", index_col=0)
    df_cn7 = pd.read_csv(data_dir / "moldset_labeled_cn7.csv", index_col=0)
    df_rg3['machine'] = 1
    df_cn7['machine'] = 0
    combined = pd.concat([df_rg3, df_cn7], ignore_index=True)
    combined = combined.drop(columns=['Clamp_Open_Position'], errors='ignore')
    
    sensor_cols = [c for c in combined.columns if c not in ('PassOrFail', 'machine')]
    combined['Virtual_ID'] = combined.groupby(sensor_cols + ['machine'], sort=False).cumcount()
    
    X = combined.drop(columns=['PassOrFail'])
    y = combined['PassOrFail']
    
    # Use one K-Fold split to get out-of-fold predictions for the whole dataset
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = ExtraTreesClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
    
    oof_preds = np.zeros(len(X))
    for train_idx, test_idx in cv.split(X, y):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_preds[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
        
    combined['pred_prob'] = oof_preds
    
    # False Negatives: Actual is 1, but pred is low
    fn_mask = (combined['PassOrFail'] == 1) & (combined['pred_prob'] < 0.5)
    fns = combined[fn_mask]
    
    print(f"Total Defects: {y.sum()}")
    print(f"False Negatives (Threshold 0.5): {len(fns)}")
    
    print("\nFalse Negatives Details:")
    print(fns[['machine', 'Virtual_ID', 'pred_prob'] + sensor_cols[:3]])

if __name__ == "__main__":
    main()
