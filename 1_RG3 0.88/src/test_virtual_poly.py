import numpy as np
import pandas as pd
from pathlib import Path
import time
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import PolynomialFeatures

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
    
    # 1. Cavity ID (Virtual ID)
    combined['Virtual_ID'] = combined.groupby(sensor_cols + ['machine'], sort=False).cumcount()
    
    # 2. Polynomial Features on Top Sensors
    X_temp = combined.drop(columns=['PassOrFail'])
    y = combined['PassOrFail']
    
    # Find top correlated features with target
    corrs = X_temp.corrwith(y).abs().sort_values(ascending=False)
    # Exclude Virtual_ID from poly transformation, just sensors
    sensor_corrs = corrs[corrs.index.isin(sensor_cols)]
    top_sensors = sensor_corrs.head(10).index.tolist()
    p(f"Top 10 correlated sensors: {top_sensors}")
    
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = pd.DataFrame(poly.fit_transform(combined[top_sensors]))
    X_poly.columns = [f"poly_{i}" for i in range(X_poly.shape[1])]
    
    # Combine everything
    X = pd.concat([combined.drop(columns=['PassOrFail']).reset_index(drop=True), X_poly.reset_index(drop=True)], axis=1)
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    model = ExtraTreesClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
    
    aucs = []
    for train_idx, test_idx in cv.split(X, y):
        Xtr, ytr = X.iloc[train_idx], y.iloc[train_idx]
        Xte, yte = X.iloc[test_idx], y.iloc[test_idx]
        model.fit(Xtr, ytr)
        probs = model.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(yte, probs))
        
    p(f"Virtual_ID + Poly ExtraTrees ROC-AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")

if __name__ == "__main__":
    main()
