import numpy as np
import pandas as pd
from pathlib import Path
import time
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

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
    
    X = combined.drop(columns=['PassOrFail'])
    y = combined['PassOrFail']
    
    spw = (y==0).sum() / max((y==1).sum(), 1)
    
    base_models = [
        ('et', ExtraTreesClassifier(n_estimators=300, max_features='sqrt', max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('cb', CatBoostClassifier(iterations=300, depth=6, learning_rate=0.03, auto_class_weights='Balanced', verbose=0, random_state=42)),
        ('lgb', lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, scale_pos_weight=spw, random_state=42, verbose=-1)),
        ('xgb', xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, scale_pos_weight=spw, random_state=42, verbosity=0))
    ]
    
    stacker = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=5,
        n_jobs=-1
    )
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    aucs = []
    for train_idx, test_idx in cv.split(X, y):
        Xtr, ytr = X.iloc[train_idx], y.iloc[train_idx]
        Xte, yte = X.iloc[test_idx], y.iloc[test_idx]
        stacker.fit(Xtr, ytr)
        probs = stacker.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(yte, probs))
        
    p(f"Stacking ROC-AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")

if __name__ == "__main__":
    main()
