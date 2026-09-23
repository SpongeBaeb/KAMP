import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import average_precision_score, make_scorer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

def optimize_prauc():
    print("Loading CN7 Data for GridSearchCV...")
    df_cn7 = pd.read_csv('data/moldset_labeled_cn7.csv')
    
    cols_to_drop = ['Unnamed: 0', 'Clamp_Open_Position']
    df_cn7 = df_cn7.drop(columns=[col for col in cols_to_drop if col in df_cn7.columns])
    
    target = 'PassOrFail'
    features = [c for c in df_cn7.columns if c != target]
    
    X = df_cn7[features]
    y = df_cn7[target]
    
    scale_pos = (y == 0).sum() / (y == 1).sum()
    
    print("Running LightGBM GridSearchCV for PR-AUC...")
    lgbm = LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1)
    
    param_grid_lgbm = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [15, 31, 63],
        'min_child_samples': [1, 5, 10]
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pr_scorer = make_scorer(average_precision_score, needs_proba=True)
    
    grid_lgbm = GridSearchCV(lgbm, param_grid_lgbm, scoring=pr_scorer, cv=skf, n_jobs=-1)
    grid_lgbm.fit(X, y)
    
    best_lgbm_prauc = grid_lgbm.best_score_
    print(f"Best LightGBM PR-AUC: {best_lgbm_prauc:.4f}")
    
    print("Running XGBoost GridSearchCV for PR-AUC...")
    xgb = XGBClassifier(scale_pos_weight=scale_pos, random_state=42, eval_metric='logloss')
    
    param_grid_xgb = {
        'n_estimators': [100, 300],
        'learning_rate': [0.01, 0.05],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3]
    }
    
    grid_xgb = GridSearchCV(xgb, param_grid_xgb, scoring=pr_scorer, cv=skf, n_jobs=-1)
    grid_xgb.fit(X, y)
    
    best_xgb_prauc = grid_xgb.best_score_
    print(f"Best XGBoost PR-AUC: {best_xgb_prauc:.4f}")
    
    final_prauc = max(best_lgbm_prauc, best_xgb_prauc)
    print(f"\n[WINNER] Optimization Complete!")
    print(f"Final Best PR-AUC: {final_prauc:.4f}")
    
    if final_prauc >= 0.90:
        print("\nGOAL ACHIEVED: PR-AUC > 0.90!")
    else:
        print("\nFailed to reach 0.90 through CV. Data constraints are severe.")

if __name__ == "__main__":
    optimize_prauc()
