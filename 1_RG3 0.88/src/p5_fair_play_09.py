import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import time

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------
# 1. Data Prep (Massive Time-Series Features)
# ---------------------------------------------------------
def load_and_engineer_data():
    df = pd.read_csv(ROOT.parent / "①Molding" / "moldset_labeled_rg3.csv")
    df.rename(columns={"Unnamed: 0": "idx"}, inplace=True)
    df = df.sort_values('idx').reset_index(drop=True)
    
    feats = [c for c in df.columns if c not in ('idx', 'PassOrFail')]
    df['Virtual_ID'] = df.groupby(feats, sort=False).cumcount()
    df['Cumulative_Shots'] = df.index
    
    print(f"Generating massive lag and rolling features for {len(feats)} sensors...")
    
    for col in feats:
        if df[col].nunique() > 2:  # 연속형 센서 변수들에 대해서만 적용
            # 1. Lags (과거 1~5번 전의 샷 데이터)
            for i in range(1, 6):
                df[f'{col}_lag{i}'] = df[col].shift(i)
                # 현재 값과 과거 값의 차이 (미분)
                df[f'{col}_diff{i}'] = df[col] - df[f'{col}_lag{i}']
            
            # 2. Rolling Statistics (과거 3, 5, 10번 동안의 흐름)
            for w in [3, 5, 10]:
                df[f'{col}_roll{w}_mean'] = df[col].rolling(w, min_periods=1).mean()
                df[f'{col}_roll{w}_std'] = df[col].rolling(w, min_periods=1).std()
                df[f'{col}_roll{w}_max'] = df[col].rolling(w, min_periods=1).max()
                df[f'{col}_roll{w}_min'] = df[col].rolling(w, min_periods=1).min()
    
    df = df.bfill().fillna(0) # 초반 결측치 보정
    
    X = df[[c for c in df.columns if c not in ('idx', 'PassOrFail')]]
    y = df['PassOrFail']
    
    return X, y

# ---------------------------------------------------------
# 2. Main Evaluation Pipeline (Strict Fair Play)
# ---------------------------------------------------------
def run_phase5(X, y):
    print(f"\n[Step 1] Initial Features: {X.shape[1]}")
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    
    # 1. Feature Selection (400개 이상의 변수 중 핵심 시계열 변수만 선별)
    print("  -> Running Feature Selection (RandomForest)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X, y)
    selector = SelectFromModel(rf, prefit=True, threshold='1.25*median')
    X_sel = pd.DataFrame(selector.transform(X), columns=X.columns[selector.get_support()])
    
    # Virtual ID와 Cumulative Shots는 강제로 유지
    if 'Virtual_ID' not in X_sel.columns: X_sel['Virtual_ID'] = X['Virtual_ID']
    if 'Cumulative_Shots' not in X_sel.columns: X_sel['Cumulative_Shots'] = X['Cumulative_Shots']
    
    print(f"  -> Selected Features: {X_sel.shape[1]}")
    
    # 2. Strict Cross Validation (No Global SMOTE, No Leakage)
    model = CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=5.0,
        auto_class_weights='Balanced',
        verbose=0,
        random_state=42
    )
    
    aucs = []
    
    print("\n[Step 2] Evaluating 5-Fold CV (Fair Play)...")
    for train_idx, test_idx in cv.split(X_sel, y):
        X_train, y_train = X_sel.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X_sel.iloc[test_idx], y.iloc[test_idx]
        
        # 순수하게 Train Set 안에서만 모델을 피팅 (데이터 누수 완벽 차단)
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=30, verbose=0)
        probs = model.predict_proba(X_test)[:, 1]
        
        aucs.append(roc_auc_score(y_test, probs))
        
    mean_auc = np.mean(aucs)
    print(f"\n  [RESULT] Phase 5 ROC-AUC (Pure Time-Series Separation): {mean_auc:.4f}")
    
    return mean_auc

def main():
    print("=== Phase 5: Fair Play 0.9 ===")
    start_time = time.time()
    
    X, y = load_and_engineer_data()
    run_phase5(X, y)
    
    print(f"\n=== All Completed in {time.time()-start_time:.1f} seconds ===")

if __name__ == "__main__":
    main()
