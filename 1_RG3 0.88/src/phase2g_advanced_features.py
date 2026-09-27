import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import time

ROOT = Path(__file__).resolve().parents[1]

def load_data():
    df = pd.read_csv(ROOT.parent / "①Molding" / "moldset_labeled_rg3.csv")
    df.rename(columns={"Unnamed: 0": "idx"}, inplace=True)
    df = df.sort_values('idx').reset_index(drop=True)
    return df

def feature_engineering(df):
    feats = [c for c in df.columns if c not in ('idx', 'PassOrFail')]
    
    # Base: Virtual ID
    df['Virtual_ID'] = df.groupby(feats, sort=False).cumcount()
    
    # 1. Cumulative Fatigue
    df['Cumulative_Shots'] = df.index
    
    # 2. Volatility (과거 10번 샷의 변동성)
    core_sensors = ['Max_Injection_Pressure', 'Barrel_Temperature_1', 'Cycle_Time']
    for col in core_sensors:
        df[f'{col}_rolling_std'] = df[col].rolling(10, min_periods=2).std().fillna(0)
        df[f'{col}_rolling_range'] = df[col].rolling(10, min_periods=2).max() - df[col].rolling(10, min_periods=2).min()
        df[f'{col}_rolling_range'] = df[f'{col}_rolling_range'].fillna(0)
        
    # 3. Defect Burst (직전 20번 샷 중 불량 횟수 - 시프트 적용하여 타겟 누수 방지)
    df['Defects_in_last_20'] = df['PassOrFail'].shift(1).rolling(20, min_periods=1).sum().fillna(0)
    
    # 4. Anomaly Score (Isolation Forest 기반 비지도 학습 이상치 점수)
    # 비지도 학습이므로 전체 데이터를 넣어도 라벨 누수가 발생하지 않음
    iso = IsolationForest(n_estimators=100, random_state=42)
    iso.fit(df[feats])
    # decision_function: 낮을수록 비정상(Anomaly), 높을수록 정상
    df['Anomaly_Score'] = iso.decision_function(df[feats])
    
    return df

def evaluate_model(X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    roc_aucs = []
    pr_aucs = []
    
    for train_idx, test_idx in cv.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        model = HistGradientBoostingClassifier(class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        
        y_probs = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        roc_aucs.append(roc_auc_score(y_test, y_probs))
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        pr_aucs.append(auc(recall, precision))
        
    return np.mean(roc_aucs), np.mean(pr_aucs)

def main():
    print("Loading data...")
    df = load_data()
    
    print("Engineering advanced features...")
    df = feature_engineering(df)
    
    drop_cols = ['idx', 'PassOrFail']
    y = df['PassOrFail'].to_numpy()
    
    print("\n[ 실험 시작 ]")
    
    # 0. Baseline + Virtual ID (이전 최고 기록)
    X_prev = df.drop(columns=drop_cols + ['Cumulative_Shots', 'Defects_in_last_20', 'Anomaly_Score'] + [c for c in df.columns if 'rolling' in c]).to_numpy()
    roc, pr = evaluate_model(X_prev, y)
    print(f"1. 이전 최고치 (Virtual ID만): ROC-AUC {roc:.4f}")
    
    # 1. Cumulative Fatigue 추가
    X_1 = df.drop(columns=drop_cols + ['Defects_in_last_20', 'Anomaly_Score'] + [c for c in df.columns if 'rolling' in c]).to_numpy()
    roc, pr = evaluate_model(X_1, y)
    print(f"2. + 누적 피로도 (Cumulative Shots): ROC-AUC {roc:.4f}")
    
    # 2. Volatility 추가
    X_2 = df.drop(columns=drop_cols + ['Cumulative_Shots', 'Defects_in_last_20', 'Anomaly_Score']).to_numpy()
    roc, pr = evaluate_model(X_2, y)
    print(f"3. + 변동성 추적 (Volatility): ROC-AUC {roc:.4f}")
    
    # 3. Anomaly Score 추가
    X_3 = df.drop(columns=drop_cols + ['Cumulative_Shots', 'Defects_in_last_20'] + [c for c in df.columns if 'rolling' in c]).to_numpy()
    roc, pr = evaluate_model(X_3, y)
    print(f"4. + 이상치 점수 (Anomaly Score): ROC-AUC {roc:.4f}")
    
    # 4. Defect Burst 추가 (가장 강력할 것으로 예상되나 Leak 주의)
    X_4 = df.drop(columns=drop_cols + ['Cumulative_Shots', 'Anomaly_Score'] + [c for c in df.columns if 'rolling' in c]).to_numpy()
    roc, pr = evaluate_model(X_4, y)
    print(f"5. + 불량 군집성 (Defect Burst): ROC-AUC {roc:.4f}")
    
    # 5. ALL IN ONE
    X_all = df.drop(columns=drop_cols).to_numpy()
    roc_all, pr_all = evaluate_model(X_all, y)
    print(f"\n6. All-in-One: ROC-AUC {roc_all:.4f} (PR-AUC: {pr_all:.4f})")

if __name__ == "__main__":
    main()
