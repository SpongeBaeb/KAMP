import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os

def run_split_comparison():
    # RG3 라벨 데이터 로드
    file_path = r"C:\Users\stebe\.gemini\antigravity-ide\scratch\KAMP\①Molding\moldset_labeled_rg3.csv"
    if not os.path.exists(file_path):
        print("데이터 파일을 찾을 수 없습니다.")
        return

    df = pd.read_csv(file_path)
    
    # KAMP 데이터는 시간순으로 정렬되어 있음
    # X, y 분리
    X = df.drop(columns=['PassOrFail', 'Unnamed: 0'], errors='ignore')
    y = df['PassOrFail']
    
    # ----------------------------------------------------
    # 1. KAMP 가이드북 방식: 랜덤 섞기 분할 (Data Leakage 발생)
    # ----------------------------------------------------
    X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    
    rf_rand = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_rand.fit(X_train_rand, y_train_rand)
    auc_rand = roc_auc_score(y_test_rand, rf_rand.predict_proba(X_test_rand)[:, 1])
    
    # ----------------------------------------------------
    # 2. 정석적인 시계열 분할 (미래 데이터를 학습하지 않음)
    # ----------------------------------------------------
    split_idx = int(len(df) * 0.7)
    X_train_time = X.iloc[:split_idx]
    y_train_time = y.iloc[:split_idx]
    
    X_test_time = X.iloc[split_idx:]
    y_test_time = y.iloc[split_idx:]
    
    rf_time = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_time.fit(X_train_time, y_train_time)
    auc_time = roc_auc_score(y_test_time, rf_time.predict_proba(X_test_time)[:, 1])
    
    # ----------------------------------------------------
    # 결과 출력
    # ----------------------------------------------------
    print("=" * 60)
    print(">>> Random Forest 성능 비교 <<<")
    print("=" * 60)
    print(f"[1] KAMP 가이드북 방식 (랜덤 섞기 - StratifiedShuffleSplit)")
    print(f"    -> ROC-AUC: {auc_rand:.4f}")
    print("-" * 60)
    print(f"[2] 올바른 시계열 방식 (순차 분할 - Chronological Split)")
    print(f"    -> ROC-AUC: {auc_time:.4f}")
    print("=" * 60)

if __name__ == "__main__":
    run_split_comparison()
