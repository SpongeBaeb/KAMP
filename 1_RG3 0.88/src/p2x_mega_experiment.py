import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
import time

try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from catboost import CatBoostClassifier
    EXTRA_MODELS = True
except ImportError:
    EXTRA_MODELS = False

ROOT = Path(__file__).resolve().parents[1]

def load_data():
    df = pd.read_csv(ROOT.parent / "①Molding" / "moldset_labeled_rg3.csv")
    df.rename(columns={"Unnamed: 0": "idx"}, inplace=True)
    df = df.sort_values('idx').reset_index(drop=True)
    return df

def feature_engineering(df):
    print("  -> Base features")
    feats = [c for c in df.columns if c not in ('idx', 'PassOrFail')]
    
    # Base: Virtual ID
    df['Virtual_ID'] = df.groupby(feats, sort=False).cumcount()
    df['Cumulative_Shots'] = df.index
    
    print("  -> Time-Series Rolling")
    for col in feats:
        if df[col].nunique() > 5:  # 연속형 변수만
            df[f'{col}_roll5_mean'] = df[col].rolling(5, min_periods=1).mean()
            df[f'{col}_roll5_std'] = df[col].rolling(5, min_periods=1).std().fillna(0)
            df[f'{col}_lag1'] = df[col].shift(1).bfill()
            df[f'{col}_lag2'] = df[col].shift(2).bfill()
            
    # 누수 없는 안전한 과거 정답 합계 (Defect Burst)
    df['Defects_in_last_10'] = df['PassOrFail'].shift(1).rolling(10, min_periods=1).sum().fillna(0)
    
    base_and_new_feats = [c for c in df.columns if c not in ('idx', 'PassOrFail')]
    X = df[base_and_new_feats]
    
    print("  -> Polynomial Features (Top 5 sensors)")
    top_5 = ['Max_Injection_Pressure', 'Barrel_Temperature_1', 'Cycle_Time', 'Max_Switch_Over_Pressure', 'Cushion_Position']
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_feats = poly.fit_transform(df[top_5])
    poly_df = pd.DataFrame(poly_feats, columns=poly.get_feature_names_out(top_5))
    poly_df = poly_df.drop(columns=top_5)
    X = pd.concat([X, poly_df], axis=1)
    
    print("  -> PCA Components")
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    pca = PCA(n_components=5, random_state=42)
    pca_feats = pca.fit_transform(X_scaled)
    for i in range(5):
        X[f'PCA_{i}'] = pca_feats[:, i]
        
    print("  -> KMeans Distances")
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans_dist = kmeans.fit_transform(X_scaled)
    for i in range(3):
        X[f'KMeans_dist_{i}'] = kmeans_dist[:, i]
        
    # 최종 결측치 처리
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
    return X, df['PassOrFail']

def feature_selection(X, y):
    print(f"  -> Selecting from {X.shape[1]} features...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X, y)
    
    # 상위 특성 선택
    selector = SelectFromModel(rf, prefit=True, threshold='median')
    X_sel = selector.transform(X)
    selected_cols = X.columns[selector.get_support()]
    print(f"  -> Selected {X_sel.shape[1]} features.")
    
    # Virtual ID가 강제로 포함되게 확인
    if 'Virtual_ID' not in selected_cols:
        print("  -> (Forcing Virtual_ID back into features)")
        X_sel = np.hstack([X_sel, X['Virtual_ID'].to_numpy().reshape(-1, 1)])
        
    return X_sel

def evaluate_models(X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    
    models = {
        'HGB': HistGradientBoostingClassifier(class_weight='balanced', max_iter=200, random_state=42),
        'RF': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    }
    
    if EXTRA_MODELS:
        models['LGBM'] = LGBMClassifier(class_weight='balanced', n_estimators=200, random_state=42, verbose=-1)
        models['XGB'] = XGBClassifier(scale_pos_weight=20, n_estimators=200, random_state=42, eval_metric='logloss')
        models['CatBoost'] = CatBoostClassifier(auto_class_weights='Balanced', iterations=200, random_state=42, verbose=0)
        
        # 앙상블 추가
        estimators = [(name, m) for name, m in models.items()]
        models['Voting_All'] = VotingClassifier(estimators=estimators, voting='soft')

    results = {}
    for name, model in models.items():
        roc_aucs = []
        for train_idx, test_idx in cv.split(X, y):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]
            
            model.fit(X_train, y_train)
            y_probs = model.predict_proba(X_test)[:, 1]
            roc_aucs.append(roc_auc_score(y_test, y_probs))
        
        results[name] = np.mean(roc_aucs)
        print(f"    - {name}: ROC-AUC {results[name]:.4f}")
        
    return results

def main():
    print("=== Phase 2-X Mega Experiment ===")
    
    print("\n1. Loading Data...")
    df = load_data()
    
    print("\n2. Mega Feature Engineering...")
    X, y = feature_engineering(df)
    
    print("\n3. Feature Selection...")
    X_sel = feature_selection(X, y)
    
    print("\n4. Model Evaluation & Ensembling...")
    evaluate_models(X_sel, y.to_numpy())
    
    print("\n=== Experiment Complete ===")

if __name__ == "__main__":
    main()
