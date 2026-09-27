import numpy as np
import pandas as pd
from pathlib import Path
import time
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import IsolationForest
from catboost import CatBoostClassifier

def p(msg): print(msg, flush=True)

ROOT = Path(__file__).resolve().parents[1]
data_dir = ROOT.parent / "\u2460Molding"

def main():
    p("="*60)
    p("  PHASE 7: THE GRANDMASTER APPROACH")
    p("="*60)
    start = time.time()
    
    # 1. Load Labeled Data
    df_rg3 = pd.read_csv(data_dir / "moldset_labeled_rg3.csv", index_col=0)
    df_cn7 = pd.read_csv(data_dir / "moldset_labeled_cn7.csv", index_col=0)
    df_rg3['machine'] = 1
    df_cn7['machine'] = 0
    labeled = pd.concat([df_rg3, df_cn7], ignore_index=True)
    labeled = labeled.drop(columns=['Clamp_Open_Position'], errors='ignore')
    
    sensor_cols = [c for c in labeled.columns if c not in ('PassOrFail', 'machine')]
    labeled['Virtual_ID'] = labeled.groupby(sensor_cols + ['machine'], sort=False).cumcount()
    
    # 2. Load Unlabeled Data for Semi-Supervised Learning
    p("Loading unlabeled data for anomaly detection...")
    try:
        un_rg3 = pd.read_csv(data_dir / "moldset_unlabeled_rg3.csv", index_col=0)
        un_cn7 = pd.read_csv(data_dir / "moldset_unlabeled_cn7.csv", index_col=0)
        un_rg3['machine'] = 1
        un_cn7['machine'] = 0
        unlabeled = pd.concat([un_rg3, un_cn7], ignore_index=True)
        unlabeled = unlabeled.drop(columns=['Clamp_Open_Position'], errors='ignore')
        has_unlabeled = True
    except Exception as e:
        p(f"Could not load unlabeled: {e}")
        has_unlabeled = False
        
    # 3. Domain Adaptation (Quantile Transformation) to fix Scale Collapse
    # We fit QuantileTransformer on Unlabeled, and transform both to bridge the domain gap
    qt = QuantileTransformer(output_distribution='normal', random_state=42)
    
    # Use top sensors for anomaly detection
    top_sensors = ['Max_Injection_Pressure', 'Max_Switch_Over_Pressure', 'Max_Back_Pressure',
                   'Cycle_Time', 'Injection_Time', 'Filling_Time', 'Plasticizing_Time']
                   
    labeled_anomaly_scores = np.zeros(len(labeled))
    
    if has_unlabeled:
        p("Training Isolation Forest on Unlabeled Data (Domain Adapted)...")
        # Ensure columns exist in unlabeled
        use_cols = [c for c in top_sensors if c in unlabeled.columns]
        
        # Transform Unlabeled
        unlabeled_scaled = qt.fit_transform(unlabeled[use_cols].fillna(0))
        
        # Train Isolation Forest
        iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42, n_jobs=-1)
        iso.fit(unlabeled_scaled)
        
        # Score Labeled Data (Domain Adapted)
        labeled_scaled = qt.transform(labeled[use_cols].fillna(0))
        # anomaly_score: lower is more anomalous. We negate it so higher = anomalous
        labeled_anomaly_scores = -iso.score_samples(labeled_scaled)
        
    labeled['Unsupervised_Anomaly_Score'] = labeled_anomaly_scores
    
    # 4. Physics & EMA Features (True Causal Drift)
    shots = labeled.drop_duplicates(subset=sensor_cols + ['machine']).copy()
    
    for machine_id in [0, 1]:
        mask = shots['machine'] == machine_id
        for col in top_sensors:
            if col in shots.columns:
                shots.loc[mask, f'{col}_EMA10'] = shots.loc[mask, col].ewm(span=10, adjust=False).mean()
                shots.loc[mask, f'{col}_EMA50'] = shots.loc[mask, col].ewm(span=50, adjust=False).mean()
                
    shots = shots.fillna(0)
    merge_cols = sensor_cols + ['machine']
    feat_df = shots.drop(columns=['PassOrFail'])
    
    final_df = pd.merge(labeled, feat_df, on=merge_cols, how='left')
    final_df['Virtual_ID'] = final_df.groupby(merge_cols, sort=False).cumcount()
    
    X = final_df.drop(columns=['PassOrFail'])
    y = final_df['PassOrFail']
    
    # 5. Model Training with Focal Loss Simulation
    p("Training CatBoost with heavy class balancing (Focal Loss approximation)...")
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    cat_features = ['Virtual_ID', 'machine']
    
    # Try different CatBoost variants
    models = {
        "CatBoost (Standard Balanced)": CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.03, 
            auto_class_weights='Balanced', cat_features=cat_features, verbose=0, random_state=42
        ),
        "CatBoost (Custom Pos Weight 50x)": CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.03, 
            scale_pos_weight=50, cat_features=cat_features, verbose=0, random_state=42
        )
    }
    
    for name, model in models.items():
        aucs = []
        for train_idx, test_idx in cv.split(X, y):
            Xtr, ytr = X.iloc[train_idx], y.iloc[train_idx]
            Xte, yte = X.iloc[test_idx], y.iloc[test_idx]
            
            model.fit(Xtr, ytr, cat_features=cat_features, verbose=False)
            probs = model.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(yte, probs))
            
        p(f"{name} ROC-AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")
        
    p(f"Completed in {time.time() - start:.1f}s")

if __name__ == "__main__":
    main()
