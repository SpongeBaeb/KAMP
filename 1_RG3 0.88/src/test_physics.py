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
    combined['Virtual_ID'] = combined.groupby(sensor_cols + ['machine'], sort=False).cumcount()
    
    # PHYSICS-BASED FEATURES
    # 1. Stroke / Distance
    combined['Stroke_Length'] = combined['Plasticizing_Position'] - combined['Cushion_Position']
    
    # 2. Velocity / Flow
    # Add small epsilon to avoid div zero
    combined['Injection_Velocity'] = combined['Stroke_Length'] / (combined['Injection_Time'] + 1e-6)
    
    # 3. Energy / Work
    combined['Injection_Work'] = combined['Max_Injection_Pressure'] * combined['Injection_Time']
    combined['Plasticizing_Work'] = combined['Average_Back_Pressure'] * combined['Plasticizing_Time']
    
    # 4. Pressure Gradients & Stability
    combined['Pressure_Drop'] = combined['Max_Injection_Pressure'] - combined['Max_Switch_Over_Pressure']
    combined['Back_Pressure_Var'] = combined['Max_Back_Pressure'] - combined['Average_Back_Pressure']
    
    # 5. Temperature Gradients
    combined['Barrel_Temp_Mean'] = combined[['Barrel_Temperature_1', 'Barrel_Temperature_2', 'Barrel_Temperature_3', 
                                            'Barrel_Temperature_4', 'Barrel_Temperature_5', 'Barrel_Temperature_6']].mean(axis=1)
    combined['Mold_Temp_Mean'] = combined[['Mold_Temperature_3', 'Mold_Temperature_4']].mean(axis=1)
    combined['Cooling_Gradient'] = combined['Barrel_Temp_Mean'] - combined['Mold_Temp_Mean']
    
    # 6. Time Ratios
    combined['Injection_Ratio'] = combined['Injection_Time'] / (combined['Cycle_Time'] + 1e-6)
    
    X = combined.drop(columns=['PassOrFail'])
    y = combined['PassOrFail']
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    models = [
        ("ExtraTrees", ExtraTreesClassifier(n_estimators=300, max_depth=8, max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1)),
        ("CatBoost", CatBoostClassifier(iterations=300, depth=6, learning_rate=0.03, auto_class_weights='Balanced', verbose=0, random_state=42))
    ]
    
    for name, model in models:
        aucs = []
        for train_idx, test_idx in cv.split(X, y):
            Xtr, ytr = X.iloc[train_idx], y.iloc[train_idx]
            Xte, yte = X.iloc[test_idx], y.iloc[test_idx]
            model.fit(Xtr, ytr)
            probs = model.predict_proba(Xte)[:, 1]
            aucs.append(roc_auc_score(yte, probs))
            
        p(f"{name} with Physics Features ROC-AUC: {np.mean(aucs):.4f} (+/- {np.std(aucs):.4f})")

if __name__ == "__main__":
    main()
