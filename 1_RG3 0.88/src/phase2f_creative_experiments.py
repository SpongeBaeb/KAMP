import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
import sys
from pathlib import Path

# Add src to path to import local modules
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from preprocess import load_raw, DROP_COLS
from train import make_model, NoveltyDetector, SEED
from evaluate import score

def run_experiments():
    print("Loading raw data...")
    lab, _, feats = load_raw()
    
    results = []
    
    # ----------------------------------------------------------------
    # Baseline on RAW data
    # ----------------------------------------------------------------
    print("\nRunning Baseline (Raw Data)...")
    X_base = lab[feats].to_numpy()
    y = lab['PassOrFail'].to_numpy()
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    praucs, rocaucs = [], []
    for tr, te in cv.split(X_base, y):
        model = make_model("HGB", "class_weight")
        model.fit(X_base[tr], y[tr])
        p = model.predict_proba(X_base[te])[:, 1]
        s = score(y[te], p)
        praucs.append(s['prauc'])
        rocaucs.append(s['roc_auc'])
        
    base_pr = np.mean(praucs)
    base_roc = np.mean(rocaucs)
    print(f"Baseline -> PR-AUC: {base_pr:.4f}, ROC-AUC: {base_roc:.4f}")
    results.append({"Experiment": "Baseline", "PR-AUC": base_pr, "ROC-AUC": base_roc})
    
    # ----------------------------------------------------------------
    # Exp 1: Virtual ID
    # ----------------------------------------------------------------
    print("\nRunning Exp 1 (Virtual ID)...")
    lab['Virtual_ID'] = lab.groupby(feats, sort=False).cumcount()
    X_exp1 = lab[feats + ['Virtual_ID']].to_numpy()
    
    praucs, rocaucs = [], []
    for tr, te in cv.split(X_exp1, y):
        model = make_model("HGB", "class_weight")
        model.fit(X_exp1[tr], y[tr])
        p = model.predict_proba(X_exp1[te])[:, 1]
        s = score(y[te], p)
        praucs.append(s['prauc'])
        rocaucs.append(s['roc_auc'])
        
    exp1_pr = np.mean(praucs)
    exp1_roc = np.mean(rocaucs)
    print(f"Exp 1 (Virtual ID) -> PR-AUC: {exp1_pr:.4f}, ROC-AUC: {exp1_roc:.4f}")
    results.append({"Experiment": "Virtual ID", "PR-AUC": exp1_pr, "ROC-AUC": exp1_roc})
    
    # ----------------------------------------------------------------
    # Exp 2: Time-Lag on Raw
    # ----------------------------------------------------------------
    print("\nRunning Exp 2 (Time-Lag on Raw)...")
    proc = [c for c in feats if c not in DROP_COLS]
    X_df = lab[proc].copy()
    
    X_df['Volatility_5'] = X_df.shift(1).rolling(5, min_periods=2).std().mean(axis=1).fillna(0)
    prev_mean = X_df.shift(1).rolling(5, min_periods=1).mean()
    X_df['Deviation_from_recent5'] = (X_df[proc] - prev_mean).abs().mean(axis=1).fillna(0)
    X_df['Change_from_prev'] = X_df[proc].diff().abs().mean(axis=1).fillna(0)
    
    X_exp2 = X_df.to_numpy()
    
    praucs, rocaucs = [], []
    for tr, te in cv.split(X_exp2, y):
        model = make_model("HGB", "class_weight")
        model.fit(X_exp2[tr], y[tr])
        p = model.predict_proba(X_exp2[te])[:, 1]
        s = score(y[te], p)
        praucs.append(s['prauc'])
        rocaucs.append(s['roc_auc'])
        
    exp2_pr = np.mean(praucs)
    exp2_roc = np.mean(rocaucs)
    print(f"Exp 2 (Time-Lag on Raw) -> PR-AUC: {exp2_pr:.4f}, ROC-AUC: {exp2_roc:.4f}")
    results.append({"Experiment": "Time-Lag on Raw", "PR-AUC": exp2_pr, "ROC-AUC": exp2_roc})
    
    # ----------------------------------------------------------------
    # Exp 3: Risk Zone Anomaly (Rolling 10)
    # ----------------------------------------------------------------
    print("\nRunning Exp 3 (Risk Zone Anomaly)...")
    X_exp3 = lab[feats].to_numpy()
    
    praucs, rocaucs = [], []
    for tr, te in cv.split(X_exp3, y):
        nd = NoveltyDetector("IsolationForest")
        nd.fit(X_exp3[tr], y[tr])
        
        # Predict on ALL data to maintain sequence for rolling window
        p_all = nd.predict_proba(X_exp3)[:, 1]
        p_series = pd.Series(p_all)
        risk_index = p_series.rolling(10, min_periods=1).mean().to_numpy()
        
        p_te = risk_index[te]
        s = score(y[te], p_te)
        praucs.append(s['prauc'])
        rocaucs.append(s['roc_auc'])
        
    exp3_pr = np.mean(praucs)
    exp3_roc = np.mean(rocaucs)
    print(f"Exp 3 (Risk Zone Anomaly) -> PR-AUC: {exp3_pr:.4f}, ROC-AUC: {exp3_roc:.4f}")
    results.append({"Experiment": "Risk Zone Anomaly", "PR-AUC": exp3_pr, "ROC-AUC": exp3_roc})

    # Save results
    out_dir = ROOT / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "p2f_creative_experiments.csv", index=False)
    print(f"\nSaved results to {out_dir / 'p2f_creative_experiments.csv'}")

if __name__ == "__main__":
    run_experiments()
