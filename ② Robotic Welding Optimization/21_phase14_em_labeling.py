import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def run_em_labeling():
    print("Loading Pure Production Data...")
    df_prod = pd.read_csv('pure_production_data.csv')
    df_prod['working time'] = pd.to_datetime(df_prod['working time'])
    maintenance_days = ['2020-03-25', '2020-03-27', '2020-03-31', '2020-04-03']
    maintenance_dates = pd.to_datetime(maintenance_days).date
    df_pure = df_prod[~df_prod['working time'].dt.date.isin(maintenance_dates)].copy().reset_index(drop=True)
    
    df_true = pd.read_csv('true_daily_defects.csv')
    df_true['date'] = pd.to_datetime(df_true['date']).dt.date
    true_defects_dict = dict(zip(df_true['date'], df_true['true_defects']))
    df_pure['date'] = df_pure['working time'].dt.date
    
    base_features = ['weld force(bar)', 'weld current(kA)', 'weld Voltage(v)', 'weld time(ms)']
    
    # 1. Initialize Labels with Isolation Forest
    print("\n[Iteration 0] Initializing labels with Isolation Forest...")
    iso_base = IsolationForest(n_estimators=100, random_state=42)
    # Using unscaled for speed, trees are scale-invariant but IF uses distance, so let's scale it.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_base = scaler.fit_transform(df_pure[base_features])
    iso_base.fit(scaled_base)
    df_pure['anomaly_score'] = iso_base.decision_function(scaled_base)
    
    df_pure['label'] = 0
    for date, group in df_pure.groupby('date'):
        if date in true_defects_dict and true_defects_dict[date] > 0:
            k = true_defects_dict[date]
            top_k_idx = group.nsmallest(k, 'anomaly_score').index
            df_pure.loc[top_k_idx, 'label'] = 1
            
    X = df_pure[base_features]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    best_labels = df_pure['label'].copy()
    highest_f1 = 0
    f1_history = []
    
    # EM Loop
    max_iters = 5
    for iteration in range(1, max_iters + 1):
        print(f"\n--- EM Iteration {iteration} ---")
        y_current = df_pure['label'].copy()
        
        # Train XGBoost and get Out-Of-Fold probabilities
        oof_preds = np.zeros(len(df_pure))
        fold_f1s = []
        
        for train_idx, test_idx in skf.split(X, y_current):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_current.iloc[train_idx], y_current.iloc[test_idx]
            
            xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss')
            xgb.fit(X_train, y_train)
            
            # Predict probabilities
            probs = xgb.predict_proba(X_test)[:, 1]
            oof_preds[test_idx] = probs
            
            # Temporary evaluation for monitoring
            preds = (probs > 0.5).astype(int)
            fold_f1s.append(f1_score(y_test, preds))
            
        avg_f1 = np.mean(fold_f1s)
        f1_history.append(avg_f1)
        print(f"Current Label Learnability F1-Score: {avg_f1:.4f}")
        
        if avg_f1 > highest_f1:
            highest_f1 = avg_f1
            best_labels = y_current.copy()
            
        # M-Step: Re-assign labels based on Out-Of-Fold probabilities
        df_pure['new_prob'] = oof_preds
        df_pure['new_label'] = 0
        
        changes = 0
        for date, group in df_pure.groupby('date'):
            if date in true_defects_dict and true_defects_dict[date] > 0:
                k = true_defects_dict[date]
                # Pick the top K highest probabilities
                top_k_idx = group.nlargest(k, 'new_prob').index
                df_pure.loc[top_k_idx, 'new_label'] = 1
                
                # Check changes
                old_labels = df_pure.loc[group.index, 'label']
                new_labels = df_pure.loc[group.index, 'new_label']
                changes += (old_labels != new_labels).sum()
                
        print(f"Labels changed for {changes // 2} records (swaps).")
        
        if changes == 0:
            print("Labels converged!")
            break
            
        df_pure['label'] = df_pure['new_label']
        
    print(f"\n[WINNER] EM Labeling Completed. Highest achievable F1-Score (Learnability): {highest_f1:.4f}")
    
    # Plotting EM progression
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=range(len(f1_history)), y=f1_history, marker='o', linewidth=2)
    plt.title("Expectation-Maximization (EM) Label Refinement", fontsize=14, fontweight='bold')
    plt.xlabel("EM Iteration", fontsize=12)
    plt.ylabel("Out-of-Fold F1-Score", fontsize=12)
    plt.xticks(range(len(f1_history)), labels=[f"Iter {i}" for i in range(len(f1_history))])
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('plots/phase14_em_labeling_evolution.png', dpi=300)
    plt.close()
    
    print("EM Evolution chart saved to 'plots/phase14_em_labeling_evolution.png'")
    
if __name__ == "__main__":
    run_em_labeling()
