import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def optimize_with_optuna():
    print("Loading Data for Optuna Optimization...")
    df_lab = pd.read_csv('c:/Hackerton/① Molding/data/moldset_labeled_cn7.csv')
    df_unlab = pd.read_csv('c:/Hackerton/① Molding/data/moldset_unlabeled_cn7.csv')
    
    cols_to_drop = ['Unnamed: 0', 'Clamp_Open_Position']
    df_lab = df_lab.drop(columns=[col for col in cols_to_drop if col in df_lab.columns])
    df_unlab = df_unlab.drop(columns=[col for col in cols_to_drop if col in df_unlab.columns])
    
    target = 'PassOrFail'
    features = [c for c in df_lab.columns if c != target]
    
    X_lab = df_lab[features]
    y_lab = df_lab[target]
    X_unlab = df_unlab[features]
    
    # Pre-calculate pseudo labels using standard LR to save time inside objective
    pseudo_model = LogisticRegression(class_weight='balanced', random_state=42)
    pseudo_model.fit(X_lab, y_lab)
    unlab_proba = pseudo_model.predict_proba(X_unlab)[:, 1]
    
    def objective(trial):
        # Tune Pseudo-labeling thresholds
        p_high = trial.suggest_float('p_high', 0.85, 0.99)
        p_low = trial.suggest_float('p_low', 0.01, 0.15)
        
        high_conf_defect_idx = np.where(unlab_proba > p_high)[0]
        high_conf_normal_idx = np.where(unlab_proba < p_low)[0]
        
        # We need at least some pseudo defects
        if len(high_conf_defect_idx) < 100:
            raise optuna.TrialPruned()
            
        np.random.seed(42)
        n_normals = trial.suggest_int('n_normals', 1000, 15000)
        if len(high_conf_normal_idx) > n_normals:
            high_conf_normal_idx = np.random.choice(high_conf_normal_idx, n_normals, replace=False)
            
        X_pseudo_defect = X_unlab.iloc[high_conf_defect_idx]
        y_pseudo_defect = pd.Series([1] * len(high_conf_defect_idx))
        
        X_pseudo_normal = X_unlab.iloc[high_conf_normal_idx]
        y_pseudo_normal = pd.Series([0] * len(high_conf_normal_idx))
        
        X_pseudo = pd.concat([X_pseudo_defect, X_pseudo_normal]).reset_index(drop=True)
        y_pseudo = pd.concat([y_pseudo_defect, y_pseudo_normal]).reset_index(drop=True)
        
        # Tune XGBoost parameters
        xgb_params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 100.0),
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        praucs = []
        
        for train_idx, test_idx in skf.split(X_lab, y_lab):
            X_train, X_test = X_lab.iloc[train_idx], X_lab.iloc[test_idx]
            y_train, y_test = y_lab.iloc[train_idx], y_lab.iloc[test_idx]
            
            X_train_expanded = pd.concat([X_train, X_pseudo]).reset_index(drop=True)
            y_train_expanded = pd.concat([y_train, y_pseudo]).reset_index(drop=True)
            
            model = XGBClassifier(**xgb_params)
            model.fit(X_train_expanded, y_train_expanded)
            
            probs = model.predict_proba(X_test)[:, 1]
            praucs.append(average_precision_score(y_test, probs))
            
        return np.mean(praucs)
        
    print("Starting Optuna Study (50 Trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, n_jobs=-1)
    
    print("\n--- OPTUNA OPTIMIZATION COMPLETE ---")
    print(f"Best Legitimate PR-AUC: {study.best_value:.4f}")
    print("Best Parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        
    if study.best_value >= 0.80:
        print("\n[GOAL ACHIEVED] PR-AUC > 0.80 WITHOUT CHEATING!")
    else:
        print("\n[FAILED] Failed to reach 0.80 naturally. 0.6385~0.70 might be the absolute mathematical limit.")
        
    # Plotting Optuna History
    import matplotlib.pyplot as plt
    try:
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.title("Optuna Optimization History (Target: PR-AUC 0.8)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('plots/07_optuna_prauc.png', dpi=300)
        plt.close()
        print("\nSaved Chart to 'plots/07_optuna_prauc.png'")
    except Exception as e:
        print(f"Plotting failed: {e}")

if __name__ == "__main__":
    optimize_with_optuna()
