import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import shap
import warnings
warnings.filterwarnings('ignore')

def run_failure_analysis():
    print("Loading data for Failure Mode Analysis...")
    df = pd.read_csv('data/moldset_labeled_cn7.csv')
    
    cols_to_drop = ['Unnamed: 0', 'Clamp_Open_Position']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    target = 'PassOrFail'
    features = [c for c in df.columns if c != target]
    
    X = df[features]
    y = df[target]
    
    # We use a single train/test split here for interpretability (make sure we have defects in test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    
    # Train XGBoost
    xgb = XGBClassifier(scale_pos_weight=scale_pos, random_state=42, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    
    # Predictions
    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)[:, 1]
    
    # Identify indices
    df_test = X_test.copy()
    df_test['True_Label'] = y_test
    df_test['Pred_Label'] = y_pred
    df_test['Prob'] = y_proba
    
    TP = df_test[(df_test['True_Label'] == 1) & (df_test['Pred_Label'] == 1)]
    FN = df_test[(df_test['True_Label'] == 1) & (df_test['Pred_Label'] == 0)]
    FP = df_test[(df_test['True_Label'] == 0) & (df_test['Pred_Label'] == 1)]
    
    print("\n--- Failure Mode Counts (Test Set) ---")
    print(f"True Positives (Correct Defect): {len(TP)}")
    print(f"False Negatives (Missed Defect): {len(FN)}")
    print(f"False Positives (False Alarm)  : {len(FP)}")
    
    # Global SHAP Analysis
    print("\nGenerating Global SHAP Analysis...")
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("Phase 3: Global Feature Importance (SHAP)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/03_shap_global.png', dpi=300)
    plt.close()
    
    # Function to generate waterfall
    def save_waterfall(index, title, filename):
        if len(index) == 0:
            print(f"No samples for {title}")
            return
            
        idx = index.index[0] # Take the first one
        idx_loc = X_test.index.get_loc(idx)
        
        # New API for waterfall
        sv = explainer(X_test)
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(sv[idx_loc], show=False, max_display=10)
        plt.title(f"{title} (Prob: {df_test.loc[idx, 'Prob']*100:.1f}%)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'plots/{filename}.png', dpi=300)
        plt.close()
        print(f"Saved {filename}.png")
        
    print("\nGenerating SHAP Waterfall Plots for Specific Scenarios...")
    save_waterfall(TP, "True Positive (Model Successfully Caught Defect)", "03_shap_TP")
    save_waterfall(FN, "False Negative (Model Missed This Defect - Failure Mode 1)", "03_shap_FN")
    save_waterfall(FP, "False Positive (Model False Alarm - Failure Mode 2)", "03_shap_FP")
    
    print("\nPhase 3 Complete.")

if __name__ == "__main__":
    run_failure_analysis()
