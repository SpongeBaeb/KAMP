import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

def run_prescriptive_ai():
    print("Loading Pure Production Data for Prescriptive AI...")
    df_prod = pd.read_excel('Welding Data Set_01.xlsx')
    df_prod['working time'] = pd.to_datetime(df_prod['working time'])
    
    # Filter maintenance days
    maintenance_days = ['2020-03-25', '2020-03-27', '2020-03-31', '2020-04-03']
    maintenance_dates = pd.to_datetime(maintenance_days).date
    df_pure = df_prod[~df_prod['working time'].dt.date.isin(maintenance_dates)].copy().reset_index(drop=True)
    
    # Load true daily defects for Isolation Forest Top-K
    df_true = pd.read_csv('true_daily_defects.csv')
    df_true['date'] = pd.to_datetime(df_true['date']).dt.date
    true_defects_dict = dict(zip(df_true['date'], df_true['true_defects']))
    df_pure['date'] = df_pure['working time'].dt.date
    
    base_features = ['weld force(bar)', 'weld current(kA)', 'weld Voltage(v)', 'weld time(ms)']
    
    # Extract Pseudo-labels using Base Isolation Forest (as it was our most reliable base for XAI)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_base = scaler.fit_transform(df_pure[base_features])
    
    iso_base = IsolationForest(n_estimators=100, random_state=42)
    iso_base.fit(scaled_base)
    df_pure['anomaly_score'] = iso_base.decision_function(scaled_base)
    
    df_pure['is_defect'] = 0
    for date, group in df_pure.groupby('date'):
        if date in true_defects_dict and true_defects_dict[date] > 0:
            k = true_defects_dict[date]
            top_k_idx = group.nsmallest(k, 'anomaly_score').index
            df_pure.loc[top_k_idx, 'is_defect'] = 1
            
    X = df_pure[base_features]
    y = df_pure['is_defect']
    
    # Train the core XGBoost model for Explainability
    xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
    xgb.fit(X, y)
    
    print("\n--- 1. Explainable AI (SHAP) ---")
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("Phase 15: Global SHAP Feature Importance", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/phase15_shap_summary.png', dpi=300)
    plt.close()
    print("SHAP Summary plot saved to 'plots/phase15_shap_summary.png'")
    
    print("\n--- 2. Counterfactual Prescriptive Optimizer ---")
    # Find a specific defect
    defect_indices = df_pure[df_pure['is_defect'] == 1].index
    if len(defect_indices) == 0:
        print("No defects found. Exiting.")
        return
        
    target_idx = defect_indices[0]
    original_row = X.iloc[[target_idx]].copy()
    original_prob = xgb.predict_proba(original_row)[0][1]
    
    print(f"\n[DETECTED DEFECT] Index: {target_idx}, Date: {df_pure.iloc[target_idx]['working time']}")
    print(f"Original Sensor Values:")
    for col in base_features:
        print(f"  - {col}: {original_row[col].values[0]:.3f}")
    print(f"Defect Probability: {original_prob*100:.2f}% (Threshold: 50.00%)")
    
    # Optimization loop (Grid Search Counterfactual)
    print("\n[OPTIMIZER] Calculating minimal changes to prevent defect...")
    
    # Define search space (+/- 10% in small increments)
    deltas = {
        'weld force(bar)': np.linspace(-0.5, 0.5, 11),
        'weld current(kA)': np.linspace(-0.5, 0.5, 11),
        'weld Voltage(v)': np.linspace(-0.5, 0.5, 11),
        'weld time(ms)': np.linspace(-10, 10, 11)
    }
    
    best_prescription = None
    min_prob = original_prob
    
    import itertools
    keys = list(deltas.keys())
    combinations = list(itertools.product(*(deltas[k] for k in keys)))
    
    for combo in combinations:
        temp_row = original_row.copy()
        for i, key in enumerate(keys):
            temp_row[key] += combo[i]
            
        prob = xgb.predict_proba(temp_row)[0][1]
        if prob < 0.5 and prob < min_prob:
            min_prob = prob
            best_prescription = dict(zip(keys, combo))
            
    if best_prescription is None:
        print("Could not find a realistic prescription to flip the prediction.")
    else:
        print("\n[WINNER] [ACTIONABLE PRESCRIPTION GENERATED]")
        print("To prevent this defect, apply the following adjustments:")
        for col, change in best_prescription.items():
            if change != 0:
                direction = "INCREASE" if change > 0 else "DECREASE"
                print(f"  -> {direction} {col} by {abs(change):.3f}")
        
        print(f"\nSimulated Defect Probability after Prescription: {min_prob*100:.2f}%")
        
    print("\nPhase 15 (Prescriptive AI) Complete. Ready for Hackathon Grand Finale.")

if __name__ == "__main__":
    run_prescriptive_ai()
