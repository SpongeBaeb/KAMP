import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import shap

def run_phase4_analysis():
    print("Loading Pure Production Data...")
    df_prod = pd.read_csv('pure_production_data.csv')
    df_prod['working time'] = pd.to_datetime(df_prod['working time'])
    
    features = ['Thickness 1(mm)', 'Thickness 2(mm)', 'weld force(bar)', 
                'weld current(kA)', 'weld Voltage(v)', 'weld time(ms)']
    scaled_cols = [f"{col}_scaled" for col in features]
    
    X = df_prod[scaled_cols]
    
    print(f"Training Isolation Forest on {len(X)} pure production records...")
    # Using a conservative contamination rate. 
    # Real defects might be very rare, let's say 1% or auto. Let's use 0.01 (1%) for starters.
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    iso_forest.fit(X)
    
    df_prod['anomaly_score'] = iso_forest.decision_function(X)
    df_prod['is_anomaly'] = iso_forest.predict(X)
    
    # -1 means anomaly, 1 means normal
    df_prod['is_anomaly'] = df_prod['is_anomaly'].map({-1: 1, 1: 0})
    
    anomalies = df_prod[df_prod['is_anomaly'] == 1]
    print(f"Detected {len(anomalies)} true anomalies.")
    
    # ---------------------------------------------------------
    # 2. Cross-Validation with True Defects
    # ---------------------------------------------------------
    print("Loading True Daily Defects...")
    df_true = pd.read_csv('true_daily_defects.csv')
    df_true['date'] = pd.to_datetime(df_true['date'])
    
    # Aggregate our predicted anomalies by day
    daily_pred = anomalies.groupby(df_prod['working time'].dt.date).size().reset_index(name='predicted_defects')
    daily_pred['working time'] = pd.to_datetime(daily_pred['working time'])
    
    # Merge
    merged = pd.merge(df_true, daily_pred, left_on='date', right_on='working time', how='left')
    merged['predicted_defects'] = merged['predicted_defects'].fillna(0)
    
    print("\n--- Daily Defects Comparison ---")
    print(merged[['date', 'true_defects', 'predicted_defects']])
    
    correlation = merged['true_defects'].corr(merged['predicted_defects'])
    print(f"\nCorrelation between True Defects and Predicted Anomalies: {correlation:.4f}")
    
    # Plotting the comparison
    plt.figure(figsize=(10, 5))
    plt.plot(merged['date'], merged['true_defects'], label='True Defects', marker='o', color='red')
    plt.plot(merged['date'], merged['predicted_defects'], label='Predicted Anomalies', marker='x', color='blue', linestyle='--')
    plt.title('True Defects vs Predicted Anomalies (Phase 4)')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/phase4_defect_comparison.png')
    plt.close()
    
    # ---------------------------------------------------------
    # 3. XAI Analysis with SHAP
    # ---------------------------------------------------------
    print("Calculating SHAP values for anomalies...")
    # Use TreeExplainer
    explainer = shap.TreeExplainer(iso_forest)
    # Explain all anomalies to see what drove them
    X_anomalies = anomalies[scaled_cols]
    
    if len(X_anomalies) > 0:
        shap_values = explainer.shap_values(X_anomalies)
        
        # Plot summary
        plt.figure()
        shap.summary_plot(shap_values, X_anomalies, feature_names=features, show=False)
        plt.tight_layout()
        plt.savefig('plots/phase4_shap_summary.png')
        plt.close()
        print("Saved SHAP summary plot to 'plots/phase4_shap_summary.png'.")
        
        # Feature importance by mean absolute shap value
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = pd.DataFrame(list(zip(features, mean_abs_shap)), columns=['Feature', 'SHAP Importance']).sort_values('SHAP Importance', ascending=False)
        print("\n--- Feature Importance for True Anomalies ---")
        print(shap_importance)
        
        # Analyze the distribution of actual feature values for anomalies vs normal
        print("\n--- Mean Feature Values: Anomalies vs Normal ---")
        normal_data = df_prod[df_prod['is_anomaly'] == 0]
        compare_df = pd.DataFrame({
            'Normal_Mean': normal_data[features].mean(),
            'Anomaly_Mean': anomalies[features].mean()
        })
        print(compare_df)
    
    # Save results
    df_prod.to_csv('phase4_true_anomalies.csv', index=False)
    merged.to_csv('phase4_validation_results.csv', index=False)
    
if __name__ == "__main__":
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')
    run_phase4_analysis()
