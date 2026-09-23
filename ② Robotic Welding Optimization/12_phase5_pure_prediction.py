import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

def run_phase5_analysis():
    print("Loading Pure Production Data...")
    df_prod = pd.read_csv('pure_production_data.csv')
    df_prod['working time'] = pd.to_datetime(df_prod['working time'])
    
    # ---------------------------------------------------------
    # 1. Date-based Masking (점검일 원천 차단)
    # ---------------------------------------------------------
    maintenance_days = ['2020-03-25', '2020-03-27', '2020-03-31', '2020-04-03']
    maintenance_dates = pd.to_datetime(maintenance_days).date
    
    # Filter out maintenance days entirely
    mask = ~df_prod['working time'].dt.date.isin(maintenance_dates)
    df_pure_prod = df_prod[mask].copy()
    
    print(f"Total data before masking: {len(df_prod)}")
    print(f"Total data after excluding maintenance days: {len(df_pure_prod)}")
    
    features = ['Thickness 1(mm)', 'Thickness 2(mm)', 'weld force(bar)', 
                'weld current(kA)', 'weld Voltage(v)', 'weld time(ms)']
    scaled_cols = [f"{col}_scaled" for col in features]
    X = df_pure_prod[scaled_cols]
    
    # ---------------------------------------------------------
    # 2. Train True Anomaly Detection Model
    # ---------------------------------------------------------
    print(f"\nTraining Isolation Forest on {len(X)} ultimate pure production records...")
    # Adjust contamination based on remaining data. Let's use 0.005 (0.5%) to find only the most extreme true defects.
    iso_forest = IsolationForest(n_estimators=100, contamination=0.005, random_state=42)
    iso_forest.fit(X)
    
    df_pure_prod['anomaly_score'] = iso_forest.decision_function(X)
    df_pure_prod['is_anomaly'] = iso_forest.predict(X)
    df_pure_prod['is_anomaly'] = df_pure_prod['is_anomaly'].map({-1: 1, 1: 0})
    
    anomalies = df_pure_prod[df_pure_prod['is_anomaly'] == 1]
    print(f"Detected {len(anomalies)} true anomalies on normal production days.")
    
    # ---------------------------------------------------------
    # 3. Validation with True Defects
    # ---------------------------------------------------------
    print("\nLoading True Daily Defects...")
    df_true = pd.read_csv('true_daily_defects.csv')
    df_true['date'] = pd.to_datetime(df_true['date']).dt.date
    
    # Only evaluate on the non-maintenance days!
    df_true_filtered = df_true[~df_true['date'].isin(maintenance_dates)].copy()
    
    daily_pred = anomalies.groupby(df_pure_prod['working time'].dt.date).size().reset_index(name='predicted_defects')
    daily_pred['working time'] = pd.to_datetime(daily_pred['working time']).dt.date
    
    merged = pd.merge(df_true_filtered, daily_pred, left_on='date', right_on='working time', how='left')
    merged['predicted_defects'] = merged['predicted_defects'].fillna(0)
    
    print("\n--- Daily Defects Comparison (Excluding Maintenance Days) ---")
    print(merged[['date', 'true_defects', 'predicted_defects']])
    
    # Calculate Correlation
    if len(merged) > 1:
        correlation = merged['true_defects'].corr(merged['predicted_defects'])
        print(f"\nCorrelation between True Defects and Predicted Anomalies: {correlation:.4f}")
    else:
        print("\nNot enough data points to calculate correlation.")
        
    # ---------------------------------------------------------
    # 4. Plotting
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(merged['date'].astype(str), merged['true_defects'], label='True Defects', marker='o', color='red')
    plt.plot(merged['date'].astype(str), merged['predicted_defects'], label='Predicted Anomalies', marker='x', color='blue', linestyle='--')
    plt.title('True Defects vs Predicted Anomalies (Maintenance Days Excluded)')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/phase5_final_prediction.png')
    plt.close()
    print("Saved final plot to 'plots/phase5_final_prediction.png'.")

if __name__ == "__main__":
    run_phase5_analysis()
