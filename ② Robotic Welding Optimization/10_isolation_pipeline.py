import pandas as pd
from sklearn.preprocessing import StandardScaler

def isolation_pipeline():
    print("Loading raw data for Data Isolation Pipeline...")
    xl = pd.ExcelFile('Welding Data Set_01.xlsx')
    df_raw = xl.parse('Raw data')
    
    # Ensure working time is datetime object
    df_raw['working time'] = pd.to_datetime(df_raw['working time']).dt.date
    
    features = ['Thickness 1(mm)', 'Thickness 2(mm)', 'weld force(bar)', 
                'weld current(kA)', 'weld Voltage(v)', 'weld time(ms)']
    
    # Drop rows with missing features
    df_clean = df_raw.dropna(subset=features).copy()
    initial_shape = df_clean.shape[0]
    
    # Define Isolation Rules
    # Rule 1: Full sweep test (> 2.8 bar)
    rule1_condition = df_clean['weld force(bar)'] > 2.8
    
    # Rule 2: Multi-dimensional extreme test 
    rule2_condition = (df_clean['weld Voltage(v)'] <= 2.62) & \
                      (df_clean['weld current(kA)'] >= 14.85) & \
                      (df_clean['weld time(ms)'] <= 70.5)
                      
    # Combine conditions (OR)
    is_maintenance = rule1_condition | rule2_condition
    
    # Separate data
    df_maintenance = df_clean[is_maintenance].copy()
    df_production = df_clean[~is_maintenance].copy()
    
    print(f"Total valid raw data points: {initial_shape}")
    print(f"Found and isolated {df_maintenance.shape[0]} maintenance/test data points.")
    print(f"Remaining Pure Production Data: {df_production.shape[0]}")
    
    # Standard scale the pure production data features for further anomaly detection
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_production[features])
    
    for i, col in enumerate(features):
        df_production[f"{col}_scaled"] = scaled_features[:, i]
        
    # Standard scale maintenance data using the same scaler (optional, but good for consistent comparison)
    if not df_maintenance.empty:
        scaled_features_maint = scaler.transform(df_maintenance[features])
        for i, col in enumerate(features):
            df_maintenance[f"{col}_scaled"] = scaled_features_maint[:, i]
    
    # Save isolated datasets
    df_production.to_csv('pure_production_data.csv', index=False)
    df_maintenance.to_csv('isolated_maintenance_data.csv', index=False)
    
    print("Saved 'pure_production_data.csv' and 'isolated_maintenance_data.csv'.")
    
    # Perform basic assertion to ensure rule conditions are empty in production data
    assert df_production[df_production['weld force(bar)'] > 2.8].empty, "Rule 1 data still in production set!"
    assert df_production[(df_production['weld Voltage(v)'] <= 2.62) & 
                         (df_production['weld current(kA)'] >= 14.85) & 
                         (df_production['weld time(ms)'] <= 70.5)].empty, "Rule 2 data still in production set!"
    
    print("Verification passed: Production data is clean.")

if __name__ == "__main__":
    isolation_pipeline()
