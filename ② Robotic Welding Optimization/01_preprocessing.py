import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    print("Loading data...")
    xl = pd.ExcelFile('Welding Data Set_01.xlsx')
    df_raw = xl.parse('Raw data')
    df_res = xl.parse('result')
    
    # 1. Process True Daily Defect Counts
    # Result sheet has daily defect counts by type. We just need total defects per day.
    df_res['working time'] = pd.to_datetime(df_res['working time']).dt.date
    daily_defects = df_res.groupby('working time')['defect'].sum().reset_index()
    daily_defects.columns = ['date', 'true_defects']
    daily_defects.to_csv('true_daily_defects.csv', index=False)
    print("True Daily Defects Extracted:")
    print(daily_defects)
    
    # 2. Process Raw Data
    print(f"Initial raw data shape: {df_raw.shape}")
    
    # Ensure working time is datetime
    df_raw['working time'] = pd.to_datetime(df_raw['working time']).dt.date
    
    # Drop rows with missing feature values
    features = ['Thickness 1(mm)', 'Thickness 2(mm)', 'weld force(bar)', 
                'weld current(kA)', 'weld Voltage(v)', 'weld time(ms)']
    
    df_clean = df_raw.dropna(subset=features).copy()
    print(f"Data shape after dropping missing features: {df_clean.shape}")
    
    # Optional: Winsorization (cap extreme outliers)
    # However, since this is anomaly detection, we MIGHT want to keep outliers!
    # Extreme values could be the defects. Let's just standard scale them.
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_clean[features])
    
    for i, col in enumerate(features):
        df_clean[f"{col}_scaled"] = scaled_features[:, i]
        
    df_clean.to_csv('preprocessed_raw.csv', index=False)
    print("Preprocessed data saved to preprocessed_raw.csv")

if __name__ == "__main__":
    preprocess_data()
