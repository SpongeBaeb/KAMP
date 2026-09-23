import pandas as pd
from sklearn.preprocessing import StandardScaler

def phase2_preprocess():
    print("Loading raw data for Phase 2...")
    xl = pd.ExcelFile('Welding Data Set_01.xlsx')
    df_raw = xl.parse('Raw data')
    
    df_raw['working time'] = pd.to_datetime(df_raw['working time']).dt.date
    
    features = ['Thickness 1(mm)', 'Thickness 2(mm)', 'weld force(bar)', 
                'weld current(kA)', 'weld Voltage(v)', 'weld time(ms)']
    df_clean = df_raw.dropna(subset=features).copy()
    
    initial_shape = df_clean.shape[0]
    
    # Filter out ALL calibration noise (anything > 4.0 bar)
    df_phase2 = df_clean[df_clean['weld force(bar)'] <= 4.0].copy()
    final_shape = df_phase2.shape[0]
    
    print(f"Removed {initial_shape - final_shape} noise (calibration) data points.")
    print(f"Remaining Pure Production Data: {final_shape}")
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_phase2[features])
    
    for i, col in enumerate(features):
        df_phase2[f"{col}_scaled"] = scaled_features[:, i]
        
    df_phase2.to_csv('phase2_preprocessed_raw.csv', index=False)
    print("Saved Phase 2 preprocessed data.")

if __name__ == "__main__":
    phase2_preprocess()
