import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

cities = {
    'Seoul': {'lat': 37.5665, 'lon': 126.9780},
    'Ulsan': {'lat': 35.5384, 'lon': 129.3114},
    'Changwon': {'lat': 35.2279, 'lon': 128.6811},
    'Gumi': {'lat': 36.1195, 'lon': 128.3445},
    'Pohang': {'lat': 36.0190, 'lon': 129.3435},
    'Busan': {'lat': 35.1796, 'lon': 129.0756},
    'Incheon': {'lat': 37.4563, 'lon': 126.7052},
    'Gwangju': {'lat': 35.1595, 'lon': 126.8526}
}

def fetch_weather(lat, lon):
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date=2020-03-23&end_date=2020-04-16&hourly=temperature_2m,relative_humidity_2m&timezone=Asia%2FSeoul"
    res = requests.get(url).json()
    
    df_w = pd.DataFrame({
        'time': pd.to_datetime(res['hourly']['time']),
        'external_temp': res['hourly']['temperature_2m'],
        'external_humidity': res['hourly']['relative_humidity_2m']
    })
    
    # Fill missing values if any
    df_w.ffill(inplace=True)
    df_w.bfill(inplace=True)
    
    # Create join keys
    df_w['date'] = df_w['time'].dt.date
    df_w['hour'] = df_w['time'].dt.hour
    
    return df_w

def run_phase11_real_weather():
    print("Loading Pure Production Data & Extracting Pseudo-Labels...")
    df_prod = pd.read_csv('pure_production_data.csv')
    df_prod['working time'] = pd.to_datetime(df_prod['working time'])
    
    maintenance_days = ['2020-03-25', '2020-03-27', '2020-03-31', '2020-04-03']
    maintenance_dates = pd.to_datetime(maintenance_days).date
    
    mask = ~df_prod['working time'].dt.date.isin(maintenance_dates)
    df_pure = df_prod[mask].copy().reset_index(drop=True)
    
    base_features = ['weld force(bar)', 'weld current(kA)', 'weld Voltage(v)', 'weld time(ms)']
    scaled_cols = [f"{col}_scaled" for col in base_features]
    
    iso_forest = IsolationForest(n_estimators=100, random_state=42)
    iso_forest.fit(df_pure[scaled_cols])
    df_pure['anomaly_score'] = iso_forest.decision_function(df_pure[scaled_cols])
    
    df_true = pd.read_csv('true_daily_defects.csv')
    df_true['date'] = pd.to_datetime(df_true['date']).dt.date
    true_defects_dict = dict(zip(df_true['date'], df_true['true_defects']))
    
    df_pure['is_defect'] = 0
    df_pure['date'] = df_pure['working time'].dt.date
    df_pure['hour'] = df_pure['working time'].dt.hour
    
    for date, group in df_pure.groupby('date'):
        if date in true_defects_dict:
            k = true_defects_dict[date]
            if k > 0:
                top_k_indices = group.nsmallest(k, 'anomaly_score').index
                df_pure.loc[top_k_indices, 'is_defect'] = 1

    city_results = {}
    print("\nStarting Open-Meteo API Fetch and Geographic Optimization...")
    
    for city_name, coords in cities.items():
        print(f"[{city_name}] Fetching real historical weather data from Open-Meteo...")
        df_weather = fetch_weather(coords['lat'], coords['lon'])
        
        # Merge weather with pure data
        df_merged = pd.merge(df_pure, df_weather[['date', 'hour', 'external_temp', 'external_humidity']], 
                             on=['date', 'hour'], how='left')
        
        # Feature Engineering
        df_merged['humidity_weld_time_interaction'] = df_merged['external_humidity'] * df_merged['weld time(ms)']
        df_merged['force_temp_ratio'] = df_merged['weld force(bar)'] / (df_merged['external_temp'] + 1)
        
        for col in base_features:
            df_merged[f"{col}_roll_mean_5"] = df_merged[col].rolling(window=5, min_periods=1).mean()
            df_merged[f"{col}_roll_std_5"] = df_merged[col].rolling(window=5, min_periods=1).std().fillna(0)
            df_merged[f"{col}_diff_prev"] = df_merged[col].diff().fillna(0)
            
        all_features = [c for c in df_merged.columns if (any(bf in c for bf in base_features) or 'external' in c or 'interaction' in c or 'ratio' in c) and not c.endswith('_scaled')]
        
        X = df_merged[all_features]
        y = df_merged['is_defect']
        
        # We use a fast XGBoost to rank the cities accurately but quickly.
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        xgb = XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, eval_metric='logloss')
        smote = SMOTE(k_neighbors=3, random_state=42)
        
        f1_scores = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Anti-Leakage Holdout for Threshold
            X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=42)
            X_tr_sub_sm, y_tr_sub_sm = smote.fit_resample(X_tr_sub, y_tr_sub)
            
            xgb.fit(X_tr_sub_sm, y_tr_sub_sm)
            y_val_sub_proba = xgb.predict_proba(X_val_sub)[:, 1]
            prec_tr, rec_tr, thresh_tr = precision_recall_curve(y_val_sub, y_val_sub_proba)
            f1_array_tr = 2 * (prec_tr * rec_tr) / (prec_tr + rec_tr + 1e-8)
            best_idx_tr = np.argmax(f1_array_tr)
            best_thresh = thresh_tr[best_idx_tr] if best_idx_tr < len(thresh_tr) else 0.5
            
            # Final train for this fold
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
            xgb.fit(X_train_sm, y_train_sm)
            y_proba = xgb.predict_proba(X_test)[:, 1]
            
            y_pred = (y_proba >= best_thresh).astype(int)
            f1_scores.append(f1_score(y_test, y_pred))
            
        avg_f1 = np.mean(f1_scores)
        print(f" -> {city_name} Real Weather F1-Score: {avg_f1:.4f}")
        city_results[city_name] = avg_f1

    # Sort results
    sorted_cities = sorted(city_results.items(), key=lambda x: x[1], reverse=True)
    best_city, best_f1 = sorted_cities[0]
    
    print(f"\n[WINNER] The Optimal Factory Location is: {best_city} (F1-Score: {best_f1:.4f})")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    cities_names = [x[0] for x in sorted_cities]
    f1_vals = [x[1] for x in sorted_cities]
    
    colors = ['red' if x == best_city else 'steelblue' for x in cities_names]
    sns.barplot(x=cities_names, y=f1_vals, palette=colors)
    plt.title("Geographic AI Performance: Finding the Factory Location\n(Based on Real 2020 Spring Weather Data)", fontsize=14, fontweight='bold')
    plt.ylabel("Unbiased F1-Score (XGBoost)", fontsize=12)
    plt.xlabel("Major Industrial Cities in South Korea", fontsize=12)
    plt.ylim(0, max(f1_vals) + 0.1)
    
    for i, v in enumerate(f1_vals):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('plots/phase11_real_weather_search.png', dpi=300)
    plt.close()
    
    print("Optimization Chart saved to 'plots/phase11_real_weather_search.png'")

if __name__ == "__main__":
    run_phase11_real_weather()
