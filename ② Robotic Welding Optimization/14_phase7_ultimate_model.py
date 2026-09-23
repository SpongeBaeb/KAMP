import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def run_phase7_ultimate():
    print("Loading Data...")
    df_prod = pd.read_csv('pure_production_data.csv')
    df_prod['working time'] = pd.to_datetime(df_prod['working time'])
    
    # 1. Filter out maintenance days
    maintenance_days = ['2020-03-25', '2020-03-27', '2020-03-31', '2020-04-03']
    maintenance_dates = pd.to_datetime(maintenance_days).date
    
    mask = ~df_prod['working time'].dt.date.isin(maintenance_dates)
    df_pure = df_prod[mask].copy().reset_index(drop=True)
    
    # Base Features
    base_features = ['weld force(bar)', 'weld current(kA)', 'weld Voltage(v)', 'weld time(ms)']
    
    # 2. FEATURE ENGINEERING: Rolling Windows (Time-Series)
    print("Generating Rolling Sequence Features...")
    for col in base_features:
        df_pure[f"{col}_roll_mean_5"] = df_pure[col].rolling(window=5, min_periods=1).mean()
        df_pure[f"{col}_roll_std_5"] = df_pure[col].rolling(window=5, min_periods=1).std().fillna(0)
        df_pure[f"{col}_diff_prev"] = df_pure[col].diff().fillna(0)
        
    all_features = [c for c in df_pure.columns if any(bf in c for bf in base_features) and not c.endswith('_scaled')]
    print(f"Total features expanded to: {len(all_features)}")
    
    # 3. Pseudo-Labeling via Isolation Forest (Same as Phase 6 to get labels)
    scaled_cols = [f"{col}_scaled" for col in base_features]
    iso_forest = IsolationForest(n_estimators=100, random_state=42)
    iso_forest.fit(df_pure[scaled_cols])
    df_pure['anomaly_score'] = iso_forest.decision_function(df_pure[scaled_cols])
    
    df_true = pd.read_csv('true_daily_defects.csv')
    df_true['date'] = pd.to_datetime(df_true['date']).dt.date
    true_defects_dict = dict(zip(df_true['date'], df_true['true_defects']))
    
    df_pure['is_defect'] = 0
    df_pure['date'] = df_pure['working time'].dt.date
    
    for date, group in df_pure.groupby('date'):
        if date in true_defects_dict:
            k = true_defects_dict[date]
            if k > 0:
                top_k_indices = group.nsmallest(k, 'anomaly_score').index
                df_pure.loc[top_k_indices, 'is_defect'] = 1
                
    X = df_pure[all_features]
    y = df_pure['is_defect']
    
    print("\n--- Model Evaluation with SMOTE (5-Fold CV) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Tuned XGBoost
    clf = XGBClassifier(
        n_estimators=150, 
        max_depth=5, 
        learning_rate=0.05, 
        # Since we use SMOTE, classes will be balanced in train set. No need for scale_pos_weight
        random_state=42, 
        eval_metric='logloss'
    )
    
    f1_scores, precisions, recalls = [], [], []
    smote = SMOTE(k_neighbors=4, random_state=42) # k=4 because we have very few minority samples
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # APPLY SMOTE ONLY ON TRAIN DATA
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        
        clf.fit(X_train_sm, y_train_sm)
        y_pred = clf.predict(X_test)
        
        f1_scores.append(f1_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        
    print(f"Average Precision: {np.mean(precisions):.4f}")
    print(f"Average Recall: {np.mean(recalls):.4f}")
    print(f"Average F1-Score: {np.mean(f1_scores):.4f}")
    
    # 4. Final Model Training on ALL DATA (SMOTE applied to all)
    X_sm, y_sm = smote.fit_resample(X, y)
    clf.fit(X_sm, y_sm)
    
    importances = clf.feature_importances_
    importance_df = pd.DataFrame({'Feature': all_features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    
    print("\n--- Top 10 Feature Importances (Phase 7) ---")
    print(importance_df.head(10))
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), color='dodgerblue')
    plt.title('Top 10 Feature Importances (Rolling + SMOTE)')
    plt.tight_layout()
    plt.savefig('plots/phase7_feature_importance.png')
    plt.close()
    
    print("\nPhase 7 Optimization Completed. Results saved to 'plots/phase7_feature_importance.png'")

if __name__ == "__main__":
    run_phase7_ultimate()
