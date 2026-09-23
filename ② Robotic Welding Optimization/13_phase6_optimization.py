import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    print("XGBoost not found. Using Random Forest instead.")
    HAS_XGB = False

def run_phase6_optimization():
    print("Loading Data...")
    df_prod = pd.read_csv('pure_production_data.csv')
    df_prod['working time'] = pd.to_datetime(df_prod['working time'])
    
    # 1. Filter out maintenance days
    maintenance_days = ['2020-03-25', '2020-03-27', '2020-03-31', '2020-04-03']
    maintenance_dates = pd.to_datetime(maintenance_days).date
    
    mask = ~df_prod['working time'].dt.date.isin(maintenance_dates)
    df_pure = df_prod[mask].copy()
    
    # We drop Thickness 1 and 2 as they have zero variance and no SHAP impact
    features = ['weld force(bar)', 'weld current(kA)', 'weld Voltage(v)', 'weld time(ms)']
    scaled_cols = [f"{col}_scaled" for col in features]
    
    # Prepare features for anomaly scoring
    X_iso = df_pure[scaled_cols]
    
    # 2. Get Anomaly Scores using IF
    print("\nExtracting Anomaly Scores...")
    iso_forest = IsolationForest(n_estimators=100, random_state=42)
    iso_forest.fit(X_iso)
    # lower score = more anomalous
    df_pure['anomaly_score'] = iso_forest.decision_function(X_iso)
    
    # 3. Pseudo-Labeling via Top-K Reverse Engineering
    df_true = pd.read_csv('true_daily_defects.csv')
    df_true['date'] = pd.to_datetime(df_true['date']).dt.date
    true_defects_dict = dict(zip(df_true['date'], df_true['true_defects']))
    
    df_pure['is_defect'] = 0
    df_pure['date'] = df_pure['working time'].dt.date
    
    total_assigned = 0
    
    for date, group in df_pure.groupby('date'):
        if date in true_defects_dict:
            k = true_defects_dict[date]
            if k > 0:
                # Get indices of the K smallest anomaly scores (most anomalous)
                top_k_indices = group.nsmallest(k, 'anomaly_score').index
                df_pure.loc[top_k_indices, 'is_defect'] = 1
                total_assigned += k
                
    print(f"\nPseudo-labeling Complete! Assigned {total_assigned} exact defects based on factory labels.")
    
    # 4. Supervised Learning
    X_clf = df_pure[features] # Using raw features so trees are easy to interpret!
    y_clf = df_pure['is_defect']
    
    print("\n--- Model Evaluation (5-Fold Cross Validation) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    if HAS_XGB:
        clf = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, scale_pos_weight=(len(y_clf)-sum(y_clf))/sum(y_clf), random_state=42, eval_metric='logloss')
        model_name = "XGBoost"
    else:
        clf = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42)
        model_name = "Random Forest"
        
    f1_scores = []
    precisions = []
    recalls = []
    
    for train_idx, test_idx in skf.split(X_clf, y_clf):
        X_train, X_test = X_clf.iloc[train_idx], X_clf.iloc[test_idx]
        y_train, y_test = y_clf.iloc[train_idx], y_clf.iloc[test_idx]
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        f1_scores.append(f1_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        
    print(f"Model: {model_name}")
    print(f"Average Precision: {np.mean(precisions):.4f}")
    print(f"Average Recall: {np.mean(recalls):.4f}")
    print(f"Average F1-Score: {np.mean(f1_scores):.4f}")
    
    # 5. Final Model Training on all data for Feature Importance
    clf.fit(X_clf, y_clf)
    
    # Feature Importance Plot
    if HAS_XGB:
        importances = clf.feature_importances_
    else:
        importances = clf.feature_importances_
        
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n--- Final Model Feature Importance ---")
    print(importance_df)
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title(f'{model_name} Feature Importance (Phase 6)')
    plt.tight_layout()
    plt.savefig('plots/phase6_feature_importance.png')
    plt.close()
    
    print("\nPhase 6 Optimization Completed. Results saved to 'plots/phase6_feature_importance.png'")

if __name__ == "__main__":
    run_phase6_optimization()
