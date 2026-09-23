import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier, StackingClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings('ignore')

def get_base_features(df):
    base_features = ['weld force(bar)', 'weld current(kA)', 'weld Voltage(v)', 'weld time(ms)']
    return base_features

def engineer_physics_features(df):
    df_feat = df.copy()
    base_features = get_base_features(df_feat)
    
    # Physics
    df_feat['physics_resistance'] = df_feat['weld Voltage(v)'] / (df_feat['weld current(kA)'] + 1e-6)
    df_feat['physics_energy'] = df_feat['weld Voltage(v)'] * df_feat['weld current(kA)'] * df_feat['weld time(ms)']
    
    all_base_and_physics = base_features + ['physics_resistance', 'physics_energy']
    
    # Time-Series Fatigue
    for col in all_base_and_physics:
        df_feat[f"{col}_roll_mean_5"] = df_feat[col].rolling(window=5, min_periods=1).mean()
        df_feat[f"{col}_roll_std_5"] = df_feat[col].rolling(window=5, min_periods=1).std().fillna(0)
        df_feat[f"{col}_roll_mean_30"] = df_feat[col].rolling(window=30, min_periods=1).mean()
        df_feat[f"{col}_ema_01"] = df_feat[col].ewm(alpha=0.1, adjust=False).mean()
        df_feat[f"{col}_diff_prev"] = df_feat[col].diff().fillna(0)
        
    feature_cols = [c for c in df_feat.columns if (any(bf in c for bf in base_features) or 'physics' in c) and not c.endswith('_scaled') and c not in ['is_defect', 'date', 'working time', 'anomaly_score']]
    return df_feat, feature_cols

def evaluate_labeling_strategy(strategy_name, df_labeled, feature_cols):
    X = df_labeled[feature_cols]
    y = df_labeled['is_defect']
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    xgb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss')
    lgbm = LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, verbose=-1)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    meta_learner = LogisticRegression(class_weight='balanced', random_state=42)
    
    stacking_clf = StackingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm), ('rf', rf)],
        final_estimator=meta_learner,
        cv=2
    )
    
    f1_scores = []
    smote = SMOTE(k_neighbors=3, random_state=42)
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Internal hold-out for unbiased threshold
        X_tr_sub, X_val_sub, y_tr_sub, y_val_sub = train_test_split(X_train, y_train, test_size=0.25, stratify=y_train, random_state=42)
        X_tr_sub_sm, y_tr_sub_sm = smote.fit_resample(X_tr_sub, y_tr_sub)
        
        stack_temp = StackingClassifier(estimators=[('xgb', xgb), ('lgbm', lgbm), ('rf', rf)], final_estimator=meta_learner, cv=2)
        stack_temp.fit(X_tr_sub_sm, y_tr_sub_sm)
        
        y_val_sub_proba = stack_temp.predict_proba(X_val_sub)[:, 1]
        prec_tr, rec_tr, thresh_tr = precision_recall_curve(y_val_sub, y_val_sub_proba)
        
        f1_array_tr = 2 * (prec_tr * rec_tr) / (prec_tr + rec_tr + 1e-8)
        best_idx_tr = np.argmax(f1_array_tr)
        best_thresh = thresh_tr[best_idx_tr] if best_idx_tr < len(thresh_tr) else 0.5
        
        # Final train
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        stacking_clf.fit(X_train_sm, y_train_sm)
        
        y_proba = stacking_clf.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= best_thresh).astype(int)
        
        f1_scores.append(f1_score(y_test, y_pred))
        
    avg_f1 = np.mean(f1_scores)
    print(f"[{strategy_name}] Unbiased F1-Score: {avg_f1:.4f}")
    return avg_f1

def run_goal_advanced_labeling():
    print("Loading Pure Production Data...")
    df_prod = pd.read_csv('pure_production_data.csv')
    df_prod['working time'] = pd.to_datetime(df_prod['working time'])
    maintenance_days = ['2020-03-25', '2020-03-27', '2020-03-31', '2020-04-03']
    maintenance_dates = pd.to_datetime(maintenance_days).date
    df_pure = df_prod[~df_prod['working time'].dt.date.isin(maintenance_dates)].copy().reset_index(drop=True)
    
    df_true = pd.read_csv('true_daily_defects.csv')
    df_true['date'] = pd.to_datetime(df_true['date']).dt.date
    true_defects_dict = dict(zip(df_true['date'], df_true['true_defects']))
    df_pure['date'] = df_pure['working time'].dt.date
    
    # Pre-calculate physics features
    df_feat, feature_cols = engineer_physics_features(df_pure)
    
    strategies = {}
    
    # Strategy 1: Isolation Forest on Base Features (The old way)
    print("\nEvaluating Strategy 1: Base IF Labeling")
    base_features = get_base_features(df_feat)
    scaled_cols = [f"{col}_scaled" for col in base_features]
    iso_base = IsolationForest(n_estimators=100, random_state=42)
    iso_base.fit(df_feat[scaled_cols])
    df_feat['anomaly_score_1'] = iso_base.decision_function(df_feat[scaled_cols])
    
    df_strat1 = df_feat.copy()
    df_strat1['is_defect'] = 0
    for date, group in df_strat1.groupby('date'):
        if date in true_defects_dict and true_defects_dict[date] > 0:
            top_k_idx = group.nsmallest(true_defects_dict[date], 'anomaly_score_1').index
            df_strat1.loc[top_k_idx, 'is_defect'] = 1
    strategies['Base_IF'] = evaluate_labeling_strategy('Base_IF', df_strat1, feature_cols)
    
    # Strategy 2: Isolation Forest on Physics Features (The new way)
    print("\nEvaluating Strategy 2: Physics-Aware IF Labeling")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaled_physics = scaler.fit_transform(df_feat[feature_cols])
    
    iso_phys = IsolationForest(n_estimators=100, random_state=42)
    iso_phys.fit(scaled_physics)
    df_feat['anomaly_score_2'] = iso_phys.decision_function(scaled_physics)
    
    df_strat2 = df_feat.copy()
    df_strat2['is_defect'] = 0
    for date, group in df_strat2.groupby('date'):
        if date in true_defects_dict and true_defects_dict[date] > 0:
            top_k_idx = group.nsmallest(true_defects_dict[date], 'anomaly_score_2').index
            df_strat2.loc[top_k_idx, 'is_defect'] = 1
    strategies['Physics_IF'] = evaluate_labeling_strategy('Physics_IF', df_strat2, feature_cols)
    
    # Strategy 3: Local Outlier Factor on Physics Features
    print("\nEvaluating Strategy 3: Physics-Aware LOF Labeling")
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
    lof.fit(scaled_physics)
    df_feat['anomaly_score_3'] = lof.decision_function(scaled_physics)
    
    df_strat3 = df_feat.copy()
    df_strat3['is_defect'] = 0
    for date, group in df_strat3.groupby('date'):
        if date in true_defects_dict and true_defects_dict[date] > 0:
            top_k_idx = group.nsmallest(true_defects_dict[date], 'anomaly_score_3').index
            df_strat3.loc[top_k_idx, 'is_defect'] = 1
    strategies['Physics_LOF'] = evaluate_labeling_strategy('Physics_LOF', df_strat3, feature_cols)

    print("\n--- FINAL RESULTS ---")
    for k, v in strategies.items():
        print(f"{k}: {v:.4f}")
        
    best_strategy = max(strategies, key=strategies.get)
    best_score = strategies[best_strategy]
    print(f"\n[WINNER] The Absolute Best Labeling Method is: {best_strategy} (F1-Score: {best_score:.4f})")
    
    # Plotting
    plt.figure(figsize=(10, 6))
    names = list(strategies.keys())
    scores = list(strategies.values())
    colors = ['red' if x == best_strategy else 'steelblue' for x in names]
    
    sns.barplot(x=names, y=scores, palette=colors)
    plt.title("Search for the Optimal Labeling Strategy (Physics-Aware)", fontsize=14, fontweight='bold')
    plt.ylabel("Unbiased F1-Score (Stacking Ensemble)", fontsize=12)
    plt.ylim(0, max(scores) + 0.1)
    
    for i, v in enumerate(scores):
        plt.text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('plots/phase13_labeling_evolution.png', dpi=300)
    plt.close()
    print("Chart saved to 'plots/phase13_labeling_evolution.png'")

if __name__ == "__main__":
    run_goal_advanced_labeling()
