import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
import shap
import warnings
warnings.filterwarnings('ignore')

def run_physics_optimization():
    print("Loading Data...")
    df_lab = pd.read_csv('c:/Hackerton/① Molding/data/moldset_labeled_cn7.csv')
    df_unlab = pd.read_csv('c:/Hackerton/① Molding/data/moldset_unlabeled_cn7.csv')
    
    cols_to_drop = ['Unnamed: 0', 'Clamp_Open_Position']
    df_lab = df_lab.drop(columns=[col for col in cols_to_drop if col in df_lab.columns])
    df_unlab = df_unlab.drop(columns=[col for col in cols_to_drop if col in df_unlab.columns])
    
    def add_physics_features(df):
        df = df.copy()
        # 1. Cooling Time Estimate
        df['Cooling_Time'] = df['Cycle_Time'] - (df['Injection_Time'] + df['Plasticizing_Time'] + df['Clamp_Close_Time'])
        # 2. Temp Gradient (Barrel Rear to Front)
        df['Temp_Gradient'] = df['Barrel_Temperature_6'] - df['Barrel_Temperature_1']
        # 3. Avg Barrel Temp
        temp_cols = [f'Barrel_Temperature_{i}' for i in range(1, 7)]
        df['Avg_Barrel_Temp'] = df[temp_cols].mean(axis=1)
        # 4. Pressure Drop
        df['Pressure_Drop'] = df['Max_Injection_Pressure'] - df['Max_Switch_Over_Pressure']
        # 5. Injection Efficiency
        df['Injection_Efficiency'] = df['Max_Injection_Speed'] / (df['Injection_Time'] + 1e-5)
        return df

    df_lab = add_physics_features(df_lab)
    df_unlab = add_physics_features(df_unlab)
    
    target = 'PassOrFail'
    features = [c for c in df_lab.columns if c != target]
    
    X_lab = df_lab[features]
    y_lab = df_lab[target]
    X_unlab = df_unlab[features]
    
    print(f"Total Features including Physics Domain: {len(features)}")
    
    # Pre-scale
    scaler = StandardScaler()
    X_lab_scaled = pd.DataFrame(scaler.fit_transform(X_lab), columns=features)
    X_unlab_scaled = pd.DataFrame(scaler.transform(X_unlab), columns=features)
    
    # 1. Pseudo-Labeling Phase
    print("Generating Pseudo-Labels...")
    pseudo_model = LogisticRegression(class_weight='balanced', random_state=42)
    pseudo_model.fit(X_lab_scaled, y_lab)
    unlab_proba = pseudo_model.predict_proba(X_unlab_scaled)[:, 1]
    
    high_conf_defect_idx = np.where(unlab_proba > 0.95)[0]
    high_conf_normal_idx = np.where(unlab_proba < 0.05)[0]
    
    np.random.seed(42)
    if len(high_conf_normal_idx) > 5000:
        high_conf_normal_idx = np.random.choice(high_conf_normal_idx, 5000, replace=False)
        
    X_pseudo_defect = X_unlab_scaled.iloc[high_conf_defect_idx]
    y_pseudo_defect = pd.Series([1] * len(high_conf_defect_idx))
    
    X_pseudo_normal = X_unlab_scaled.iloc[high_conf_normal_idx]
    y_pseudo_normal = pd.Series([0] * len(high_conf_normal_idx))
    
    X_pseudo = pd.concat([X_pseudo_defect, X_pseudo_normal]).reset_index(drop=True)
    y_pseudo = pd.concat([y_pseudo_defect, y_pseudo_normal]).reset_index(drop=True)
    
    # 2. Ensemble CV Phase
    print("Running Semi-Supervised Ensemble with Physics Features...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    lr = LogisticRegression(class_weight='balanced', random_state=42)
    svc = SVC(probability=True, class_weight='balanced', random_state=42)
    nb = GaussianNB()
    
    voting = VotingClassifier(estimators=[('lr', lr), ('svc', svc), ('nb', nb)], voting='soft')
    
    praucs = []
    
    for train_idx, test_idx in skf.split(X_lab_scaled, y_lab):
        X_train, X_test = X_lab_scaled.iloc[train_idx], X_lab_scaled.iloc[test_idx]
        y_train, y_test = y_lab.iloc[train_idx], y_lab.iloc[test_idx]
        
        X_train_expanded = pd.concat([X_train, X_pseudo]).reset_index(drop=True)
        y_train_expanded = pd.concat([y_train, y_pseudo]).reset_index(drop=True)
        
        voting.fit(X_train_expanded, y_train_expanded)
        probs = voting.predict_proba(X_test)[:, 1]
        praucs.append(average_precision_score(y_test, probs))
        
    final_prauc = np.mean(praucs)
    print(f"\n[PHASE 8 COMPLETE]")
    print(f"Physics-Informed PR-AUC: {final_prauc:.4f}")
    
    # 3. SHAP Analysis (Explaining the Physics Features)
    print("\nGenerating SHAP Visualizations for Physics Features...")
    explainer = shap.LinearExplainer(pseudo_model, X_lab_scaled)
    shap_values = explainer.shap_values(X_lab_scaled)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_lab_scaled, show=False)
    plt.title("SHAP Feature Importance (Including Physics Features)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/08_physics_shap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved SHAP plot to 'plots/08_physics_shap.png'")

if __name__ == "__main__":
    run_physics_optimization()
