import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def add_physics_features(df):
    df = df.copy()
    df['Cooling_Time'] = df['Cycle_Time'] - (df['Injection_Time'] + df['Plasticizing_Time'] + df['Clamp_Close_Time'])
    df['Temp_Gradient'] = df['Barrel_Temperature_6'] - df['Barrel_Temperature_1']
    df['Avg_Barrel_Temp'] = df[[f'Barrel_Temperature_{i}' for i in range(1, 7)]].mean(axis=1)
    df['Pressure_Drop'] = df['Max_Injection_Pressure'] - df['Max_Switch_Over_Pressure']
    df['Injection_Efficiency'] = df['Max_Injection_Speed'] / (df['Injection_Time'] + 1e-5)
    return df

def run_business_cost_optimization():
    print("Loading Data for Business Cost Analysis...")
    df_lab = pd.read_csv('c:/Hackerton/① Molding/data/moldset_labeled_cn7.csv')
    df_unlab = pd.read_csv('c:/Hackerton/① Molding/data/moldset_unlabeled_cn7.csv')

    for df in [df_lab, df_unlab]:
        for c in ['Unnamed: 0', 'Clamp_Open_Position']:
            if c in df.columns: df.drop(columns=[c], inplace=True)

    df_lab = add_physics_features(df_lab)
    df_unlab = add_physics_features(df_unlab)

    target = 'PassOrFail'
    features = [c for c in df_lab.columns if c != target]

    X_lab = df_lab[features].values
    y_lab = df_lab[target].values
    X_unlab = df_unlab[features].values
    
    scaler = StandardScaler()
    X_lab_sc = scaler.fit_transform(X_lab)
    X_unlab_sc = scaler.transform(X_unlab)

    print("Generating Pseudo-Labels (Phase 6 strategy)...")
    pseudo_model = LogisticRegression(class_weight='balanced', random_state=42)
    pseudo_model.fit(X_lab_sc, y_lab)
    p_unlab = pseudo_model.predict_proba(X_unlab_sc)[:, 1]

    hi_def = np.where(p_unlab > 0.95)[0]
    hi_nor = np.where(p_unlab < 0.05)[0]
    np.random.seed(42)
    if len(hi_nor) > 5000: hi_nor = np.random.choice(hi_nor, 5000, replace=False)
    
    Xp = np.vstack([X_unlab_sc[hi_def], X_unlab_sc[hi_nor]])
    yp = np.concatenate([np.ones(len(hi_def)), np.zeros(len(hi_nor))])

    print("Running Ensemble CV to get probabilities...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    voting = VotingClassifier(estimators=[
        ('lr', LogisticRegression(class_weight='balanced', random_state=42)),
        ('svc', SVC(probability=True, class_weight='balanced', random_state=42)),
        ('nb', GaussianNB()),
    ], voting='soft', weights=[2, 2, 1])

    all_y_true = []
    all_y_probs = []

    for tr, te in skf.split(X_lab_sc, y_lab):
        Xtr = np.vstack([X_lab_sc[tr], Xp])
        ytr = np.concatenate([y_lab[tr], yp])
        
        voting.fit(Xtr, ytr)
        probs = voting.predict_proba(X_lab_sc[te])[:, 1]
        
        all_y_true.extend(y_lab[te])
        all_y_probs.extend(probs)
        
    all_y_true = np.array(all_y_true)
    all_y_probs = np.array(all_y_probs)

    print("Calculating Business Cost across thresholds...")
    # Business Cost Assumption
    C_FP = 1  # 1만원
    C_FN = 100 # 100만원
    
    thresholds = np.linspace(0.0001, 0.99, 10000)
    costs = []
    recalls = []
    
    for t in thresholds:
        preds = (all_y_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(all_y_true, preds, labels=[0,1]).ravel()
        
        cost = (fp * C_FP) + (fn * C_FN)
        costs.append(cost)
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        recalls.append(recall)

    costs = np.array(costs)
    min_cost_idx = np.argmin(costs)
    best_threshold = thresholds[min_cost_idx]
    min_cost = costs[min_cost_idx]
    best_recall = recalls[min_cost_idx]
    
    # Cost at default threshold 0.5
    default_preds = (all_y_probs >= 0.5).astype(int)
    _, def_fp, def_fn, _ = confusion_matrix(all_y_true, default_preds, labels=[0,1]).ravel()
    def_cost = (def_fp * C_FP) + (def_fn * C_FN)
    
    print(f"\n--- Business Cost Analysis ---")
    print(f"Default Threshold (0.5) Cost: {def_cost}만원")
    print(f"Optimal Threshold: {best_threshold:.4f}")
    print(f"Minimum Cost: {min_cost}만원")
    print(f"Recall at Optimal Threshold: {best_recall:.4f}")
    print(f"Cost Savings: {def_cost - min_cost}만원 ({(def_cost - min_cost)/def_cost*100:.1f}%)")

    print("\nGenerating Visualization...")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:red'
    ax1.set_xlabel('Decision Threshold (AI Confidence)', fontsize=12)
    ax1.set_ylabel('Total Expected Cost (만 원)', color=color, fontsize=12)
    ax1.plot(thresholds, costs, color=color, linewidth=2, label='Total Cost (FP*1 + FN*100)')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Mark Min Cost
    ax1.axvline(x=best_threshold, color='green', linestyle='--', linewidth=2, label=f'Optimal Threshold ({best_threshold:.2f})')
    ax1.scatter(best_threshold, min_cost, color='green', s=100, zorder=5)
    ax1.text(best_threshold + 0.02, min_cost + 100, f'Min Cost\n{min_cost}만원', color='green', fontweight='bold')
    
    # Mark Default Cost
    ax1.scatter(0.5, def_cost, color='gray', s=100, zorder=5)
    ax1.text(0.5 - 0.05, def_cost + 100, f'Default (0.5)\n{def_cost}만원', color='gray', fontweight='bold', ha='right')

    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Recall (재현율)', color=color, fontsize=12)  
    ax2.plot(thresholds, recalls, color=color, linewidth=2, linestyle=':', label='Recall')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9), bbox_transform=ax1.transAxes)
    plt.title('Business Cost Optimization: Shifting the Decision Boundary', fontsize=16, fontweight='bold')
    plt.savefig('plots/11_business_cost.png', dpi=300)
    plt.close()
    
    print("Saved plot to 'plots/11_business_cost.png'")

if __name__ == "__main__":
    run_business_cost_optimization()
