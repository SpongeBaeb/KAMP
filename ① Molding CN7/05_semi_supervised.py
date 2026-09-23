import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def run_semi_supervised():
    print("Loading Data for Semi-Supervised Learning...")
    df_lab = pd.read_csv('data/moldset_labeled_cn7.csv')
    df_unlab = pd.read_csv('data/moldset_unlabeled_cn7.csv')
    
    cols_to_drop = ['Unnamed: 0', 'Clamp_Open_Position']
    df_lab = df_lab.drop(columns=[col for col in cols_to_drop if col in df_lab.columns])
    df_unlab = df_unlab.drop(columns=[col for col in cols_to_drop if col in df_unlab.columns])
    
    target = 'PassOrFail'
    features = [c for c in df_lab.columns if c != target]
    
    X_lab = df_lab[features]
    y_lab = df_lab[target]
    X_unlab = df_unlab[features]
    
    print(f"Labeled Data: {len(X_lab)}, Unlabeled Data: {len(X_unlab)}")
    
    # Baseline Model Evaluation (for comparison)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    
    base_pr_auc, base_recall = [], []
    for train_idx, test_idx in skf.split(X_lab, y_lab):
        X_train, X_test = X_lab.iloc[train_idx], X_lab.iloc[test_idx]
        y_train, y_test = y_lab.iloc[train_idx], y_lab.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        base_pr_auc.append(average_precision_score(y_test, y_proba))
        base_recall.append(recall_score(y_test, y_pred, zero_division=0))
        
    print(f"\n[Baseline] PR-AUC: {np.mean(base_pr_auc):.4f} | Recall: {np.mean(base_recall):.4f}")
    
    # 1. Train on ALL labeled data to generate pseudo-labels
    model.fit(X_lab, y_lab)
    
    # 2. Predict on Unlabeled
    print("\nGenerating Pseudo-Labels for Unlabeled Data...")
    unlab_proba = model.predict_proba(X_unlab)[:, 1]
    
    # 3. Select highly confident samples
    # We want to be strict to avoid confirmation bias
    high_conf_defect_idx = np.where(unlab_proba > 0.95)[0]
    high_conf_normal_idx = np.where(unlab_proba < 0.01)[0] # Very strict for normal since it's the majority
    
    # Subsample normals to maintain some balance (don't overwhelm with millions of normals)
    np.random.seed(42)
    if len(high_conf_normal_idx) > 5000:
        high_conf_normal_idx = np.random.choice(high_conf_normal_idx, 5000, replace=False)
        
    print(f"High Confidence Defects found: {len(high_conf_defect_idx)}")
    print(f"High Confidence Normals added: {len(high_conf_normal_idx)}")
    
    X_pseudo_defect = X_unlab.iloc[high_conf_defect_idx]
    y_pseudo_defect = pd.Series([1] * len(high_conf_defect_idx))
    
    X_pseudo_normal = X_unlab.iloc[high_conf_normal_idx]
    y_pseudo_normal = pd.Series([0] * len(high_conf_normal_idx))
    
    X_pseudo = pd.concat([X_pseudo_defect, X_pseudo_normal]).reset_index(drop=True)
    y_pseudo = pd.concat([y_pseudo_defect, y_pseudo_normal]).reset_index(drop=True)
    
    # 4. Retrain and Evaluate (Semi-Supervised)
    # We must only evaluate on the ORIGINAL labeled data (using the same folds)
    semi_pr_auc, semi_recall = [], []
    
    print("\nEvaluating Semi-Supervised Model...")
    for train_idx, test_idx in skf.split(X_lab, y_lab):
        X_train, X_test = X_lab.iloc[train_idx], X_lab.iloc[test_idx]
        y_train, y_test = y_lab.iloc[train_idx], y_lab.iloc[test_idx]
        
        # Combine original train fold with ALL pseudo-labeled data
        X_train_expanded = pd.concat([X_train, X_pseudo]).reset_index(drop=True)
        y_train_expanded = pd.concat([y_train, y_pseudo]).reset_index(drop=True)
        
        model_semi = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
        model_semi.fit(X_train_expanded, y_train_expanded)
        
        y_proba = model_semi.predict_proba(X_test)[:, 1]
        y_pred = model_semi.predict(X_test)
        
        semi_pr_auc.append(average_precision_score(y_test, y_proba))
        semi_recall.append(recall_score(y_test, y_pred, zero_division=0))
        
    print(f"[Semi-Supervised] PR-AUC: {np.mean(semi_pr_auc):.4f} | Recall: {np.mean(semi_recall):.4f}")
    
    # Plotting Improvement
    labels = ['Baseline', 'Semi-Supervised (Self-Training)']
    pr_aucs = [np.mean(base_pr_auc), np.mean(semi_pr_auc)]
    recalls = [np.mean(base_recall), np.mean(semi_recall)]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, pr_aucs, width, label='PR-AUC', color='steelblue')
    rects2 = ax.bar(x + width/2, recalls, width, label='Recall', color='darkorange')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Phase 5: Performance Boost with Semi-Supervised Learning', fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.legend(loc='lower center', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
                        
    autolabel(rects1)
    autolabel(rects2)
    
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig('plots/05_semi_supervised.png', dpi=300)
    plt.close()
    
    print("\nPhase 5 Complete. Chart saved to 'plots/05_semi_supervised.png'")

if __name__ == "__main__":
    run_semi_supervised()
