import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

def run_baseline_model():
    print("Loading labeled CN7 data...")
    df = pd.read_csv('data/moldset_labeled_cn7.csv')
    
    cols_to_drop = ['Unnamed: 0', 'Clamp_Open_Position']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    target = 'PassOrFail'
    features = [c for c in df.columns if c != target]
    
    X = df[features]
    y = df[target]
    
    # Stratified 5-Fold to ensure the 17 defects are distributed across folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Calculate pos_weight for XGBoost
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos = neg_count / pos_count
    
    models = {
        'Logistic Regression (Balanced)': LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000),
        'Random Forest (Balanced)': RandomForestClassifier(class_weight='balanced', random_state=42),
        'LightGBM (Balanced)': LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1),
        'XGBoost (Scale Pos Weight)': XGBClassifier(scale_pos_weight=scale_pos, random_state=42, eval_metric='logloss'),
        'XGBoost (SMOTE)': XGBClassifier(random_state=42, eval_metric='logloss')
    }
    
    results = []
    
    print("\nStarting Cross-Validation Evaluation...")
    for model_name, model in models.items():
        metrics = {'F1': [], 'Precision': [], 'Recall': [], 'PR-AUC': []}
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            if 'SMOTE' in model_name:
                smote = SMOTE(random_state=42, k_neighbors=3) # small k due to few positive samples
                X_train, y_train = smote.fit_resample(X_train, y_train)
                
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
            
            metrics['F1'].append(f1_score(y_test, y_pred, zero_division=0))
            metrics['Precision'].append(precision_score(y_test, y_pred, zero_division=0))
            metrics['Recall'].append(recall_score(y_test, y_pred, zero_division=0))
            metrics['PR-AUC'].append(average_precision_score(y_test, y_proba))
            
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        results.append({
            'Model': model_name,
            'F1': avg_metrics['F1'],
            'Precision': avg_metrics['Precision'],
            'Recall': avg_metrics['Recall'],
            'PR-AUC': avg_metrics['PR-AUC']
        })
        print(f"[{model_name}] F1: {avg_metrics['F1']:.4f} | Recall: {avg_metrics['Recall']:.4f} | PR-AUC: {avg_metrics['PR-AUC']:.4f}")
        
    df_results = pd.DataFrame(results)
    
    # Plotting
    df_melted = df_results.melt(id_vars='Model', var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted, palette='viridis')
    plt.title("Phase 2: Baseline Model Performance Comparison (5-Fold CV)", fontsize=16, fontweight='bold')
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for p in plt.gca().patches:
        plt.gca().annotate(f"{p.get_height():.2f}", 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=9)
                           
    plt.tight_layout()
    plt.savefig('plots/02_model_comparison.png', dpi=300)
    plt.close()
    
    print("\nPhase 2 Complete. Chart saved to 'plots/02_model_comparison.png'")

if __name__ == "__main__":
    run_baseline_model()
