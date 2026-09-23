import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

def run_inspection_priority():
    print("Loading data for Inspection Priority System...")
    df = pd.read_csv('data/moldset_labeled_cn7.csv')
    
    cols_to_drop = ['Unnamed: 0', 'Clamp_Open_Position']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    target = 'PassOrFail'
    features = [c for c in df.columns if c != target]
    
    X = df[features]
    y = df[target]
    
    # We use Logistic Regression as it showed the highest recall in Phase 2
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
    
    # Get Out-Of-Fold probabilities for the entire dataset
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_proba = cross_val_predict(model, X, y, cv=skf, method='predict_proba')[:, 1]
    
    # Create Priority DataFrame
    df_priority = pd.DataFrame({
        'True_Label': y,
        'Defect_Probability': y_proba
    })
    
    # Sort by probability descending (High Priority first)
    df_priority = df_priority.sort_values(by='Defect_Probability', ascending=False).reset_index(drop=True)
    
    # Calculate Cumulative Recall
    total_defects = df_priority['True_Label'].sum()
    df_priority['Cumulative_Defects'] = df_priority['True_Label'].cumsum()
    df_priority['Recall'] = df_priority['Cumulative_Defects'] / total_defects
    df_priority['Percentage_Inspected'] = (df_priority.index + 1) / len(df_priority) * 100
    
    # Find key business metrics (e.g., how much to inspect to catch 80% of defects)
    thresholds = [0.5, 0.8, 0.9, 1.0]
    inspection_costs = []
    
    for t in thresholds:
        # Find the first row where Recall >= t
        idx = df_priority[df_priority['Recall'] >= t].index
        if len(idx) > 0:
            cost = df_priority.loc[idx[0], 'Percentage_Inspected']
            inspection_costs.append((t*100, cost))
    
    print("\n--- Business Impact: Smart Inspection System ---")
    for target_recall, cost in inspection_costs:
        print(f"To catch {target_recall:g}% of all defects, you only need to inspect the top {cost:.1f}% of production.")
        
    # Plotting Cumulative Gain Curve (Inspection Priority Curve)
    plt.figure(figsize=(10, 6))
    plt.plot(df_priority['Percentage_Inspected'], df_priority['Recall'] * 100, 
             color='red', linewidth=3, label='AI Priority Inspection')
    
    # Baseline (Random Inspection)
    plt.plot([0, 100], [0, 100], linestyle='--', color='gray', label='Random Inspection (No AI)')
    
    # Add annotations for key business points
    for target_recall, cost in inspection_costs:
        if target_recall in [80, 100]:
            plt.scatter(cost, target_recall, color='black', s=50, zorder=5)
            plt.annotate(f'Catch {target_recall:g}%\nInspect {cost:.1f}%', 
                         (cost, target_recall), 
                         xytext=(cost + 5, target_recall - 10),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                         fontsize=10, fontweight='bold')
                         
    plt.title("Phase 4: AI-Driven Inspection Priority System", fontsize=15, fontweight='bold')
    plt.xlabel("% of Total Production Inspected", fontsize=12)
    plt.ylabel("% of Total Defects Caught (Recall)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=12)
    plt.xlim(0, 100)
    plt.ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig('plots/04_inspection_priority_curve.png', dpi=300)
    plt.close()
    
    print("\nPhase 4 Complete. Chart saved to 'plots/04_inspection_priority_curve.png'")

if __name__ == "__main__":
    run_inspection_priority()
