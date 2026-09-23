import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
import shap
import matplotlib.pyplot as plt
import os

plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('plots', exist_ok=True)

def perform_xai_analysis():
    print("Loading pseudo-labeled data...")
    df = pd.read_csv('pseudo_labeled_data.csv')
    
    # We want to extract rules on original unscaled features for readability
    features = ['Thickness 1(mm)', 'Thickness 2(mm)', 'weld force(bar)', 
                'weld current(kA)', 'weld Voltage(v)', 'weld time(ms)']
    
    X = df[features]
    y = df['is_anomaly']  # 1 for anomaly, 0 for normal
    
    print(f"Total Anomalies found: {y.sum()} out of {len(y)}")
    
    # 1. Surrogate Decision Tree for Rule Extraction
    dt = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced')
    dt.fit(X, y)
    
    rules = export_text(dt, feature_names=features)
    print("\n--- Extracted Rules for Defects (Anomalies) ---")
    print(rules)
    
    with open('defect_rules.txt', 'w', encoding='utf-8') as f:
        f.write(rules)
        
    # 2. SHAP Analysis on a simpler model or directly on the DT
    # TreeExplainer is fast for Decision Trees
    explainer = shap.TreeExplainer(dt)
    shap_values = explainer.shap_values(X)
    
    # Depending on SHAP version, shap_values might be a list (for binary classification)
    if isinstance(shap_values, list):
        shap_values_to_plot = shap_values[1]  # We care about class 1 (anomaly)
    else:
        shap_values_to_plot = shap_values
        
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_to_plot, X, show=False)
    plt.title('불량 예측에 미치는 주요 요인 (SHAP Summary)')
    plt.tight_layout()
    plt.savefig('plots/shap_summary.png', dpi=300)
    plt.close()
    print("Saved SHAP summary to plots/shap_summary.png")
    
    # 3. Analyze "Model Failure Conditions"
    # The models had negative correlation, meaning they predicted anomalies on days with few true defects,
    # and missed defects on days with many. Let's see the average feature values of anomalies vs normal.
    mean_normal = df[df['is_anomaly'] == 0][features].mean()
    mean_anomaly = df[df['is_anomaly'] == 1][features].mean()
    
    comparison = pd.DataFrame({'Normal': mean_normal, 'Anomaly (High Risk)': mean_anomaly})
    comparison['Difference (%)'] = ((comparison['Anomaly (High Risk)'] - comparison['Normal']) / comparison['Normal']) * 100
    print("\n--- Average Condition Comparison ---")
    print(comparison)
    comparison.to_csv('condition_comparison.csv')

if __name__ == "__main__":
    perform_xai_analysis()
