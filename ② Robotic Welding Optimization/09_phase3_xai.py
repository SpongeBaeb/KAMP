import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
import shap
import matplotlib.pyplot as plt
import os

plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

def phase3_xai():
    df = pd.read_csv('phase3_pseudo_labeled.csv')
    features = ['Thickness 1(mm)', 'Thickness 2(mm)', 'weld force(bar)', 
                'weld current(kA)', 'weld Voltage(v)', 'weld time(ms)']
    
    X = df[features]
    y = df['is_anomaly']
    
    print(f"Total Anomalies found in Phase 3: {y.sum()} out of {len(y)}")
    
    dt = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight='balanced')
    dt.fit(X, y)
    
    rules = export_text(dt, feature_names=features)
    print("\n--- Phase 3 Rules ---")
    print(rules)
    
    # Analyze dates
    print("\n--- Anomaly Dates ---")
    print(df[y==1]['working time'].value_counts())
    
    explainer = shap.TreeExplainer(dt)
    shap_values = explainer.shap_values(X)
    shap_values_to_plot = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_to_plot, X, show=False)
    plt.title('Phase 3 진짜 불량 원인 분석 (캘리브레이션 100% 제거 후)')
    plt.tight_layout()
    plt.savefig('plots/phase3_shap.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    phase3_xai()
