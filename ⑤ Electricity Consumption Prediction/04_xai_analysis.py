# %%
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler
import os

# Create plots dir if not exists
os.makedirs('plots', exist_ok=True)

df = pd.read_csv('preprocessed_data.csv')
df = df.dropna().reset_index(drop=True)

features = ['생산량', '기온', '풍속', '습도', '강수량', '공장인원', 'Peak_lag1', 'Peak_lag24', 'Peak_MA3', 'Peak_MA6', 'Peak_MA24', 'Financial_Risk_Index']
X = df[features]
y = df['Peak']

# Load the LightGBM residual model for SHAP analysis
# To make it simple and independent, we can train a quick LGBM on the target directly just for SHAP interpretation of the overall dataset
print("Training a direct LightGBM model for overall feature importance (SHAP)...")
model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 1. SHAP Analysis
print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Plot SHAP summary
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, show=False)
plt.title('SHAP Feature Importance (Impact on Peak Power)')
plt.tight_layout()
plt.savefig('plots/shap_summary.png')
plt.close()
print("SHAP summary plot saved to 'plots/shap_summary.png'")

# 2. Explicit Risk Condition Rule Extraction
# 상위 5% 피크에 해당하는 데이터를 위험 상태(1)로 정의
p95 = y.quantile(0.95)
print(f"\n95th Percentile of Peak Power (Top 5% Threshold): {p95}")

df['Risk_Flag'] = (df['Peak'] > p95).astype(int)
y_risk = df['Risk_Flag']

# Train Decision Tree to extract rules
dt = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced')
dt.fit(X, y_risk)

# Extract and print rules
print("\nExplicit Risk Condition Rules (Decision Tree):")
tree_rules = export_text(dt, feature_names=features)
print(tree_rules)

# Save rules to text file
with open('risk_rules.txt', 'w', encoding='utf-8') as f:
    f.write(f"Top 5% Peak Threshold: {p95}\n")
    f.write(tree_rules)

print("\nXAI and Rule Extraction completed.")
