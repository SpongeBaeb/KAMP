import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os

plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False
os.makedirs('plots', exist_ok=True)

def evaluate_and_train():
    df = pd.read_csv('phase2_preprocessed_raw.csv')
    true_defects = pd.read_csv('true_daily_defects.csv')
    true_defects['date'] = true_defects['date'].astype(str)
    
    features = [c for c in df.columns if c.endswith('_scaled')]
    
    # 39 total true defects, so contamination around 39/11859 = 0.0032
    contaminations = [0.002, 0.003, 0.004, 0.005]
    best_corr = -1
    best_df = None
    best_merged = None
    
    print("--- Phase 2 Anomaly Detection ---")
    for c in contaminations:
        clf = IsolationForest(contamination=c, random_state=42, n_estimators=200)
        preds = clf.fit_predict(df[features])
        df['is_anomaly'] = (preds == -1).astype(int)
        
        daily_pred = df.groupby('working time')['is_anomaly'].sum().reset_index()
        daily_pred.columns = ['date', 'pred_defects']
        daily_pred['date'] = daily_pred['date'].astype(str)
        
        merged = pd.merge(daily_pred, true_defects, on='date', how='left').fillna(0)
        
        # Calculate Correlation
        if merged['pred_defects'].std() > 0:
            corr, _ = pearsonr(merged['true_defects'], merged['pred_defects'])
        else:
            corr = 0
            
        print(f"Contamination: {c} | Correlation: {corr:.3f}")
        
        if corr > best_corr:
            best_corr = corr
            best_df = df.copy()
            best_merged = merged.copy()
            
    print(f"Best Correlation achieved: {best_corr:.3f}")
    best_df.to_csv('phase2_pseudo_labeled.csv', index=False)
    
    # Plot Phase 2 comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(best_merged['date']))
    width = 0.35
    ax.bar(x - width/2, best_merged['true_defects'], width, label='육안 검사 불량 (True Defects)', color='gray')
    ax.bar(x + width/2, best_merged['pred_defects'], width, label='Phase 2 AI 표면 불량 예측', color='blue')
    ax.set_title(f'Phase 2: 노이즈 제거 후 표면 불량 예측 상관관계 (Corr: {best_corr:.2f})')
    ax.set_xticks(x)
    ax.set_xticklabels(best_merged['date'], rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig('plots/phase2_daily_comparison.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    evaluate_and_train()
